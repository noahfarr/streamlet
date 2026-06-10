from dataclasses import dataclass
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
from flax import core, struct

from streax.optimizers import Optimizer
from streax.utils import Timestep, Transition, canonicalize_dtype
from streax.utils.typing import Array, Environment, EnvParams, EnvState, Key, PyTree


@struct.dataclass(frozen=True)
class RecurrentQLambdaConfig:
    gamma: float
    trace_lambda: float
    unroll: int = struct.field(pytree_node=False, default=4)


@struct.dataclass(frozen=True)
class RecurrentQLambdaState:
    step: int
    timestep: Timestep
    carry: PyTree
    env_state: EnvState
    q_params: core.FrozenDict[str, Any]
    q_trace: PyTree
    q_optimizer_state: PyTree


@dataclass
class RecurrentQLambda:
    cfg: RecurrentQLambdaConfig
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    epsilon_schedule: Callable
    q_optimizer: Optimizer
    aux_loss: Callable | None = None

    def _env_step(
        self, state: RecurrentQLambdaState, key: Key, epsilon: Array
    ) -> tuple[RecurrentQLambdaState, Transition]:
        random_key, sample_key, step_key = jax.random.split(key, 3)

        action_space = self.env.action_space(self.env_params)
        random_action = jax.random.randint(
            random_key,
            action_space.shape,
            minval=0,
            maxval=action_space.n,
        )

        ((next_carry, q_values), intermediates), q_vjp = jax.vjp(
            lambda params: self.q_network.apply(
                params, state.carry, *state.timestep, mutable=["intermediates"]
            ),
            state.q_params,
        )
        greedy_action = jnp.argmax(q_values, axis=-1)

        explore = jax.random.uniform(sample_key, ()) < epsilon
        action = jnp.where(explore, random_action, greedy_action)
        non_greedy = explore & (random_action != greedy_action)

        q_value = q_values[action]
        (q_grads,) = q_vjp((
            (
                jax.tree.map(jnp.zeros_like, next_carry),
                jax.nn.one_hot(action, action_space.n, dtype=q_values.dtype),
            ),
            jax.tree.map(jnp.zeros_like, intermediates),
        ))

        next_obs, env_state, reward, done, info = self.env.step(
            step_key, state.env_state, action, self.env_params
        )
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            aux={
                "non_greedy": non_greedy,
                "q_value": q_value,
                "q_values": q_values,
                "q_grads": q_grads,
                "intermediates": intermediates,
                "q_vjp": q_vjp,
                "carry": state.carry,
                "next_carry": next_carry,
            },
        )

        return (
            state.replace(
                step=state.step + 1,
                timestep=Timestep(
                    obs=next_obs,
                    action=jnp.where(done, jnp.zeros_like(action), action),
                    reward=jnp.where(done, jnp.zeros_like(reward), reward),
                    done=done,
                ),
                carry=next_carry,
                env_state=env_state,
            ),
            transition,
        )

    def _update_step(
        self,
        state: RecurrentQLambdaState,
        transition: Transition,
    ) -> RecurrentQLambdaState:
        non_greedy = transition.aux["non_greedy"]
        q_value = transition.aux["q_value"]
        q_values = transition.aux["q_values"]
        q_grads = transition.aux["q_grads"]
        intermediates = transition.aux["intermediates"]
        q_vjp = transition.aux["q_vjp"]
        next_carry = transition.aux["next_carry"]

        q_trace = jax.tree.map(
            lambda trace, grad: self.cfg.gamma * self.cfg.trace_lambda * trace + grad,
            state.q_trace,
            q_grads,
        )

        next_q_value, curvature = self.q_optimizer.bootstrap(
            state.q_optimizer_state,
            state.q_params,
            q_grads,
            q_trace,
            lambda params: self.q_network.apply(params, next_carry, *transition.second)[
                1
            ].max(axis=-1),
            self.cfg.gamma,
            1.0 - transition.second.done.astype(jnp.float32),
        )
        td_error = (
            transition.second.reward
            + self.cfg.gamma * next_q_value * (1.0 - transition.second.done)
            - q_value
        )
        q_updates, q_optimizer_state = self.q_optimizer.update(
            state.q_optimizer_state,
            q_grads,
            q_trace,
            td_error,
            curvature,
        )

        q_params = jax.tree.map(lambda p, u: p + u, state.q_params, q_updates)

        if self.aux_loss is not None:
            _, next_intermediates = self.q_network.apply(
                state.q_params, next_carry, *transition.second, mutable=["intermediates"]
            )
            transition = transition.replace(
                aux={**transition.aux, "next_intermediates": next_intermediates}
            )
            cotangents = jax.grad(
                lambda i: self.aux_loss(
                    transition.replace(aux={**transition.aux, "intermediates": i})
                )
            )(intermediates)
            (aux_grads,) = q_vjp((
                (jax.tree.map(jnp.zeros_like, next_carry), jnp.zeros_like(q_values)),
                cotangents,
            ))
            q_params = jax.tree.map(lambda p, g: p - g, q_params, aux_grads)

        q_trace = jax.tree.map(
            lambda t: jnp.where(
                transition.second.done | non_greedy, jnp.zeros_like(t), t
            ),
            q_trace,
        )

        lox.log(
            {
                "q_network/q_value": q_value.mean(),
                "q_network/td_error": td_error.mean(),
                "q_network/absolute_td_error": jnp.abs(td_error).mean(),
            }
        )

        return state.replace(
            q_params=q_params,
            q_trace=q_trace,
            q_optimizer_state=q_optimizer_state,
        )

    def init(self, key: Key) -> RecurrentQLambdaState:
        env_key, q_key, carry_key = jax.random.split(key, 3)
        obs, env_state = self.env.reset(env_key, self.env_params)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape,
            dtype=canonicalize_dtype(action_space.dtype),
        )
        timestep = Timestep(
            obs=obs, action=action, reward=jnp.float32(0.0), done=jnp.bool_(True)
        )

        carry = self.q_network.initialize_carry(carry_key)
        q_params = self.q_network.init(q_key, carry, *timestep)

        q_optimizer_state = self.q_optimizer.init(q_params)

        q_trace = jax.tree.map(jnp.zeros_like, q_params)

        return RecurrentQLambdaState(
            step=0,
            timestep=timestep,
            carry=carry,
            env_state=env_state,
            q_params=q_params,
            q_trace=q_trace,
            q_optimizer_state=q_optimizer_state,
        )

    def train(
        self, key: Key, state: RecurrentQLambdaState, num_steps: int
    ) -> RecurrentQLambdaState:
        def step(state, key):
            epsilon = self.epsilon_schedule(state.step)
            lox.log({"training/epsilon": epsilon})
            state, transition = self._env_step(state, key, epsilon)
            return self._update_step(state, transition), None

        state, _ = jax.lax.scan(
            step,
            state,
            jax.random.split(key, num_steps),
            unroll=self.cfg.unroll,
        )
        return state

    def evaluate(
        self, key: Key, state: RecurrentQLambdaState, num_steps: int
    ) -> RecurrentQLambdaState:
        reset_key, carry_key, eval_key = jax.random.split(key, 3)
        obs, env_state = self.env.reset(reset_key, self.env_params)

        action_space = self.env.action_space(self.env_params)
        state = state.replace(
            step=0,
            timestep=Timestep(
                obs=obs,
                action=jnp.zeros(
                    action_space.shape,
                    dtype=canonicalize_dtype(action_space.dtype),
                ),
                reward=jnp.float32(0.0),
                done=jnp.bool_(True),
            ),
            carry=self.q_network.initialize_carry(carry_key),
            env_state=env_state,
        )

        def step(state, key):
            state, _ = self._env_step(state, key, 0.0)
            return state, None

        state, _ = jax.lax.scan(step, state, jax.random.split(eval_key, num_steps))
        return state
