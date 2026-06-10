from dataclasses import dataclass
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from streax.optimizers import Optimizer
from streax.utils import Timestep, Transition, canonicalize_dtype
from streax.utils.typing import Array, Environment, EnvParams, EnvState, Key, PyTree


@struct.dataclass(frozen=True)
class QLambdaConfig:
    gamma: float
    trace_lambda: float
    unroll: int = struct.field(pytree_node=False, default=2)


@struct.dataclass(frozen=True)
class QLambdaState:
    step: int
    timestep: Timestep
    env_state: EnvState
    q_params: core.FrozenDict[str, Any]
    q_trace: PyTree
    q_optimizer_state: PyTree
    aux_optimizer_state: PyTree


@dataclass
class QLambda:
    cfg: QLambdaConfig
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    epsilon_schedule: Callable
    q_optimizer: Optimizer
    aux_loss: Callable | None = None
    aux_optimizer: optax.GradientTransformation = optax.sgd(1e-3)

    def _env_step(
        self, state: QLambdaState, key: Key, epsilon: Array
    ) -> tuple[QLambdaState, Transition, tuple]:
        random_key, sample_key, step_key = jax.random.split(key, 3)

        action_space = self.env.action_space(self.env_params)
        random_action = jax.random.randint(
            random_key,
            action_space.shape,
            minval=0,
            maxval=action_space.n,
        )

        (q_values, intermediates), q_vjp = jax.vjp(
            lambda params: self.q_network.apply(
                params, state.timestep.obs, mutable=["intermediates"]
            ),
            state.q_params,
        )
        greedy_action = jnp.argmax(q_values, axis=-1)

        action = jnp.where(
            jax.random.uniform(sample_key, ()) < epsilon, random_action, greedy_action
        )
        non_greedy = jax.random.uniform(sample_key, ()) < epsilon & (
            random_action != greedy_action
        )

        next_obs, env_state, reward, done, info = self.env.step(
            step_key, state.env_state, action, self.env_params
        )
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            aux={"non_greedy": non_greedy},
        )

        lox.log({**info})

        return (
            state.replace(
                step=state.step + 1,
                timestep=Timestep(
                    obs=next_obs,
                    action=jnp.where(done, jnp.zeros_like(action), action),
                    reward=jnp.where(done, jnp.zeros_like(reward), reward),
                    done=done,
                ),
                env_state=env_state,
            ),
            transition,
            (q_values, intermediates, q_vjp),
        )

    def _update_step(
        self,
        state: QLambdaState,
        transition: Transition,
        linearization: tuple,
    ) -> QLambdaState:
        q_values, intermediates, q_vjp = linearization
        q_value = q_values[transition.second.action]

        num_actions = self.env.action_space(self.env_params).n
        (q_grads,) = q_vjp(
            (
                jax.nn.one_hot(
                    transition.second.action, num_actions, dtype=q_values.dtype
                ),
                jax.tree.map(jnp.zeros_like, intermediates),
            )
        )

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
            lambda params: self.q_network.apply(params, transition.second.obs).max(
                axis=-1
            ),
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

        aux_optimizer_state = state.aux_optimizer_state
        if self.aux_loss is not None:
            _, targets = self.q_network.apply(
                state.q_params, transition.second.obs, mutable=["intermediates"]
            )
            cotangents = jax.grad(self.aux_loss)(intermediates, targets, transition)
            (aux_grads,) = q_vjp((jnp.zeros_like(q_values), cotangents))
            aux_updates, aux_optimizer_state = self.aux_optimizer.update(
                aux_grads, state.aux_optimizer_state, q_params
            )
            q_params = optax.apply_updates(q_params, aux_updates)

        q_trace = jax.tree.map(
            lambda t: jnp.where(
                transition.second.done | transition.aux["non_greedy"],
                jnp.zeros_like(t),
                t,
            ),
            q_trace,
        )

        lox.log(
            {
                "q_network/q_value": q_value.mean(),
                "q_network/td_error": td_error.mean(),
            }
        )

        return state.replace(
            q_params=q_params,
            q_trace=q_trace,
            q_optimizer_state=q_optimizer_state,
            aux_optimizer_state=aux_optimizer_state,
        )

    def init(self, key: Key) -> QLambdaState:
        env_key, q_key = jax.random.split(key)
        obs, env_state = self.env.reset(env_key, self.env_params)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape,
            dtype=canonicalize_dtype(action_space.dtype),
        )
        timestep = Timestep(obs=obs, action=action, reward=0.0, done=True)
        q_params = self.q_network.init(q_key, obs)

        q_optimizer_state = self.q_optimizer.init(q_params)
        aux_optimizer_state = self.aux_optimizer.init(q_params)

        q_trace = jax.tree.map(jnp.zeros_like, q_params)

        state = dict(
            step=0,
            timestep=timestep,
            env_state=env_state,
            q_params=q_params,
            q_trace=q_trace,
            q_optimizer_state=q_optimizer_state,
            aux_optimizer_state=aux_optimizer_state,
        )

        return QLambdaState(**state)

    def train(self, key: Key, state: QLambdaState, num_steps: int) -> QLambdaState:
        def step(state, key):
            epsilon = self.epsilon_schedule(state.step)
            lox.log({"training/epsilon": epsilon})
            state, transition, linearization = self._env_step(state, key, epsilon)
            return self._update_step(state, transition, linearization), None

        state, _ = jax.lax.scan(
            step,
            state,
            jax.random.split(key, num_steps),
            unroll=self.cfg.unroll,
        )
        return state

    def evaluate(self, key: Key, state: QLambdaState, num_steps: int) -> QLambdaState:
        reset_key, eval_key = jax.random.split(key)
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
                reward=0.0,
                done=True,
            ),
            env_state=env_state,
        )

        def step(state, key):
            state, _, _ = self._env_step(state, key, 0.0)
            return state, None

        state, _ = jax.lax.scan(step, state, jax.random.split(eval_key, num_steps))
        return state
