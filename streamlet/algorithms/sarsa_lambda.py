from dataclasses import dataclass
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
from flax import core, struct

from streamlet.optimizers import Optimizer
from streamlet.utils import Timestep, Transition, canonicalize_dtype
from streamlet.utils.typing import (
    Array,
    Discrete,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)


@struct.dataclass(frozen=True)
class SARSALambdaConfig:
    gamma: float
    trace_lambda: float
    unroll: int = struct.field(pytree_node=False, default=4)


@struct.dataclass(frozen=True)
class SARSALambdaState:
    step: int
    timestep: Timestep
    next_action: Array
    env_state: EnvState
    q_params: core.FrozenDict[str, Any]
    q_trace: PyTree
    q_optimizer_state: PyTree


@dataclass
class SARSALambda:
    cfg: SARSALambdaConfig
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    epsilon_schedule: Callable
    q_optimizer: Optimizer
    aux_loss: Callable | None = None

    def __post_init__(self):
        action_space = self.env.action_space(self.env_params)
        assert isinstance(action_space, Discrete), (
            "SARSALambda requires a discrete action space, got "
            f"{type(action_space).__name__}."
        )
        assert 0.0 <= self.cfg.gamma <= 1.0, (
            f"gamma must be in [0, 1], got {self.cfg.gamma}."
        )
        assert 0.0 <= self.cfg.trace_lambda <= 1.0, (
            f"trace_lambda must be in [0, 1], got {self.cfg.trace_lambda}."
        )
        assert self.cfg.unroll >= 1, (
            f"unroll must be >= 1, got {self.cfg.unroll}."
        )

    def env_step(
        self, state: SARSALambdaState, key: Key
    ) -> tuple[SARSALambdaState, Transition]:
        random_key, sample_key, step_key = jax.random.split(key, 3)

        action = state.next_action

        (q_values, intermediates), q_vjp = jax.vjp(
            lambda params: self.q_network.apply(
                params, state.timestep.obs, mutable=["intermediates"]
            ),
            state.q_params,
        )

        next_obs, env_state, reward, done, info = self.env.step(
            step_key, state.env_state, action, self.env_params
        )
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        action_space = self.env.action_space(self.env_params)
        random_action = jax.random.randint(
            random_key,
            action_space.shape,
            minval=0,
            maxval=action_space.n,
        )
        next_q_values = self.q_network.apply(state.q_params, next_obs)
        greedy_action = jnp.argmax(next_q_values, axis=-1)

        epsilon = self.epsilon_schedule(state.step + 1)
        explore = jax.random.uniform(sample_key, ()) < epsilon
        next_action = jnp.where(explore, random_action, greedy_action)

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            aux={
                "next_action": next_action,
                "q_values": q_values,
                "intermediates": intermediates,
                "q_vjp": q_vjp,
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
                next_action=next_action,
                env_state=env_state,
            ),
            transition,
        )

    def update_step(
        self, state: SARSALambdaState, transition: Transition
    ) -> SARSALambdaState:
        action = transition.second.action
        next_action = transition.aux["next_action"]
        q_values = transition.aux["q_values"]
        intermediates = transition.aux["intermediates"]
        q_vjp = transition.aux["q_vjp"]
        q_value = q_values[action]

        num_actions = self.env.action_space(self.env_params).n
        (q_grads,) = q_vjp(
            (
                jax.nn.one_hot(action, num_actions, dtype=q_values.dtype),
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
            lambda params: self.q_network.apply(params, transition.second.obs)[
                next_action
            ],
            self.cfg.gamma,
            1.0 - transition.second.done.astype(jnp.float32),
        )
        td_error = (
            transition.second.reward
            + self.cfg.gamma * next_q_value * (1.0 - transition.second.done)
            - q_value
        )
        q_updates, q_optimizer_state = self.q_optimizer.update(
            state.q_optimizer_state, q_grads, q_trace, td_error, curvature,
        )

        q_params = jax.tree.map(lambda p, u: p + u, state.q_params, q_updates)

        if self.aux_loss is not None:
            _, next_intermediates = self.q_network.apply(
                state.q_params, transition.second.obs, mutable=["intermediates"]
            )
            transition = transition.replace(
                aux={**transition.aux, "next_intermediates": next_intermediates}
            )
            cotangents = jax.grad(
                lambda i: self.aux_loss(
                    transition.replace(aux={**transition.aux, "intermediates": i})
                )
            )(intermediates)
            (aux_grads,) = q_vjp((jnp.zeros_like(q_values), cotangents))
            q_params = jax.tree.map(lambda p, g: p - g, q_params, aux_grads)

        q_trace = jax.tree.map(
            lambda t: jnp.where(transition.second.done, jnp.zeros_like(t), t), q_trace
        )

        lox.log(
            {
                "q_network/q_value": q_value.mean(),
                "q_network/td_error": td_error.mean(),
                "q_network/absolute_td_error": jnp.abs(td_error).mean(),
                "training/epsilon": self.epsilon_schedule(state.step),
            }
        )

        return state.replace(
            q_params=q_params,
            q_trace=q_trace,
            q_optimizer_state=q_optimizer_state,
        )

    def init(self, key: Key) -> SARSALambdaState:
        env_key, q_key, action_key = jax.random.split(key, 3)
        obs, env_state = self.env.reset(env_key, self.env_params)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape, dtype=canonicalize_dtype(action_space.dtype)
        )
        timestep = Timestep(obs=obs, action=action, reward=0.0, done=True)
        q_params = self.q_network.init(q_key, obs)

        q_optimizer_state = self.q_optimizer.init(q_params)

        q_trace = jax.tree.map(jnp.zeros_like, q_params)

        next_action = jax.random.randint(
            action_key, action_space.shape, minval=0, maxval=action_space.n
        )

        return SARSALambdaState(
            step=0,
            timestep=timestep,
            next_action=next_action,
            env_state=env_state,
            q_params=q_params,
            q_trace=q_trace,
            q_optimizer_state=q_optimizer_state,
        )

    def train(
        self, key: Key, state: SARSALambdaState, num_steps: int
    ) -> SARSALambdaState:
        def step(state, key):
            state, transition = self.env_step(state, key)
            return self.update_step(state, transition), None

        state, _ = jax.lax.scan(
            step,
            state,
            jax.random.split(key, num_steps),
            unroll=self.cfg.unroll,
        )
        return state

    def evaluate(
        self, key: Key, state: SARSALambdaState, num_steps: int
    ) -> SARSALambdaState:
        reset_key, eval_key = jax.random.split(key)
        obs, env_state = self.env.reset(reset_key, self.env_params)

        action_space = self.env.action_space(self.env_params)
        first_action = jnp.argmax(self.q_network.apply(state.q_params, obs), axis=-1)
        state = state.replace(
            step=0,
            timestep=Timestep(
                obs=obs,
                action=jnp.zeros(
                    action_space.shape, dtype=canonicalize_dtype(action_space.dtype)
                ),
                reward=0.0,
                done=True,
            ),
            next_action=first_action,
            env_state=env_state,
        )

        def greedy_step(state: SARSALambdaState, key: Key):
            next_obs, env_state, reward, done, info = self.env.step(
                key, state.env_state, state.next_action, self.env_params
            )
            next_action = jnp.argmax(
                self.q_network.apply(state.q_params, next_obs), axis=-1
            )
            return (
                state.replace(
                    timestep=Timestep(
                        obs=next_obs,
                        action=state.next_action,
                        reward=jnp.asarray(reward, dtype=jnp.float32),
                        done=jnp.asarray(done, dtype=jnp.bool_),
                    ),
                    next_action=next_action,
                    env_state=env_state,
                ),
                None,
            )

        state, _ = jax.lax.scan(
            greedy_step, state, jax.random.split(eval_key, num_steps)
        )
        return state
