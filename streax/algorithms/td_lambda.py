from dataclasses import dataclass
from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
from flax import core, struct

from streax.optimizers import AlphaBound, Implicit, Calibrated, ObGD, Optimizer
from streax.utils import Timestep, Transition, canonicalize_dtype
from streax.utils.typing import (
    Array,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)


@struct.dataclass(frozen=True)
class TDLambdaConfig:
    gamma: float
    trace_lambda: float
    unroll: int = struct.field(pytree_node=False, default=4)


@struct.dataclass(frozen=True)
class TDLambdaState:
    step: int
    update_step: int
    timestep: Timestep
    env_state: EnvState
    value_params: core.FrozenDict[str, Any]
    value_trace: PyTree
    value_optimizer_state: PyTree


@dataclass
class TDLambda:
    cfg: TDLambdaConfig
    env: Environment
    env_params: EnvParams
    value_network: nn.Module
    value_optimizer: Optimizer

    def _step(self, state: TDLambdaState, key: Key) -> tuple[TDLambdaState, Transition]:
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape, dtype=canonicalize_dtype(action_space.dtype)
        )

        next_obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, action, self.env_params
        )
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
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
                env_state=env_state,
            ),
            transition,
        )

    def _update_step(
        self, state: TDLambdaState, key: Key
    ) -> tuple[TDLambdaState, None]:
        state, transition = self._step(state, key)
        state = self._update(state, transition)
        return state.replace(update_step=state.update_step + 1), None

    def _update(self, state: TDLambdaState, transition: Transition) -> TDLambdaState:
        def value_fn(params):
            return self.value_network.apply(params, transition.first.obs).squeeze(-1)

        value, value_vjp = jax.vjp(value_fn, state.value_params)
        (value_grads,) = value_vjp(jnp.ones_like(value))

        def compute_td_error(next_value):
            return (
                transition.second.reward
                + self.cfg.gamma * (1.0 - transition.second.done) * next_value
                - value
            )

        reset_trace = transition.second.done
        discount = jnp.float32(self.cfg.gamma * self.cfg.trace_lambda)

        value_trace = jax.tree.map(
            lambda t, g: (discount * t + g).astype(t.dtype), state.value_trace, value_grads
        )

        if isinstance(self.value_optimizer, (Implicit, Calibrated, AlphaBound)) or (
            isinstance(self.value_optimizer, ObGD) and self.value_optimizer.cfg.exact
        ):
            interaction_trace = value_trace
            if isinstance(self.value_optimizer, (Calibrated, Implicit)):
                interaction_trace = self.value_optimizer.precondition(
                    state.value_optimizer_state, value_trace
                )

            gradient_trace = sum(
                jnp.sum(g * z)
                for g, z in zip(
                    jax.tree.leaves(value_grads),
                    jax.tree.leaves(interaction_trace),
                )
            )

            def bootstrap_value(params, obs):
                return self.value_network.apply(params, obs).squeeze(-1)

            # The JVP primal is V(s'), the bootstrap value, so we reuse it for the
            # TD error instead of a separate forward pass.
            next_value, next_grad_trace = jax.jvp(
                lambda params: bootstrap_value(params, transition.second.obs),
                (state.value_params,),
                (interaction_trace,),
            )
            td_error = compute_td_error(next_value)
            not_done = 1.0 - transition.second.done.astype(jnp.float32)
            curvature = gradient_trace - self.cfg.gamma * not_done * next_grad_trace

            value_updates, value_optimizer_state = self.value_optimizer.update(
                state.value_optimizer_state,
                value_grads,
                value_trace,
                td_error,
                curvature,
            )
        else:
            next_value = self.value_network.apply(
                state.value_params, transition.second.obs
            ).squeeze(-1)
            td_error = compute_td_error(next_value)
            value_updates, value_optimizer_state = self.value_optimizer.update(
                state.value_optimizer_state,
                value_grads,
                value_trace,
                td_error,
            )

        value_params = jax.tree.map(
            lambda p, u: p + u, state.value_params, value_updates
        )

        new_value_trace = jax.tree.map(
            lambda t: jnp.where(reset_trace, jnp.zeros_like(t), t),
            value_trace,
        )

        log_dict = {
            "value/value": next_value.mean(),
            "value/td_error": td_error.mean(),
            "value/absolute_td_error": jnp.abs(td_error).mean(),
            "value/cumulant": transition.second.reward.mean(),
        }
        lox.log(log_dict)

        return state.replace(
            value_params=value_params,
            value_trace=new_value_trace,
            value_optimizer_state=value_optimizer_state,
        )

    def init(self, key: Key) -> TDLambdaState:
        env_key, value_key = jax.random.split(key)
        obs, env_state = self.env.reset(env_key, self.env_params)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape, dtype=canonicalize_dtype(action_space.dtype)
        )
        timestep = Timestep(obs=obs, action=action, reward=0.0, done=True)
        value_params = self.value_network.init(value_key, obs)

        value_optimizer_state = self.value_optimizer.init(value_params)

        value_trace = jax.tree.map(jnp.zeros_like, value_params)

        return TDLambdaState(
            step=0,
            update_step=0,
            timestep=timestep,
            env_state=env_state,
            value_params=value_params,
            value_trace=value_trace,
            value_optimizer_state=value_optimizer_state,
        )

    def warmup(self, key: Key, state: TDLambdaState, num_steps: int) -> TDLambdaState:
        return state

    def train(self, key: Key, state: TDLambdaState, num_steps: int) -> TDLambdaState:
        keys = jax.random.split(key, num_steps)
        state, _ = jax.lax.scan(
            self._update_step,
            state,
            keys,
            unroll=self.cfg.unroll,
        )
        return state

    def evaluate(
        self, key: Key, state: TDLambdaState, num_steps: int
    ) -> TDLambdaState:
        reset_key, eval_key = jax.random.split(key)
        obs, env_state = self.env.reset(reset_key, self.env_params)

        action_space = self.env.action_space(self.env_params)
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
            env_state=env_state,
        )

        state, _ = jax.lax.scan(
            self._step,
            state,
            jax.random.split(eval_key, num_steps),
        )
        return state
