from dataclasses import dataclass
from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from streax.optimizers import Implicit, Calibrated, ObGD, Optimizer
from streax.utils import Timestep, Transition, broadcast, canonicalize_dtype
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
    num_envs: int
    gamma: float
    trace_lambda: float
    unroll: int = struct.field(pytree_node=False, default=2)


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
            (self.cfg.num_envs, *action_space.shape), dtype=canonicalize_dtype(action_space.dtype)
        )

        step_keys = jax.random.split(key, self.cfg.num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
        )

        return (
            state.replace(
                step=state.step + self.cfg.num_envs,
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
        step_key, _ = jax.random.split(key)
        state, transition = self._step(state, step_key)
        state = self._update(state, transition)
        return state.replace(update_step=state.update_step + 1), None

    def _update(self, state: TDLambdaState, transition: Transition) -> TDLambdaState:
        def compute_td_error(params):
            v = self.value_network.apply(params, transition.first.obs)
            value = v.squeeze(-1)
            next_v = self.value_network.apply(params, transition.second.obs)
            next_value = next_v.squeeze(-1)
            td_error = (
                transition.second.reward
                + self.cfg.gamma * (1.0 - transition.second.done) * next_value
                - value
            )
            return value, td_error

        values, value_vjp, td_error = jax.vjp(
            compute_td_error, state.value_params, has_aux=True
        )
        batch = self.cfg.num_envs
        (value_grads,) = jax.vmap(value_vjp)(jnp.eye(batch, dtype=values.dtype))

        reset_trace = transition.second.done
        discount = jnp.broadcast_to(
            jnp.float32(self.cfg.gamma * self.cfg.trace_lambda), reset_trace.shape
        )

        value_trace = jax.tree.map(
            lambda t, g: broadcast(discount, t) * t + g, state.value_trace, value_grads
        )

        if isinstance(self.value_optimizer, (Implicit, Calibrated)) or (
            isinstance(self.value_optimizer, ObGD) and self.value_optimizer.cfg.exact
        ):
            interaction_trace = value_trace
            if isinstance(self.value_optimizer, (Calibrated, Implicit)):
                interaction_trace = self.value_optimizer.precondition(
                    state.value_optimizer_state, value_trace
                )

            gradient_trace = sum(
                jnp.sum(g * z, axis=tuple(range(1, g.ndim)))
                for g, z in zip(
                    jax.tree.leaves(value_grads),
                    jax.tree.leaves(interaction_trace),
                )
            )

            def bootstrap_value(params, obs):
                return self.value_network.apply(params, obs).squeeze(-1)

            def directional(obs, direction):
                _, jvp_value = jax.jvp(
                    lambda params: bootstrap_value(params, obs),
                    (state.value_params,),
                    (direction,),
                )
                return jvp_value

            next_grad_trace = jax.vmap(directional)(
                transition.second.obs, interaction_trace
            )
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
            lambda t: jnp.where(broadcast(reset_trace, t), jnp.zeros_like(t), t),
            value_trace,
        )

        next_value = self.value_network.apply(
            value_params, transition.second.obs
        ).squeeze(-1)

        log_dict = {
            "value/value": next_value.mean(),
            "value/td_error": td_error.mean(),
            "value/cumulant": transition.second.reward.mean(),
            "value_trace/trace_norm": optax.global_norm(new_value_trace),
        }
        lox.log(log_dict)

        return state.replace(
            value_params=value_params,
            value_trace=new_value_trace,
            value_optimizer_state=value_optimizer_state,
        )

    def init(self, key: Key) -> TDLambdaState:
        env_key, value_key = jax.random.split(key)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape), dtype=canonicalize_dtype(action_space.dtype)
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        value_params = self.value_network.init(value_key, obs)

        value_optimizer_state = self.value_optimizer.init(
            value_params, self.cfg.num_envs
        )

        value_trace = jax.tree.map(
            lambda p: jnp.zeros((self.cfg.num_envs, *p.shape), dtype=p.dtype),
            value_params,
        )

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
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
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
        reset_keys = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_keys, self.env_params
        )

        action_space = self.env.action_space(self.env_params)
        state = state.replace(
            step=0,
            timestep=Timestep(
                obs=obs,
                action=jnp.zeros(
                    (self.cfg.num_envs, *action_space.shape), dtype=canonicalize_dtype(action_space.dtype)
                ),
                reward=jnp.zeros(self.cfg.num_envs),
                done=jnp.ones(self.cfg.num_envs, dtype=jnp.bool_),
            ),
            env_state=env_state,
        )

        state, _ = jax.lax.scan(
            self._step,
            state,
            jax.random.split(eval_key, num_steps),
        )
        return state
