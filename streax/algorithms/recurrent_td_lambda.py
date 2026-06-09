from dataclasses import dataclass
from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
from flax import core, struct

from streax.optimizers import Optimizer
from streax.utils import Timestep, Transition, canonicalize_dtype
from streax.utils.typing import Array, Environment, EnvParams, EnvState, Key, PyTree


@struct.dataclass(frozen=True)
class RecurrentTDLambdaConfig:
    gamma: float
    trace_lambda: float
    unroll: int = struct.field(pytree_node=False, default=4)


@struct.dataclass(frozen=True)
class RecurrentTDLambdaState:
    step: int
    update_step: int
    timestep: Timestep
    carry: PyTree
    env_state: EnvState
    value_params: core.FrozenDict[str, Any]
    value_trace: PyTree
    value_optimizer_state: PyTree


@dataclass
class RecurrentTDLambda:
    cfg: RecurrentTDLambdaConfig
    env: Environment
    env_params: EnvParams
    value_network: nn.Module
    value_optimizer: Optimizer

    def _apply(
        self, params: PyTree, carry: PyTree, timestep: Timestep
    ) -> tuple[PyTree, Array, dict]:
        return self.value_network.apply(
            params,
            carry,
            timestep.obs,
            timestep.action,
            timestep.reward,
            timestep.done,
        )

    def _reset_carry(self, carry: PyTree, done: Array) -> PyTree:
        return jax.tree.map(
            lambda leaf: jnp.where(done, jnp.zeros_like(leaf), leaf), carry
        )

    def _step(
        self, state: RecurrentTDLambdaState, key: Key
    ) -> tuple[RecurrentTDLambdaState, Transition]:
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape, dtype=canonicalize_dtype(action_space.dtype)
        )

        (carry_next, value, aux), value_vjp = jax.vjp(
            lambda p: self._apply(p, state.carry, state.timestep), state.value_params
        )
        (value_grads,) = value_vjp((
            jax.tree.map(jnp.zeros_like, carry_next),
            jnp.ones_like(value),
            jax.tree.map(jnp.zeros_like, aux),
        ))
        value = value.squeeze(-1)

        next_obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, action, self.env_params
        )
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            aux={
                "value": value,
                "value_grads": value_grads,
                "carry_next": carry_next,
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
                carry=self._reset_carry(carry_next, done),
                env_state=env_state,
            ),
            transition,
        )

    def _update_step(
        self, state: RecurrentTDLambdaState, key: Key
    ) -> tuple[RecurrentTDLambdaState, None]:
        state, transition = self._step(state, key)
        state = self._update(state, transition)
        return state.replace(update_step=state.update_step + 1), None

    def _update(
        self, state: RecurrentTDLambdaState, transition: Transition
    ) -> RecurrentTDLambdaState:
        value = transition.aux["value"]
        value_grads = transition.aux["value_grads"]
        carry_next = transition.aux["carry_next"]

        reset_trace = transition.second.done
        discount = jnp.float32(self.cfg.gamma * self.cfg.trace_lambda)

        value_trace = jax.tree.map(
            lambda t, g: (discount * t + g).astype(t.dtype),
            state.value_trace,
            value_grads,
        )

        def bootstrap_value(params):
            _, next_value, _ = self._apply(params, carry_next, transition.second)
            return next_value.squeeze(-1)

        not_done = 1.0 - transition.second.done.astype(jnp.float32)
        next_value, curvature = self.value_optimizer.bootstrap(
            state.value_optimizer_state,
            state.value_params,
            value_grads,
            value_trace,
            bootstrap_value,
            self.cfg.gamma,
            not_done,
        )
        td_error = (
            transition.second.reward
            + self.cfg.gamma * (1.0 - transition.second.done) * next_value
            - value
        )
        value_updates, value_optimizer_state = self.value_optimizer.update(
            state.value_optimizer_state,
            value_grads,
            value_trace,
            td_error,
            curvature,
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

    def init(self, key: Key) -> RecurrentTDLambdaState:
        env_key, value_key, carry_key = jax.random.split(key, 3)
        obs, env_state = self.env.reset(env_key, self.env_params)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape, dtype=canonicalize_dtype(action_space.dtype)
        )
        timestep = Timestep(
            obs=obs, action=action, reward=jnp.float32(0.0), done=jnp.bool_(True)
        )

        carry = self.value_network.initialize_carry(carry_key)
        value_params = self.value_network.init(
            value_key,
            carry,
            timestep.obs,
            timestep.action,
            timestep.reward,
            timestep.done,
        )

        value_optimizer_state = self.value_optimizer.init(value_params)

        value_trace = jax.tree.map(jnp.zeros_like, value_params)

        return RecurrentTDLambdaState(
            step=0,
            update_step=0,
            timestep=timestep,
            carry=carry,
            env_state=env_state,
            value_params=value_params,
            value_trace=value_trace,
            value_optimizer_state=value_optimizer_state,
        )

    def warmup(
        self, key: Key, state: RecurrentTDLambdaState, num_steps: int
    ) -> RecurrentTDLambdaState:
        return state

    def train(
        self, key: Key, state: RecurrentTDLambdaState, num_steps: int
    ) -> RecurrentTDLambdaState:
        keys = jax.random.split(key, num_steps)
        state, _ = jax.lax.scan(
            self._update_step,
            state,
            keys,
            unroll=self.cfg.unroll,
        )
        return state

    def evaluate(
        self, key: Key, state: RecurrentTDLambdaState, num_steps: int
    ) -> RecurrentTDLambdaState:
        reset_key, carry_key, eval_key = jax.random.split(key, 3)
        obs, env_state = self.env.reset(reset_key, self.env_params)

        action_space = self.env.action_space(self.env_params)
        state = state.replace(
            step=0,
            timestep=Timestep(
                obs=obs,
                action=jnp.zeros(
                    action_space.shape, dtype=canonicalize_dtype(action_space.dtype)
                ),
                reward=jnp.float32(0.0),
                done=jnp.bool_(True),
            ),
            carry=self.value_network.initialize_carry(carry_key),
            env_state=env_state,
        )

        state, _ = jax.lax.scan(
            self._step,
            state,
            jax.random.split(eval_key, num_steps),
        )
        return state
