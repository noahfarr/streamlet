from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from streax.optimizers import Implicit, Measured, MeasuredMode, ObGD, Optimizer
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
class RecurrentTDLambdaConfig:
    num_envs: int
    gamma: float
    trace_lambda: float
    unroll: int = struct.field(pytree_node=False, default=2)


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
    ) -> tuple[PyTree, Array]:
        # The recurrent network receives observation, action, reward and done as
        # separate positional arguments; it returns the advanced carry and value.
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
            lambda leaf: jnp.where(broadcast(done, leaf), jnp.zeros_like(leaf), leaf),
            carry,
        )

    def _step(
        self, state: RecurrentTDLambdaState, key: Key
    ) -> tuple[RecurrentTDLambdaState, Transition]:
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape), dtype=canonicalize_dtype(action_space.dtype)
        )

        # TD takes no action, but the carry must still advance through the obs.
        carry_next, _ = self._apply(state.value_params, state.carry, state.timestep)

        step_keys = jax.random.split(key, self.cfg.num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            aux={"carry_in": state.carry, "carry_next": carry_next},
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
                carry=self._reset_carry(carry_next, done),
                env_state=env_state,
            ),
            transition,
        )

    def _update_step(
        self, state: RecurrentTDLambdaState, key: Key
    ) -> tuple[RecurrentTDLambdaState, None]:
        step_key, _ = jax.random.split(key)
        state, transition = self._step(state, step_key)
        state = self._update(state, transition)
        return state.replace(update_step=state.update_step + 1), None

    def _update(
        self, state: RecurrentTDLambdaState, transition: Transition
    ) -> RecurrentTDLambdaState:
        carry_in = transition.aux["carry_in"]
        carry_next = transition.aux["carry_next"]

        def compute_td_error(params):
            _, v = self._apply(params, carry_in, transition.first)
            value = v.squeeze(-1)
            _, next_v = self._apply(params, carry_next, transition.second)
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

        if isinstance(self.value_optimizer, (Implicit, Measured)) or (
            isinstance(self.value_optimizer, ObGD) and self.value_optimizer.cfg.exact
        ):
            # The interaction must use the same preconditioned trace direction
            # P z that the optimizer's update applies, so X = (g - gamma g')(P z).
            # Implicit has no preconditioner; Measured optionally applies one.
            interaction_trace = value_trace
            if isinstance(self.value_optimizer, Measured):
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

            def bootstrap_value(params, carry, timestep):
                _, v = self._apply(params, carry, timestep)
                return v.squeeze(-1)

            def directional(carry, timestep, direction):
                _, jvp_value = jax.jvp(
                    lambda params: bootstrap_value(params, carry, timestep),
                    (state.value_params,),
                    (direction,),
                )
                return jvp_value

            next_grad_trace = jax.vmap(directional)(
                carry_next, transition.second, interaction_trace
            )
            not_done = 1.0 - transition.second.done.astype(jnp.float32)
            curvature = gradient_trace - self.cfg.gamma * not_done * next_grad_trace

            squared_grad_norm = None
            if isinstance(self.value_optimizer, Measured) and (
                self.value_optimizer.cfg.mode is MeasuredMode.FROBENIUS
            ):
                _, bootstrap_vjp = jax.vjp(
                    lambda params: bootstrap_value(
                        params, carry_next, transition.second
                    ),
                    state.value_params,
                )
                (next_grads,) = jax.vmap(bootstrap_vjp)(
                    jnp.eye(batch, dtype=values.dtype)
                )
                grad_diff = jax.tree.map(
                    lambda g, gn: g - self.cfg.gamma * broadcast(not_done, gn) * gn,
                    value_grads,
                    next_grads,
                )
                squared_grad_norm = sum(
                    jnp.sum(jnp.square(b), axis=tuple(range(1, b.ndim)))
                    for b in jax.tree.leaves(grad_diff)
                )

            value_updates, value_optimizer_state = self.value_optimizer.update(
                state.value_optimizer_state,
                value_grads,
                value_trace,
                td_error,
                curvature,
                squared_grad_norm,
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

        _, next_v = self._apply(value_params, carry_next, transition.second)
        next_value = next_v.squeeze(-1)

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

    def init(self, key: Key) -> RecurrentTDLambdaState:
        env_key, value_key, carry_key = jax.random.split(key, 3)
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

        carry = self.value_network.initialize_carry(carry_key, self.cfg.num_envs)
        value_params = self.value_network.init(
            value_key,
            carry,
            timestep.obs,
            timestep.action,
            timestep.reward,
            timestep.done,
        )

        value_optimizer_state = self.value_optimizer.init(
            value_params, self.cfg.num_envs
        )

        value_trace = jax.tree.map(
            lambda p: jnp.zeros((self.cfg.num_envs, *p.shape), dtype=p.dtype),
            value_params,
        )

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
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
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
            carry=self.value_network.initialize_carry(carry_key, self.cfg.num_envs),
            env_state=env_state,
        )

        state, _ = jax.lax.scan(
            self._step,
            state,
            jax.random.split(eval_key, num_steps),
        )
        return state
