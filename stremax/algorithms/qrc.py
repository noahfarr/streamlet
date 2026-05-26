from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from stremax.optimizers import Optimizer
from stremax.utils import Timestep, Transition, broadcast
from stremax.utils.typing import (
    Array,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)


@struct.dataclass(frozen=True)
class QRCConfig:
    num_envs: int
    gamma: float
    trace_lambda: float
    gradient_correction: bool
    regularization_coefficient: float
    unroll: int = struct.field(pytree_node=False)


@struct.dataclass(frozen=True)
class QRCState:
    step: int
    update_step: int
    timestep: Timestep
    env_state: EnvState
    q_params: core.FrozenDict[str, Any]
    h_params: core.FrozenDict[str, Any]
    q_optimizer_state: PyTree
    h_optimizer_state: PyTree
    q_trace: PyTree
    h_trace: PyTree
    bias_trace: Array


@dataclass
class QRC:
    cfg: QRCConfig
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    h_network: nn.Module
    q_optimizer: Optimizer
    h_optimizer: Optimizer
    epsilon_schedule: Callable

    def _greedy_action(
        self, key: Key, state: QRCState
    ) -> tuple[QRCState, Array, Array]:
        q_values = self.q_network.apply(state.q_params, state.timestep.obs)
        action = jnp.argmax(q_values, axis=-1)
        return (
            state,
            action,
            jnp.zeros(self.cfg.num_envs, dtype=jnp.bool_),
        )

    def _random_action(
        self, key: Key, state: QRCState
    ) -> tuple[QRCState, Array, Array]:
        action_space = self.env.action_space(self.env_params)
        action = jax.random.randint(
            key,
            (self.cfg.num_envs, *action_space.shape),
            minval=0,
            maxval=action_space.n,
        )
        return state, action, jnp.ones(self.cfg.num_envs, dtype=jnp.bool_)

    def _epsilon_greedy_action(
        self, key: Key, state: QRCState
    ) -> tuple[QRCState, Array, Array]:
        random_key, greedy_key, sample_key = jax.random.split(key, 3)
        state, random_action, _ = self._random_action(random_key, state)
        state, greedy_action, _ = self._greedy_action(greedy_key, state)

        epsilon = self.epsilon_schedule(state.step)
        is_random = jax.random.uniform(sample_key, (self.cfg.num_envs,)) < epsilon
        action = jnp.where(
            broadcast(is_random, greedy_action), random_action, greedy_action
        )
        non_greedy = is_random & (random_action != greedy_action)
        return state, action, non_greedy

    def _step(
        self, state: QRCState, key: Key, *, policy: Callable
    ) -> tuple[QRCState, Transition]:
        action_key, step_key = jax.random.split(key)
        state, action, non_greedy = policy(action_key, state)

        step_keys = jax.random.split(step_key, self.cfg.num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            aux={"non_greedy": non_greedy},
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
        self, state: QRCState, key: Key, *, policy: Callable
    ) -> tuple[QRCState, None]:
        step_key, _ = jax.random.split(key)
        state, transition = self._step(state, step_key, policy=policy)
        state = self._update(state, transition)
        return state.replace(update_step=state.update_step + 1), None

    def _update(self, state: QRCState, transition: Transition) -> QRCState:
        action = transition.second.action
        non_greedy = transition.aux["non_greedy"]

        def compute_td_error(params):
            q_values = self.q_network.apply(params, transition.first.obs)
            q_value = jnp.take_along_axis(
                q_values, action[:, None], axis=-1
            ).squeeze(-1)
            next_q_values = self.q_network.apply(params, transition.second.obs)
            next_value = next_q_values.max(axis=-1)
            td_error = (
                transition.second.reward
                + self.cfg.gamma * next_value * (1.0 - transition.second.done)
                - q_value
            )
            return q_value, td_error

        (q_values, td_errors), q_vjp = jax.vjp(
            compute_td_error, state.q_params, has_aux=False
        )

        batch = self.cfg.num_envs
        eye = jnp.eye(batch, dtype=q_values.dtype)
        zeros = jnp.zeros((batch, batch), dtype=q_values.dtype)
        (q_grads,) = jax.vmap(q_vjp)((eye, zeros))
        (td_grads,) = jax.vmap(q_vjp)((zeros, eye))

        def compute_h(params):
            h_values = self.h_network.apply(params, transition.first.obs)
            h_value = jnp.take_along_axis(
                h_values, action[:, None], axis=-1
            ).squeeze(-1)
            return h_value

        h_values, h_vjp = jax.vjp(compute_h, state.h_params)
        (h_grads,) = jax.vmap(h_vjp)(jnp.eye(batch, dtype=h_values.dtype))

        reset_trace = transition.second.done | non_greedy
        discount = jnp.float32(self.cfg.gamma * self.cfg.trace_lambda)

        q_trace = jax.tree.map(
            lambda t, g: discount * t + g, state.q_trace, q_grads
        )
        h_trace = jax.tree.map(
            lambda t, g: discount * t + g, state.h_trace, h_grads
        )
        bias_trace = discount * state.bias_trace + h_values

        q_updates = jax.tree.map(
            lambda td_g: -broadcast(bias_trace, td_g) * td_g, td_grads
        )

        if self.cfg.gradient_correction:
            q_updates = jax.tree.map(
                lambda update, t, g: update
                + broadcast(td_errors, t) * t
                - broadcast(h_values, g) * g,
                q_updates,
                q_trace,
                q_grads,
            )

        h_updates = jax.tree.map(
            lambda t, g, p: broadcast(td_errors, t) * t
            - broadcast(h_values, g) * g
            - self.cfg.regularization_coefficient * p[None],
            h_trace,
            h_grads,
            state.h_params,
        )

        q_grads_final = jax.tree.map(lambda x: -x.mean(axis=0), q_updates)
        h_grads_final = jax.tree.map(lambda x: -x.mean(axis=0), h_updates)

        q_param_updates, q_optimizer_state = self.q_optimizer.update(
            state.q_optimizer_state, q_grads_final
        )
        h_param_updates, h_optimizer_state = self.h_optimizer.update(
            state.h_optimizer_state, h_grads_final
        )
        q_params = jax.tree.map(lambda p, u: p + u, state.q_params, q_param_updates)
        h_params = jax.tree.map(lambda p, u: p + u, state.h_params, h_param_updates)

        new_q_trace = jax.tree.map(
            lambda t: jnp.where(broadcast(reset_trace, t), jnp.zeros_like(t), t),
            q_trace,
        )
        new_h_trace = jax.tree.map(
            lambda t: jnp.where(broadcast(reset_trace, t), jnp.zeros_like(t), t),
            h_trace,
        )
        new_bias_trace = jnp.where(reset_trace, jnp.zeros_like(bias_trace), bias_trace)

        q_target = q_values + td_errors
        explained_variance = 1 - jnp.var(td_errors) / (jnp.var(q_target) + 1e-8)
        lox.log(
            {
                "q_network/q_value": q_values.mean(),
                "q_network/td_error": td_errors.mean(),
                "q_network/explained_variance": explained_variance,
                "q_network/gradient_norm": optax.global_norm(q_grads_final),
                "q_network/update_norm": optax.global_norm(q_param_updates),
                "h_network/h_value": h_values.mean(),
                "h_network/gradient_norm": optax.global_norm(h_grads_final),
                "h_network/update_norm": optax.global_norm(h_param_updates),
                "h_network/bias_trace": bias_trace.mean(),
                "training/epsilon": self.epsilon_schedule(state.step),
                "q_trace/trace_norm": optax.global_norm(new_q_trace),
                "h_trace/trace_norm": optax.global_norm(new_h_trace),
            }
        )

        return state.replace(
            q_params=q_params,
            h_params=h_params,
            q_optimizer_state=q_optimizer_state,
            h_optimizer_state=h_optimizer_state,
            q_trace=new_q_trace,
            h_trace=new_h_trace,
            bias_trace=new_bias_trace,
        )

    def init(self, key: Key) -> QRCState:
        env_key, q_key, h_key = jax.random.split(key, 3)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)

        q_params = self.q_network.init(q_key, obs)
        h_params = self.h_network.init(h_key, obs)
        q_optimizer_state = self.q_optimizer.init(q_params, self.cfg.num_envs)
        h_optimizer_state = self.h_optimizer.init(h_params, self.cfg.num_envs)

        q_trace = jax.tree.map(
            lambda p: jnp.zeros((self.cfg.num_envs, *p.shape), dtype=p.dtype),
            q_params,
        )
        h_trace = jax.tree.map(
            lambda p: jnp.zeros((self.cfg.num_envs, *p.shape), dtype=p.dtype),
            h_params,
        )
        bias_trace = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)

        return QRCState(
            step=0,
            update_step=0,
            timestep=timestep,
            env_state=env_state,
            q_params=q_params,
            h_params=h_params,
            q_optimizer_state=q_optimizer_state,
            h_optimizer_state=h_optimizer_state,
            q_trace=q_trace,
            h_trace=h_trace,
            bias_trace=bias_trace,
        )

    def warmup(self, key: Key, state: QRCState, num_steps: int) -> QRCState:
        step_keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._random_action),
            state,
            step_keys,
            unroll=self.cfg.unroll,
        )
        return state

    def train(self, key: Key, state: QRCState, num_steps: int) -> QRCState:
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            partial(self._update_step, policy=self._epsilon_greedy_action),
            state,
            keys,
            unroll=self.cfg.unroll,
        )
        return state

    def evaluate(
        self, key: Key, state: QRCState, num_steps: int, deterministic: bool = True
    ) -> QRCState:
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
                    (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
                ),
                reward=jnp.zeros(self.cfg.num_envs),
                done=jnp.ones(self.cfg.num_envs, dtype=jnp.bool_),
            ),
            env_state=env_state,
        )

        policy = self._greedy_action if deterministic else self._epsilon_greedy_action
        state, _ = jax.lax.scan(
            partial(self._step, policy=policy),
            state,
            jax.random.split(eval_key, num_steps),
        )
        return state
