from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from streax.optimizers import AlphaBound, Implicit, Calibrated, ObGD, Optimizer
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
class QLambdaConfig:
    num_envs: int
    gamma: float
    trace_lambda: float
    unroll: int = struct.field(pytree_node=False, default=2)


@struct.dataclass(frozen=True)
class QLambdaState:
    step: int
    update_step: int
    timestep: Timestep
    env_state: EnvState
    q_params: core.FrozenDict[str, Any]
    q_trace: PyTree
    q_optimizer_state: PyTree


@dataclass
class QLambda:
    cfg: QLambdaConfig
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    epsilon_schedule: Callable
    q_optimizer: Optimizer

    def _greedy_action(
        self, key: Key, state: QLambdaState
    ) -> tuple[QLambdaState, Array, Array]:
        q_values = self.q_network.apply(state.q_params, state.timestep.obs)
        action = jnp.argmax(q_values, axis=-1)
        return (
            state,
            action,
            jnp.zeros(self.cfg.num_envs, dtype=jnp.bool_),
        )

    def _random_action(
        self, key: Key, state: QLambdaState
    ) -> tuple[QLambdaState, Array, Array]:
        action_space = self.env.action_space(self.env_params)
        action = jax.random.randint(
            key,
            (self.cfg.num_envs, *action_space.shape),
            minval=0,
            maxval=action_space.n,
        )
        return state, action, jnp.ones(self.cfg.num_envs, dtype=jnp.bool_)

    def _epsilon_greedy_action(
        self, key: Key, state: QLambdaState
    ) -> tuple[QLambdaState, Array, Array]:
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
        self, state: QLambdaState, key: Key, *, policy: Callable
    ) -> tuple[QLambdaState, Transition]:
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
        self, state: QLambdaState, key: Key, *, policy: Callable
    ) -> tuple[QLambdaState, None]:
        step_key, _ = jax.random.split(key)
        state, transition = self._step(state, step_key, policy=policy)
        state = self._update(state, transition)
        return state.replace(update_step=state.update_step + 1), None

    def _update(
        self,
        state: QLambdaState,
        transition: Transition,
    ) -> QLambdaState:
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

        q_values, q_vjp, td_error = jax.vjp(
            compute_td_error, state.q_params, has_aux=True
        )
        batch = self.cfg.num_envs
        (q_grads,) = jax.vmap(q_vjp)(jnp.eye(batch, dtype=q_values.dtype))

        reset = transition.second.done | non_greedy
        discount = jnp.broadcast_to(
            jnp.float32(self.cfg.gamma * self.cfg.trace_lambda), reset.shape
        )

        q_trace = jax.tree.map(
            lambda t, g: broadcast(discount, t) * t + g, state.q_trace, q_grads
        )

        if isinstance(self.q_optimizer, (Implicit, Calibrated, AlphaBound)) or (
            isinstance(self.q_optimizer, ObGD) and self.q_optimizer.cfg.exact
        ):
            interaction_trace = q_trace
            if isinstance(self.q_optimizer, (Calibrated, Implicit)):
                interaction_trace = self.q_optimizer.precondition(
                    state.q_optimizer_state, q_trace
                )

            gradient_trace = sum(
                jnp.sum(g * z, axis=tuple(range(1, g.ndim)))
                for g, z in zip(
                    jax.tree.leaves(q_grads), jax.tree.leaves(interaction_trace)
                )
            )

            def bootstrap_value(params, obs):
                return self.q_network.apply(params, obs).max(axis=-1)

            def directional(obs, direction):
                _, jvp_value = jax.jvp(
                    lambda params: bootstrap_value(params, obs),
                    (state.q_params,),
                    (direction,),
                )
                return jvp_value

            next_grad_trace = jax.vmap(directional)(
                transition.second.obs, interaction_trace
            )
            not_done = 1.0 - transition.second.done.astype(jnp.float32)
            curvature = gradient_trace - self.cfg.gamma * not_done * next_grad_trace

            q_updates, q_optimizer_state = self.q_optimizer.update(
                state.q_optimizer_state,
                q_grads,
                q_trace,
                td_error,
                curvature,
            )
        else:
            q_updates, q_optimizer_state = self.q_optimizer.update(
                state.q_optimizer_state, q_grads, q_trace, td_error,
            )

        q_params = jax.tree.map(lambda p, u: p + u, state.q_params, q_updates)

        new_q_trace = jax.tree.map(
            lambda t: jnp.where(broadcast(reset, t), jnp.zeros_like(t), t), q_trace
        )

        log_dict = {
            "q_network/q_value": q_values.mean(),
            "q_network/td_error": td_error.mean(),
            "training/epsilon": self.epsilon_schedule(state.step),
            "q_trace/trace_norm": optax.global_norm(new_q_trace),
        }
        lox.log(log_dict)

        new_state = dict(
            q_params=q_params,
            q_trace=new_q_trace,
            q_optimizer_state=q_optimizer_state,
        )

        return state.replace(**new_state)

    def init(self, key: Key) -> QLambdaState:
        env_key, q_key = jax.random.split(key)
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
        q_params = self.q_network.init(q_key, obs)

        q_optimizer_state = self.q_optimizer.init(q_params, self.cfg.num_envs)

        q_trace = jax.tree.map(
            lambda p: jnp.zeros((self.cfg.num_envs, *p.shape), dtype=p.dtype),
            q_params,
        )

        state = dict(
            step=0,
            update_step=0,
            timestep=timestep,
            env_state=env_state,
            q_params=q_params,
            q_trace=q_trace,
            q_optimizer_state=q_optimizer_state,
        )

        return QLambdaState(**state)

    def warmup(self, key: Key, state: QLambdaState, num_steps: int) -> QLambdaState:
        step_keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._random_action),
            state,
            step_keys,
            unroll=self.cfg.unroll,
        )
        return state

    def train(self, key: Key, state: QLambdaState, num_steps: int) -> QLambdaState:
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            partial(self._update_step, policy=self._epsilon_greedy_action),
            state,
            keys,
            unroll=self.cfg.unroll,
        )
        return state

    def evaluate(self, key: Key, state: QLambdaState, num_steps: int) -> QLambdaState:
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
            partial(self._step, policy=self._greedy_action),
            state,
            jax.random.split(eval_key, num_steps),
        )
        return state
