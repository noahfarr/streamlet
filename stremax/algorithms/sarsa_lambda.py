from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from stremax.optimizers import Implicit, Optimizer
from stremax.utils import Timestep, Transition, broadcast, canonicalize_dtype
from stremax.utils.typing import (
    Array,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)


@struct.dataclass(frozen=True)
class SARSALambdaConfig:
    num_envs: int
    gamma: float
    trace_lambda: float
    unroll: int = struct.field(pytree_node=False, default=2)


@struct.dataclass(frozen=True)
class SARSALambdaState:
    step: int
    update_step: int
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

    def _sample_action(
        self, key: Key, q_params: PyTree, obs: Array, step: Array
    ) -> Array:
        random_key, sample_key = jax.random.split(key)
        action_space = self.env.action_space(self.env_params)
        random_action = jax.random.randint(
            random_key,
            (self.cfg.num_envs, *action_space.shape),
            minval=0,
            maxval=action_space.n,
        )
        q_values = self.q_network.apply(q_params, obs)
        greedy_action = jnp.argmax(q_values, axis=-1)

        epsilon = self.epsilon_schedule(step)
        is_random = jax.random.uniform(sample_key, (self.cfg.num_envs,)) < epsilon
        action = jnp.where(
            broadcast(is_random, greedy_action), random_action, greedy_action
        )
        return action

    def _step(self, state: SARSALambdaState, key: Key) -> tuple[SARSALambdaState, Transition]:
        sample_key, step_key = jax.random.split(key)

        action = state.next_action

        step_keys = jax.random.split(step_key, self.cfg.num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        next_action = self._sample_action(
            sample_key, state.q_params, next_obs, state.step + self.cfg.num_envs
        )

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            aux={"next_action": next_action},
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
                next_action=next_action,
                env_state=env_state,
            ),
            transition,
        )

    def _update_step(
        self, state: SARSALambdaState, key: Key
    ) -> tuple[SARSALambdaState, None]:
        state, transition = self._step(state, key)
        state = self._update(state, transition)
        return state.replace(update_step=state.update_step + 1), None

    def _update(
        self, state: SARSALambdaState, transition: Transition
    ) -> SARSALambdaState:
        action = transition.second.action
        next_action = transition.aux["next_action"]

        def compute_td_error(params):
            q_values = self.q_network.apply(params, transition.first.obs)
            q_value = jnp.take_along_axis(
                q_values, action[:, None], axis=-1
            ).squeeze(-1)
            next_q_values = self.q_network.apply(params, transition.second.obs)
            next_value = jnp.take_along_axis(
                next_q_values, next_action[:, None], axis=-1
            ).squeeze(-1)
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

        reset = transition.second.done
        discount = jnp.broadcast_to(
            jnp.float32(self.cfg.gamma * self.cfg.trace_lambda), reset.shape
        )

        q_trace = jax.tree.map(
            lambda t, g: broadcast(discount, t) * t + g, state.q_trace, q_grads
        )

        if isinstance(self.q_optimizer, Implicit):
            gradient_trace = sum(
                jnp.sum(g * z, axis=tuple(range(1, g.ndim)))
                for g, z in zip(jax.tree.leaves(q_grads), jax.tree.leaves(q_trace))
            )

            def bootstrap_value(params, obs, a):
                return self.q_network.apply(params, obs)[a]

            def directional(obs, a, direction):
                _, jvp_value = jax.jvp(
                    lambda params: bootstrap_value(params, obs, a),
                    (state.q_params,),
                    (direction,),
                )
                return jvp_value

            next_grad_trace = jax.vmap(directional)(
                transition.second.obs, next_action, q_trace
            )
            curvature = gradient_trace - self.cfg.gamma * (
                1.0 - transition.second.done.astype(jnp.float32)
            ) * next_grad_trace
            q_updates, q_optimizer_state = self.q_optimizer.update(
                state.q_optimizer_state, q_grads, q_trace, td_error, curvature,
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

        return state.replace(
            q_params=q_params,
            q_trace=new_q_trace,
            q_optimizer_state=q_optimizer_state,
        )

    def init(self, key: Key) -> SARSALambdaState:
        env_key, q_key, action_key = jax.random.split(key, 3)
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

        next_action = self._sample_action(action_key, q_params, obs, jnp.int32(0))

        return SARSALambdaState(
            step=0,
            update_step=0,
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
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            self._update_step, state, keys, unroll=self.cfg.unroll
        )
        return state

    def evaluate(
        self, key: Key, state: SARSALambdaState, num_steps: int
    ) -> SARSALambdaState:
        reset_key, eval_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_keys, self.env_params
        )

        action_space = self.env.action_space(self.env_params)
        first_action = jnp.argmax(self.q_network.apply(state.q_params, obs), axis=-1)
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
            next_action=first_action,
            env_state=env_state,
        )

        def greedy_step(state: SARSALambdaState, key: Key):
            step_keys = jax.random.split(key, self.cfg.num_envs)
            next_obs, env_state, reward, done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_keys, state.env_state, state.next_action, self.env_params)
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
