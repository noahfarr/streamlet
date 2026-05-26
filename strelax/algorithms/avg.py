from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from strelax.optimizers import Optimizer
from strelax.utils import Timestep, Transition, TDErrorScalerState, broadcast
from strelax.utils.typing import (
    Array,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)


@struct.dataclass(frozen=True)
class AVGConfig:
    num_envs: int
    gamma: float
    alpha: float
    trace_lambda: float = 0.0
    unroll: int = struct.field(pytree_node=False, default=2)


@struct.dataclass(frozen=True)
class AVGState:
    step: int
    update_step: int
    obs: Array
    done: Array
    env_state: EnvState
    actor_params: core.FrozenDict[str, Any]
    actor_optimizer_state: PyTree
    critic_params: core.FrozenDict[str, Any]
    critic_optimizer_state: PyTree
    critic_trace: PyTree
    td_scaler: TDErrorScalerState


@dataclass
class AVG:
    cfg: AVGConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    actor_optimizer: Optimizer
    critic_optimizer: Optimizer

    def _sample_action(
        self, params: PyTree, obs: Array, key: Key
    ) -> tuple[Array, Array]:
        dist = self.actor_network.apply(params, obs)
        return dist.sample_and_log_prob(seed=key)

    def _stochastic_action(
        self, key: Key, state: AVGState
    ) -> tuple[Array, Array]:
        keys = jax.random.split(key, self.cfg.num_envs)
        return jax.vmap(self._sample_action, in_axes=(None, 0, 0))(
            state.actor_params, state.obs, keys
        )

    def _deterministic_action(
        self, key: Key, state: AVGState
    ) -> tuple[Array, Array]:
        dist = self.actor_network.apply(state.actor_params, state.obs)
        action = dist.bijector.forward(dist.distribution.mode())
        log_prob = dist.log_prob(action)
        return action, log_prob

    def _step(
        self, state: AVGState, key: Key, *, policy: Callable
    ) -> tuple[AVGState, Transition]:
        action_key, step_key = jax.random.split(key)
        action, log_prob = policy(action_key, state)

        step_keys = jax.random.split(step_key, self.cfg.num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        transition = Transition(
            first=Timestep(obs=state.obs, done=state.done),
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            aux={"log_prob": log_prob},
        )

        return (
            state.replace(
                step=state.step + self.cfg.num_envs,
                obs=next_obs,
                done=done,
                env_state=env_state,
            ),
            transition,
        )

    def _update_step(
        self, state: AVGState, key: Key
    ) -> tuple[AVGState, None]:
        sample_key, step_key, next_action_key = jax.random.split(key, 3)
        action_keys = jax.random.split(sample_key, self.cfg.num_envs)

        action, log_prob = jax.vmap(self._sample_action, in_axes=(None, 0, 0))(
            state.actor_params, state.obs, action_keys
        )
        action = jax.lax.stop_gradient(action)
        log_prob = jax.lax.stop_gradient(log_prob)

        step_keys = jax.random.split(step_key, self.cfg.num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)
        not_done = 1.0 - done.astype(jnp.float32)

        next_dist = self.actor_network.apply(
            jax.lax.stop_gradient(state.actor_params), next_obs
        )
        next_action, next_log_prob = next_dist.sample_and_log_prob(
            seed=next_action_key
        )
        next_q = self.critic_network.apply(
            jax.lax.stop_gradient(state.critic_params), next_obs, next_action
        )
        target_v = next_q - self.cfg.alpha * next_log_prob

        r_ent = reward - self.cfg.alpha * log_prob
        td_scaler = state.td_scaler.update(r_ent, done, self.cfg.gamma)
        sigma = td_scaler.sigma()

        q = self.critic_network.apply(
            jax.lax.stop_gradient(state.critic_params), state.obs, action
        )
        td_error = (reward + not_done * self.cfg.gamma * target_v - q) / sigma

        def compute_actor_loss(actor_params: PyTree, obs: Array, key: Key) -> Array:
            dist = self.actor_network.apply(actor_params, obs)
            reparam_action, reparam_log_prob = dist.sample_and_log_prob(seed=key)
            reparam_q = self.critic_network.apply(
                jax.lax.stop_gradient(state.critic_params), obs, reparam_action
            )
            return self.cfg.alpha * reparam_log_prob - reparam_q

        actor_losses, actor_grads = jax.vmap(
            jax.value_and_grad(compute_actor_loss), in_axes=(None, 0, 0)
        )(state.actor_params, state.obs, action_keys)
        actor_ascent = jax.tree.map(jnp.negative, actor_grads)
        actor_td_error = jnp.ones((self.cfg.num_envs,), dtype=jnp.float32)
        actor_updates, actor_optimizer_state = self.actor_optimizer.update(
            state.actor_optimizer_state, actor_grads, actor_ascent, actor_td_error
        )
        actor_params = jax.tree.map(
            lambda p, u: p + u, state.actor_params, actor_updates
        )

        def compute_q_value(params: PyTree, obs: Array, action: Array) -> Array:
            return self.critic_network.apply(params, obs, action)

        q_grads = jax.vmap(jax.grad(compute_q_value), in_axes=(None, 0, 0))(
            state.critic_params, state.obs, action
        )

        trace_decay = self.cfg.gamma * self.cfg.trace_lambda
        keep = trace_decay * (1.0 - state.done.astype(jnp.float32))
        critic_trace = jax.tree.map(
            lambda t, g: broadcast(keep, t) * t + g, state.critic_trace, q_grads
        )

        critic_updates, critic_optimizer_state = self.critic_optimizer.update(
            state.critic_optimizer_state, q_grads, critic_trace, td_error
        )
        critic_params = jax.tree.map(
            lambda p, u: p + u, state.critic_params, critic_updates
        )

        target = q + td_error
        explained_variance = 1 - jnp.var(td_error) / (jnp.var(target) + 1e-8)
        lox.log(
            {
                "actor/loss": actor_losses.mean(),
                "actor/log_prob": log_prob.mean(),
                "critic/q": q.mean(),
                "critic/target_v": target_v.mean(),
                "critic/td_error": td_error.mean(),
                "critic/sigma": sigma.mean(),
                "critic/explained_variance": explained_variance,
                "critic_trace/trace_norm": optax.global_norm(critic_trace),
            }
        )

        return (
            state.replace(
                step=state.step + self.cfg.num_envs,
                update_step=state.update_step + 1,
                obs=next_obs,
                done=done,
                env_state=env_state,
                actor_params=actor_params,
                actor_optimizer_state=actor_optimizer_state,
                critic_params=critic_params,
                critic_optimizer_state=critic_optimizer_state,
                critic_trace=critic_trace,
                td_scaler=td_scaler,
            ),
            None,
        )

    def init(self, key: Key) -> AVGState:
        env_key, actor_key, critic_key = jax.random.split(key, 3)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
        )
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)

        actor_params = self.actor_network.init(actor_key, obs)
        critic_params = self.critic_network.init(critic_key, obs, action)

        actor_optimizer_state = self.actor_optimizer.init(
            actor_params, self.cfg.num_envs
        )
        critic_optimizer_state = self.critic_optimizer.init(
            critic_params, self.cfg.num_envs
        )

        critic_trace = jax.tree.map(
            lambda p: jnp.zeros((self.cfg.num_envs, *p.shape), dtype=p.dtype),
            critic_params,
        )

        return AVGState(
            step=0,
            update_step=0,
            obs=obs,
            done=done,
            env_state=env_state,
            actor_params=actor_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_trace=critic_trace,
            td_scaler=TDErrorScalerState.init(self.cfg.num_envs),
        )

    def warmup(self, key: Key, state: AVGState, num_steps: int) -> AVGState:
        return state

    def train(self, key: Key, state: AVGState, num_steps: int) -> AVGState:
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            self._update_step, state, keys, unroll=self.cfg.unroll
        )
        return state

    def evaluate(
        self,
        key: Key,
        state: AVGState,
        num_steps: int,
        deterministic: bool = True,
    ) -> AVGState:
        reset_key, eval_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_keys, self.env_params
        )
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        state = state.replace(step=0, obs=obs, done=done, env_state=env_state)

        policy = (
            self._deterministic_action if deterministic else self._stochastic_action
        )
        state, _ = jax.lax.scan(
            partial(self._step, policy=policy),
            state,
            jax.random.split(eval_key, num_steps // self.cfg.num_envs),
        )
        return state
