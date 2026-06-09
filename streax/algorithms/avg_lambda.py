from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from streax.optimizers import Optimizer
from streax.utils import Timestep, Transition, TDErrorScalerState, canonicalize_dtype
from streax.utils.typing import (
    Array,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)


@struct.dataclass(frozen=True)
class AVGLambdaConfig:
    gamma: float
    alpha: float
    trace_lambda: float = 0.0
    unroll: int = struct.field(pytree_node=False, default=4)


@struct.dataclass(frozen=True)
class AVGLambdaState:
    step: int
    update_step: int
    timestep: Timestep
    env_state: EnvState
    actor_params: core.FrozenDict[str, Any]
    actor_optimizer_state: PyTree
    critic_params: core.FrozenDict[str, Any]
    critic_optimizer_state: PyTree
    critic_trace: PyTree
    td_scaler: TDErrorScalerState
    aux_actor_optimizer_state: PyTree
    aux_critic_optimizer_state: PyTree


@dataclass
class AVGLambda:
    cfg: AVGLambdaConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    actor_optimizer: Optimizer
    critic_optimizer: Optimizer
    aux_actor_loss: Callable = lambda params, transition: 0.0
    aux_critic_loss: Callable = lambda params, transition: 0.0
    aux_actor_optimizer: optax.GradientTransformation = optax.sgd(1e-3)
    aux_critic_optimizer: optax.GradientTransformation = optax.sgd(1e-3)

    def _sample_action(
        self, params: PyTree, obs: Array, key: Key
    ) -> tuple[Array, Array]:
        dist = self.actor_network.apply(params, obs)
        return dist.sample_and_log_prob(seed=key)

    def _stochastic_action(
        self, key: Key, state: AVGLambdaState
    ) -> tuple[Array, Array]:
        return self._sample_action(state.actor_params, state.timestep.obs, key)

    def _deterministic_action(
        self, key: Key, state: AVGLambdaState
    ) -> tuple[Array, Array]:
        dist = self.actor_network.apply(state.actor_params, state.timestep.obs)
        action = dist.bijector.forward(dist.distribution.mode())
        log_prob = dist.log_prob(action)
        return action, log_prob

    def _step(
        self, state: AVGLambdaState, key: Key, *, policy: Callable
    ) -> tuple[AVGLambdaState, Transition]:
        action_key, step_key = jax.random.split(key)
        action, log_prob = policy(action_key, state)

        next_obs, env_state, reward, done, info = self.env.step(
            step_key, state.env_state, action, self.env_params
        )
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            aux={"log_prob": log_prob},
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
        self, state: AVGLambdaState, key: Key
    ) -> tuple[AVGLambdaState, None]:
        sample_key, step_key, next_action_key = jax.random.split(key, 3)

        action, log_prob = self._sample_action(
            state.actor_params, state.timestep.obs, sample_key
        )
        action = jax.lax.stop_gradient(action)
        log_prob = jax.lax.stop_gradient(log_prob)

        next_obs, env_state, reward, done, info = self.env.step(
            step_key, state.env_state, action, self.env_params
        )
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)
        not_done = 1.0 - done.astype(jnp.float32)

        next_dist = self.actor_network.apply(
            jax.lax.stop_gradient(state.actor_params), next_obs
        )
        next_action, next_log_prob = next_dist.sample_and_log_prob(
            seed=next_action_key
        )
        next_q_value = self.critic_network.apply(
            jax.lax.stop_gradient(state.critic_params), next_obs, next_action
        )
        next_value = next_q_value - self.cfg.alpha * next_log_prob

        entropy_reward = reward - self.cfg.alpha * log_prob
        td_scaler = state.td_scaler.update(entropy_reward, done, self.cfg.gamma)
        sigma = td_scaler.sigma()

        def get_q_value(params: PyTree, obs: Array, action: Array) -> Array:
            q = self.critic_network.apply(params, obs, action)
            return q

        q_value, q_grads = jax.value_and_grad(get_q_value)(
            state.critic_params, state.timestep.obs, action
        )
        td_error = (reward + not_done * self.cfg.gamma * next_value - q_value) / sigma

        def compute_actor_loss(actor_params: PyTree, obs: Array, key: Key) -> Array:
            dist = self.actor_network.apply(actor_params, obs)
            sampled_action, sampled_log_prob = dist.sample_and_log_prob(seed=key)
            sampled_q = self.critic_network.apply(
                jax.lax.stop_gradient(state.critic_params), obs, sampled_action
            )
            return self.cfg.alpha * sampled_log_prob - sampled_q

        actor_loss, actor_grads = jax.value_and_grad(compute_actor_loss)(
            state.actor_params, state.timestep.obs, sample_key
        )
        actor_ascent = jax.tree.map(jnp.negative, actor_grads)
        actor_td_error = jnp.float32(1.0)
        actor_updates, actor_optimizer_state = self.actor_optimizer.update(
            state.actor_optimizer_state, actor_grads, actor_ascent, actor_td_error
        )
        actor_params = jax.tree.map(
            lambda p, u: p + u, state.actor_params, actor_updates
        )

        trace_decay = self.cfg.gamma * self.cfg.trace_lambda
        keep = trace_decay * (1.0 - state.timestep.done.astype(jnp.float32))
        critic_trace = jax.tree.map(
            lambda t, g: (keep * t + g).astype(t.dtype), state.critic_trace, q_grads
        )

        critic_updates, critic_optimizer_state = self.critic_optimizer.update(
            state.critic_optimizer_state, q_grads, critic_trace, td_error
        )
        critic_params = jax.tree.map(
            lambda p, u: p + u, state.critic_params, critic_updates
        )

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
        )

        _, aux_actor_grads = jax.value_and_grad(self.aux_actor_loss)(
            actor_params, transition
        )
        aux_actor_updates, aux_actor_optimizer_state = self.aux_actor_optimizer.update(
            aux_actor_grads, state.aux_actor_optimizer_state, actor_params
        )
        actor_params = optax.apply_updates(actor_params, aux_actor_updates)

        _, aux_critic_grads = jax.value_and_grad(self.aux_critic_loss)(
            critic_params, transition
        )
        aux_critic_updates, aux_critic_optimizer_state = (
            self.aux_critic_optimizer.update(
                aux_critic_grads, state.aux_critic_optimizer_state, critic_params
            )
        )
        critic_params = optax.apply_updates(critic_params, aux_critic_updates)

        target = q_value + td_error
        explained_variance = 1 - jnp.var(td_error) / (jnp.var(target) + 1e-8)
        lox.log(
            {
                "actor/loss": actor_loss.mean(),
                "actor/log_prob": log_prob.mean(),
                "critic/q_value": q_value.mean(),
                "critic/next_value": next_value.mean(),
                "critic/td_error": td_error.mean(),
                "critic/absolute_td_error": jnp.abs(td_error).mean(),
                "critic/sigma": sigma.mean(),
                "critic/explained_variance": explained_variance,
            }
        )

        return (
            state.replace(
                step=state.step + 1,
                update_step=state.update_step + 1,
                timestep=Timestep(
                    obs=next_obs,
                    action=jnp.where(done, jnp.zeros_like(action), action),
                    reward=jnp.where(done, jnp.zeros_like(reward), reward),
                    done=done,
                ),
                env_state=env_state,
                actor_params=actor_params,
                actor_optimizer_state=actor_optimizer_state,
                critic_params=critic_params,
                critic_optimizer_state=critic_optimizer_state,
                critic_trace=critic_trace,
                td_scaler=td_scaler,
                aux_actor_optimizer_state=aux_actor_optimizer_state,
                aux_critic_optimizer_state=aux_critic_optimizer_state,
            ),
            None,
        )

    def init(self, key: Key) -> AVGLambdaState:
        env_key, actor_key, critic_key = jax.random.split(key, 3)
        obs, env_state = self.env.reset(env_key, self.env_params)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape, dtype=canonicalize_dtype(action_space.dtype)
        )
        timestep = Timestep(obs=obs, action=action, reward=0.0, done=True)

        actor_params = self.actor_network.init(actor_key, obs)
        critic_params = self.critic_network.init(critic_key, obs, action)

        actor_optimizer_state = self.actor_optimizer.init(actor_params)
        critic_optimizer_state = self.critic_optimizer.init(critic_params)
        aux_actor_optimizer_state = self.aux_actor_optimizer.init(actor_params)
        aux_critic_optimizer_state = self.aux_critic_optimizer.init(critic_params)

        critic_trace = jax.tree.map(jnp.zeros_like, critic_params)

        return AVGLambdaState(
            step=0,
            update_step=0,
            timestep=timestep,
            env_state=env_state,
            actor_params=actor_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_trace=critic_trace,
            td_scaler=TDErrorScalerState.init(),
            aux_actor_optimizer_state=aux_actor_optimizer_state,
            aux_critic_optimizer_state=aux_critic_optimizer_state,
        )

    def warmup(self, key: Key, state: AVGLambdaState, num_steps: int) -> AVGLambdaState:
        return state

    def train(self, key: Key, state: AVGLambdaState, num_steps: int) -> AVGLambdaState:
        keys = jax.random.split(key, num_steps)
        state, _ = jax.lax.scan(
            self._update_step, state, keys, unroll=self.cfg.unroll
        )
        return state

    def evaluate(
        self,
        key: Key,
        state: AVGLambdaState,
        num_steps: int,
    ) -> AVGLambdaState:
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
            partial(self._step, policy=self._deterministic_action),
            state,
            jax.random.split(eval_key, num_steps),
        )
        return state
