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
from streax.utils import (
    TDErrorScalerState,
    Timestep,
    Transition,
    broadcast,
    canonicalize_dtype,
)
from streax.utils.typing import (
    Array,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)


@struct.dataclass(frozen=True)
class RecurrentAVGLambdaConfig:
    num_envs: int
    gamma: float
    alpha: float
    trace_lambda: float = 0.0
    unroll: int = struct.field(pytree_node=False, default=2)


@struct.dataclass(frozen=True)
class RecurrentAVGLambdaState:
    step: int
    update_step: int
    timestep: Timestep
    actor_carry: PyTree
    critic_carry: PyTree
    env_state: EnvState
    actor_params: core.FrozenDict[str, Any]
    actor_optimizer_state: PyTree
    critic_params: core.FrozenDict[str, Any]
    critic_optimizer_state: PyTree
    critic_trace: PyTree
    td_scaler: TDErrorScalerState


@dataclass
class RecurrentAVGLambda:
    cfg: RecurrentAVGLambdaConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    actor_optimizer: Optimizer
    critic_optimizer: Optimizer

    def _actor_apply(
        self, params: PyTree, carry: PyTree, timestep: Timestep
    ) -> tuple[PyTree, Any]:
        return self.actor_network.apply(
            params,
            carry,
            timestep.obs,
            timestep.action,
            timestep.reward,
            timestep.done,
        )

    def _critic_apply(
        self, params: PyTree, carry: PyTree, timestep: Timestep, action: Array
    ) -> tuple[PyTree, Array]:
        return self.critic_network.apply(
            params,
            carry,
            timestep.obs,
            action,
            timestep.reward,
            timestep.done,
        )

    def _reset_carry(self, carry: PyTree, done: Array) -> PyTree:
        return jax.tree.map(
            lambda leaf: jnp.where(broadcast(done, leaf), jnp.zeros_like(leaf), leaf),
            carry,
        )

    def _stochastic_action(
        self, key: Key, dist: Any
    ) -> tuple[Array, Array]:
        return dist.sample_and_log_prob(seed=key)

    def _deterministic_action(
        self, key: Key, dist: Any
    ) -> tuple[Array, Array]:
        action = dist.bijector.forward(dist.distribution.mode())
        log_prob = dist.log_prob(action)
        return action, log_prob

    def _step(
        self, state: RecurrentAVGLambdaState, key: Key, *, policy: Callable
    ) -> tuple[RecurrentAVGLambdaState, Transition]:
        action_key, step_key = jax.random.split(key)

        actor_carry_next, dist = self._actor_apply(
            state.actor_params, state.actor_carry, state.timestep
        )
        action, log_prob = policy(action_key, dist)

        step_keys = jax.random.split(step_key, self.cfg.num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            aux={"log_prob": log_prob},
        )

        return (
            state.replace(
                step=state.step + self.cfg.num_envs,
                timestep=Timestep(
                    obs=next_obs,
                    action=jnp.where(broadcast(done, action), jnp.zeros_like(action), action),
                    reward=jnp.where(done, jnp.zeros_like(reward), reward),
                    done=done,
                ),
                actor_carry=self._reset_carry(actor_carry_next, done),
                env_state=env_state,
            ),
            transition,
        )

    def _update_step(
        self, state: RecurrentAVGLambdaState, key: Key
    ) -> tuple[RecurrentAVGLambdaState, None]:
        sample_key, step_key, next_action_key = jax.random.split(key, 3)
        action_keys = jax.random.split(sample_key, self.cfg.num_envs)
        timestep = state.timestep

        def sample(actor_params, carry, ts, k):
            carry, dist = self._actor_apply(actor_params, carry, ts)
            action, log_prob = dist.sample_and_log_prob(seed=k)
            return carry, action, log_prob

        actor_carry_next, action, log_prob = jax.vmap(
            sample, in_axes=(None, 0, 0, 0)
        )(state.actor_params, state.actor_carry, timestep, action_keys)
        action = jax.lax.stop_gradient(action)
        log_prob = jax.lax.stop_gradient(log_prob)

        step_keys = jax.random.split(step_key, self.cfg.num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)
        not_done = 1.0 - done.astype(jnp.float32)

        next_timestep = Timestep(
            obs=next_obs, action=action, reward=reward, done=done
        )
        actor_carry_for_next = self._reset_carry(actor_carry_next, done)

        def critic_value(critic_params, carry, ts, action):
            carry, q = self._critic_apply(critic_params, carry, ts, action)
            return carry, q

        critic_carry_next, q = jax.vmap(critic_value, in_axes=(None, 0, 0, 0))(
            jax.lax.stop_gradient(state.critic_params),
            state.critic_carry,
            timestep,
            action,
        )
        critic_carry_for_next = self._reset_carry(critic_carry_next, done)

        def target_value(actor_params, critic_params, a_carry, c_carry, ts, k):
            _, next_dist = self._actor_apply(actor_params, a_carry, ts)
            next_action, next_log_prob = next_dist.sample_and_log_prob(seed=k)
            _, next_q = self._critic_apply(critic_params, c_carry, ts, next_action)
            return next_q, next_log_prob

        next_action_keys = jax.random.split(next_action_key, self.cfg.num_envs)
        next_q, next_log_prob = jax.vmap(
            target_value, in_axes=(None, None, 0, 0, 0, 0)
        )(
            jax.lax.stop_gradient(state.actor_params),
            jax.lax.stop_gradient(state.critic_params),
            actor_carry_for_next,
            critic_carry_for_next,
            next_timestep,
            next_action_keys,
        )
        target_v = next_q - self.cfg.alpha * next_log_prob

        r_ent = reward - self.cfg.alpha * log_prob
        td_scaler = state.td_scaler.update(r_ent, done, self.cfg.gamma)
        sigma = td_scaler.sigma()
        td_error = (reward + not_done * self.cfg.gamma * target_v - q) / sigma

        def compute_actor_loss(actor_params, a_carry, c_carry, ts, key):
            _, dist = self._actor_apply(actor_params, a_carry, ts)
            reparam_action, reparam_log_prob = dist.sample_and_log_prob(seed=key)
            _, reparam_q = self._critic_apply(
                jax.lax.stop_gradient(state.critic_params), c_carry, ts, reparam_action
            )
            return self.cfg.alpha * reparam_log_prob - reparam_q

        actor_losses, actor_grads = jax.vmap(
            jax.value_and_grad(compute_actor_loss), in_axes=(None, 0, 0, 0, 0)
        )(
            state.actor_params,
            state.actor_carry,
            state.critic_carry,
            timestep,
            action_keys,
        )
        actor_ascent = jax.tree.map(jnp.negative, actor_grads)
        actor_td_error = jnp.ones((self.cfg.num_envs,), dtype=jnp.float32)
        actor_updates, actor_optimizer_state = self.actor_optimizer.update(
            state.actor_optimizer_state, actor_grads, actor_ascent, actor_td_error
        )
        actor_params = jax.tree.map(
            lambda p, u: p + u, state.actor_params, actor_updates
        )

        def compute_q_value(critic_params, c_carry, ts, action):
            _, q = self._critic_apply(critic_params, c_carry, ts, action)
            return q

        q_grads = jax.vmap(jax.grad(compute_q_value), in_axes=(None, 0, 0, 0))(
            state.critic_params, state.critic_carry, timestep, action
        )

        trace_decay = self.cfg.gamma * self.cfg.trace_lambda
        keep = trace_decay * (1.0 - state.timestep.done.astype(jnp.float32))
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
                timestep=Timestep(
                    obs=next_obs,
                    action=jnp.where(
                        broadcast(done, action), jnp.zeros_like(action), action
                    ),
                    reward=jnp.where(done, jnp.zeros_like(reward), reward),
                    done=done,
                ),
                actor_carry=actor_carry_for_next,
                critic_carry=critic_carry_for_next,
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

    def init(self, key: Key) -> RecurrentAVGLambdaState:
        env_key, actor_key, critic_key, actor_carry_key, critic_carry_key = (
            jax.random.split(key, 5)
        )
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

        actor_carry = self.actor_network.initialize_carry(
            actor_carry_key, self.cfg.num_envs
        )
        critic_carry = self.critic_network.initialize_carry(
            critic_carry_key, self.cfg.num_envs
        )
        actor_params = self.actor_network.init(
            actor_key,
            actor_carry,
            timestep.obs,
            timestep.action,
            timestep.reward,
            timestep.done,
        )
        critic_params = self.critic_network.init(
            critic_key,
            critic_carry,
            timestep.obs,
            action,
            timestep.reward,
            timestep.done,
        )

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

        return RecurrentAVGLambdaState(
            step=0,
            update_step=0,
            timestep=timestep,
            actor_carry=actor_carry,
            critic_carry=critic_carry,
            env_state=env_state,
            actor_params=actor_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_trace=critic_trace,
            td_scaler=TDErrorScalerState.init(self.cfg.num_envs),
        )

    def warmup(
        self, key: Key, state: RecurrentAVGLambdaState, num_steps: int
    ) -> RecurrentAVGLambdaState:
        return state

    def train(
        self, key: Key, state: RecurrentAVGLambdaState, num_steps: int
    ) -> RecurrentAVGLambdaState:
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            self._update_step, state, keys, unroll=self.cfg.unroll
        )
        return state

    def evaluate(
        self,
        key: Key,
        state: RecurrentAVGLambdaState,
        num_steps: int,
        deterministic: bool = True,
    ) -> RecurrentAVGLambdaState:
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
            actor_carry=self.actor_network.initialize_carry(
                carry_key, self.cfg.num_envs
            ),
            env_state=env_state,
        )

        policy = (
            self._deterministic_action if deterministic else self._stochastic_action
        )
        state, _ = jax.lax.scan(
            partial(self._step, policy=policy),
            state,
            jax.random.split(eval_key, num_steps // self.cfg.num_envs),
        )
        return state
