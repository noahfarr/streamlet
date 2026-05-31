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
class ACLambdaConfig:
    num_envs: int
    gamma: float
    trace_lambda: float
    entropy_coefficient: float = 0.01
    unroll: int = struct.field(pytree_node=False, default=2)


@struct.dataclass(frozen=True)
class ACLambdaState:
    step: int
    update_step: int
    timestep: Timestep
    env_state: EnvState
    actor_params: core.FrozenDict[str, Any]
    critic_params: core.FrozenDict[str, Any]
    actor_trace: PyTree
    critic_trace: PyTree
    actor_optimizer_state: PyTree
    critic_optimizer_state: PyTree


@dataclass
class ACLambda:
    cfg: ACLambdaConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    actor_optimizer: Optimizer
    critic_optimizer: Optimizer

    def _stochastic_action(
        self, key: Key, state: ACLambdaState
    ) -> tuple[ACLambdaState, Array, Array]:
        dist = self.actor_network.apply(state.actor_params, state.timestep.obs)
        action, log_prob = dist.sample_and_log_prob(seed=key)
        return state, action, log_prob

    def _deterministic_action(
        self, key: Key, state: ACLambdaState
    ) -> tuple[ACLambdaState, Array, Array]:
        dist = self.actor_network.apply(state.actor_params, state.timestep.obs)
        action = dist.mode()
        log_prob = dist.log_prob(action)
        return state, action, log_prob

    def _step(
        self, state: ACLambdaState, key: Key, *, policy: Callable
    ) -> tuple[ACLambdaState, Transition]:
        action_key, step_key = jax.random.split(key)
        state, action, log_prob = policy(action_key, state)

        step_keys = jax.random.split(step_key, self.cfg.num_envs)
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
        self, state: ACLambdaState, key: Key, *, policy: Callable
    ) -> tuple[ACLambdaState, None]:
        step_key, _ = jax.random.split(key)
        state, transition = self._step(state, step_key, policy=policy)
        state = self._update(state, transition)
        return state.replace(update_step=state.update_step + 1), None

    def _update(
        self,
        state: ACLambdaState,
        transition: Transition,
    ) -> ACLambdaState:
        action = transition.second.action

        def compute_log_probs(params):
            dist = self.actor_network.apply(params, transition.first.obs)
            return dist.log_prob(action), dist.entropy()

        def compute_td_error(params):
            v = self.critic_network.apply(params, transition.first.obs)
            critic_value = v.squeeze(-1)
            next_v = self.critic_network.apply(params, transition.second.obs)
            next_value = next_v.squeeze(-1)
            td_error = (
                transition.second.reward
                + self.cfg.gamma * (1.0 - transition.second.done) * next_value
                - critic_value
            )
            return critic_value, td_error

        critic_values, critic_vjp, td_error = jax.vjp(
            compute_td_error, state.critic_params, has_aux=True
        )
        batch = self.cfg.num_envs
        (critic_grads,) = jax.vmap(critic_vjp)(jnp.eye(batch, dtype=critic_values.dtype))

        (log_probs, entropy_values), actor_vjp = jax.vjp(
            compute_log_probs, state.actor_params
        )
        eye = jnp.eye(batch, dtype=log_probs.dtype)
        zeros = jnp.zeros((batch, batch), dtype=log_probs.dtype)
        (log_prob_grads,) = jax.vmap(actor_vjp)((eye, zeros))
        (entropy_grads,) = jax.vmap(actor_vjp)((zeros, eye))

        reset_trace = transition.second.done
        discount = jnp.broadcast_to(
            jnp.float32(self.cfg.gamma * self.cfg.trace_lambda), reset_trace.shape
        )
        actor_grads = jax.tree.map(
            lambda lpg, eg: lpg
            + broadcast(jnp.sign(td_error), eg) * self.cfg.entropy_coefficient * eg,
            log_prob_grads,
            entropy_grads,
        )

        def accumulate(trace, gradient):
            return jax.tree.map(
                lambda t, g: broadcast(discount, t) * t + g, trace, gradient
            )

        def reset_eligibility(trace):
            return jax.tree.map(
                lambda t: jnp.where(broadcast(reset_trace, t), jnp.zeros_like(t), t),
                trace,
            )

        actor_trace = accumulate(state.actor_trace, actor_grads)
        critic_trace = accumulate(state.critic_trace, critic_grads)

        actor_updates, actor_optimizer_state = self.actor_optimizer.update(
            state.actor_optimizer_state, actor_grads, actor_trace, td_error,
        )

        if isinstance(self.critic_optimizer, Implicit):
            gradient_trace = sum(
                jnp.sum(g * z, axis=tuple(range(1, g.ndim)))
                for g, z in zip(
                    jax.tree.leaves(critic_grads), jax.tree.leaves(critic_trace)
                )
            )

            def bootstrap_value(params, obs):
                return self.critic_network.apply(params, obs).squeeze(-1)

            def directional(obs, direction):
                _, jvp_value = jax.jvp(
                    lambda params: bootstrap_value(params, obs),
                    (state.critic_params,),
                    (direction,),
                )
                return jvp_value

            next_grad_trace = jax.vmap(directional)(
                transition.second.obs, critic_trace
            )
            curvature = gradient_trace - self.cfg.gamma * (
                1.0 - transition.second.done.astype(jnp.float32)
            ) * next_grad_trace
            critic_updates, critic_optimizer_state = self.critic_optimizer.update(
                state.critic_optimizer_state,
                critic_grads,
                critic_trace,
                td_error,
                curvature,
            )
        else:
            critic_updates, critic_optimizer_state = self.critic_optimizer.update(
                state.critic_optimizer_state, critic_grads, critic_trace, td_error,
            )

        actor_params = jax.tree.map(
            lambda p, u: p + u,
            state.actor_params,
            actor_updates,
        )
        critic_params = jax.tree.map(
            lambda p, u: p + u, state.critic_params, critic_updates
        )

        new_actor_trace = reset_eligibility(actor_trace)
        new_critic_trace = reset_eligibility(critic_trace)

        target = critic_values + td_error
        explained_variance = 1 - jnp.var(td_error) / (jnp.var(target) + 1e-8)
        log_dict = {
            "critic/value": critic_values.mean(),
            "critic/td_error": td_error.mean(),
            "critic/explained_variance": explained_variance,
            "actor/log_prob": log_probs.mean(),
            "actor/entropy": entropy_values.mean(),
            "actor_trace/trace_norm": optax.global_norm(new_actor_trace),
            "critic_trace/trace_norm": optax.global_norm(new_critic_trace),
        }
        lox.log(log_dict)

        new_state = dict(
            actor_params=actor_params,
            critic_params=critic_params,
            actor_trace=new_actor_trace,
            critic_trace=new_critic_trace,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
        )

        return state.replace(**new_state)

    def init(self, key: Key) -> ACLambdaState:
        env_key, actor_key, critic_key = jax.random.split(key, 3)
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
        actor_params = self.actor_network.init(actor_key, obs)
        critic_params = self.critic_network.init(critic_key, obs)

        actor_optimizer_state = self.actor_optimizer.init(
            actor_params, self.cfg.num_envs
        )
        critic_optimizer_state = self.critic_optimizer.init(
            critic_params, self.cfg.num_envs
        )

        actor_trace = jax.tree.map(
            lambda p: jnp.zeros((self.cfg.num_envs, *p.shape), dtype=p.dtype),
            actor_params,
        )
        critic_trace = jax.tree.map(
            lambda p: jnp.zeros((self.cfg.num_envs, *p.shape), dtype=p.dtype),
            critic_params,
        )

        state = dict(
            step=0,
            update_step=0,
            timestep=timestep,
            env_state=env_state,
            actor_params=actor_params,
            critic_params=critic_params,
            actor_trace=actor_trace,
            critic_trace=critic_trace,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
        )

        return ACLambdaState(**state)

    def warmup(self, key: Key, state: ACLambdaState, num_steps: int) -> ACLambdaState:
        return state

    def train(self, key: Key, state: ACLambdaState, num_steps: int) -> ACLambdaState:
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            partial(self._update_step, policy=self._stochastic_action),
            state,
            keys,
            unroll=self.cfg.unroll,
        )
        return state

    def evaluate(
        self,
        key: Key,
        state: ACLambdaState,
        num_steps: int,
        deterministic: bool = True,
    ) -> ACLambdaState:
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

        policy = self._deterministic_action if deterministic else self._stochastic_action
        state, _ = jax.lax.scan(
            partial(self._step, policy=policy),
            state,
            jax.random.split(eval_key, num_steps),
        )
        return state
