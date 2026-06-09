from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
from flax import core, struct

from streax.optimizers import Optimizer
from streax.utils import Timestep, Transition, canonicalize_dtype
from streax.utils.typing import Array, Environment, EnvParams, EnvState, Key, PyTree


@struct.dataclass(frozen=True)
class ACLambdaConfig:
    gamma: float
    trace_lambda: float
    entropy_coefficient: float = 0.01
    unroll: int = struct.field(pytree_node=False, default=4)


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
    ) -> tuple[ACLambdaState, Array, dict]:
        def log_prob_and_entropy(params):
            dist, aux = self.actor_network.apply(params, state.timestep.obs)
            action, _ = dist.sample_and_log_prob(seed=key)
            action = jax.lax.stop_gradient(action)
            return (dist.log_prob(action), dist.entropy()), (action, aux)

        (log_prob, entropy), actor_vjp, (action, aux) = jax.vjp(
            log_prob_and_entropy, state.actor_params, has_aux=True
        )
        (log_prob_grads,) = actor_vjp(
            (jnp.ones_like(log_prob), jnp.zeros_like(entropy))
        )
        (entropy_grads,) = actor_vjp((jnp.zeros_like(log_prob), jnp.ones_like(entropy)))
        aux = {
            **aux,
            "log_prob": log_prob,
            "entropy": entropy,
            "log_prob_grads": log_prob_grads,
            "entropy_grads": entropy_grads,
        }
        return state, action, aux

    def _deterministic_action(
        self, key: Key, state: ACLambdaState
    ) -> tuple[ACLambdaState, Array, dict]:
        dist, aux = self.actor_network.apply(state.actor_params, state.timestep.obs)
        action = dist.mode()
        aux = {**aux, "log_prob": dist.log_prob(action), "entropy": dist.entropy()}
        return state, action, aux

    def _step(
        self, state: ACLambdaState, key: Key, *, policy: Callable
    ) -> tuple[ACLambdaState, Transition]:
        action_key, step_key = jax.random.split(key)
        state, action, aux = policy(action_key, state)

        next_obs, env_state, reward, done, info = self.env.step(
            step_key, state.env_state, action, self.env_params
        )
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            aux=aux,
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
        self, state: ACLambdaState, key: Key, *, policy: Callable
    ) -> tuple[ACLambdaState, None]:
        state, transition = self._step(state, key, policy=policy)
        state = self._update(state, transition)
        return state.replace(update_step=state.update_step + 1), None

    def _update(
        self,
        state: ACLambdaState,
        transition: Transition,
    ) -> ACLambdaState:
        aux = transition.aux
        log_probs = aux["log_prob"]
        entropy_values = aux["entropy"]
        log_prob_grads = aux["log_prob_grads"]
        entropy_grads = aux["entropy_grads"]

        def get_value(params):
            critic_value, _ = self.critic_network.apply(params, transition.first.obs)
            return critic_value.squeeze(-1)

        critic_value, critic_vjp = jax.vjp(get_value, state.critic_params)
        (critic_grads,) = critic_vjp(jnp.ones_like(critic_value))

        reset_trace = transition.second.done
        discount = jnp.float32(self.cfg.gamma * self.cfg.trace_lambda)

        def accumulate(trace, gradient):
            return jax.tree.map(
                lambda t, g: (discount * t + g).astype(t.dtype), trace, gradient
            )

        def reset_eligibility(trace):
            return jax.tree.map(
                lambda t: jnp.where(reset_trace, jnp.zeros_like(t), t),
                trace,
            )

        critic_trace = accumulate(state.critic_trace, critic_grads)

        def get_next_value(params):
            critic_value, _ = self.critic_network.apply(params, transition.second.obs)
            return critic_value.squeeze(-1)

        not_done = 1.0 - transition.second.done.astype(jnp.float32)
        next_value, curvature = self.critic_optimizer.bootstrap(
            state.critic_optimizer_state,
            state.critic_params,
            critic_grads,
            critic_trace,
            get_next_value,
            self.cfg.gamma,
            not_done,
        )
        td_error = (
            transition.second.reward
            + self.cfg.gamma * (1.0 - transition.second.done) * next_value
            - critic_value
        )

        actor_grads = jax.tree.map(
            lambda lpg, eg: lpg
            + jnp.sign(td_error) * self.cfg.entropy_coefficient * eg,
            log_prob_grads,
            entropy_grads,
        )
        actor_trace = accumulate(state.actor_trace, actor_grads)

        actor_updates, actor_optimizer_state = self.actor_optimizer.update(
            state.actor_optimizer_state,
            actor_grads,
            actor_trace,
            td_error,
        )

        critic_updates, critic_optimizer_state = self.critic_optimizer.update(
            state.critic_optimizer_state,
            critic_grads,
            critic_trace,
            td_error,
            curvature,
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

        target = critic_value + td_error
        explained_variance = 1 - jnp.var(td_error) / (jnp.var(target) + 1e-8)
        log_dict = {
            "critic/value": critic_value.mean(),
            "critic/td_error": td_error.mean(),
            "critic/absolute_td_error": jnp.abs(td_error).mean(),
            "critic/explained_variance": explained_variance,
            "actor/log_prob": log_probs.mean(),
            "actor/entropy": entropy_values.mean(),
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
        obs, env_state = self.env.reset(env_key, self.env_params)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape, dtype=canonicalize_dtype(action_space.dtype)
        )
        timestep = Timestep(obs=obs, action=action, reward=0.0, done=True)
        actor_params = self.actor_network.init(actor_key, obs)
        critic_params = self.critic_network.init(critic_key, obs)

        actor_optimizer_state = self.actor_optimizer.init(actor_params)
        critic_optimizer_state = self.critic_optimizer.init(critic_params)

        actor_trace = jax.tree.map(jnp.zeros_like, actor_params)
        critic_trace = jax.tree.map(jnp.zeros_like, critic_params)

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
        keys = jax.random.split(key, num_steps)
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
    ) -> ACLambdaState:
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
