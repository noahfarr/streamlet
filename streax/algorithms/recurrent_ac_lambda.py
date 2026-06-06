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
class RecurrentACLambdaConfig:
    gamma: float
    trace_lambda: float
    entropy_coefficient: float = 0.01
    unroll: int = struct.field(pytree_node=False, default=4)


@struct.dataclass(frozen=True)
class RecurrentACLambdaState:
    step: int
    update_step: int
    timestep: Timestep
    actor_carry: PyTree
    critic_carry: PyTree
    env_state: EnvState
    actor_params: core.FrozenDict[str, Any]
    critic_params: core.FrozenDict[str, Any]
    actor_trace: PyTree
    critic_trace: PyTree
    actor_optimizer_state: PyTree
    critic_optimizer_state: PyTree


@dataclass
class RecurrentACLambda:
    cfg: RecurrentACLambdaConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    actor_optimizer: Optimizer
    critic_optimizer: Optimizer

    def _apply_actor(
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

    def _apply_critic(
        self, params: PyTree, carry: PyTree, timestep: Timestep
    ) -> tuple[PyTree, Array]:
        return self.critic_network.apply(
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

    def _critic_value_and_grad(
        self, state: RecurrentACLambdaState
    ) -> tuple[PyTree, Array, PyTree]:
        (critic_carry_next, value), critic_vjp = jax.vjp(
            lambda p: self._apply_critic(p, state.critic_carry, state.timestep),
            state.critic_params,
        )
        carry_bar = jax.tree.map(jnp.zeros_like, critic_carry_next)
        (critic_grads,) = critic_vjp((carry_bar, jnp.ones_like(value)))
        return critic_carry_next, value.squeeze(-1), critic_grads

    def _stochastic_action(
        self, key: Key, state: RecurrentACLambdaState
    ) -> tuple[RecurrentACLambdaState, Array, dict]:
        def actor_outputs(params):
            actor_carry_next, dist = self._apply_actor(
                params, state.actor_carry, state.timestep
            )
            action, _ = dist.sample_and_log_prob(seed=key)
            action = jax.lax.stop_gradient(action)
            return (actor_carry_next, dist.log_prob(action), dist.entropy()), action

        (actor_carry_next, log_prob, entropy), actor_vjp, action = jax.vjp(
            actor_outputs, state.actor_params, has_aux=True
        )
        carry_bar = jax.tree.map(jnp.zeros_like, actor_carry_next)
        one = jnp.ones_like(log_prob)
        zero = jnp.zeros_like(log_prob)
        (log_prob_grads,) = actor_vjp((carry_bar, one, zero))
        (entropy_grads,) = actor_vjp((carry_bar, zero, one))

        critic_carry_next, critic_value, critic_grads = self._critic_value_and_grad(
            state
        )

        aux = {
            "log_prob": log_prob,
            "entropy": entropy,
            "log_prob_grads": log_prob_grads,
            "entropy_grads": entropy_grads,
            "critic_value": critic_value,
            "critic_grads": critic_grads,
            "actor_carry_next": actor_carry_next,
            "critic_carry_next": critic_carry_next,
        }
        return state, action, aux

    def _deterministic_action(
        self, key: Key, state: RecurrentACLambdaState
    ) -> tuple[RecurrentACLambdaState, Array, dict]:
        actor_carry_next, dist = self._apply_actor(
            state.actor_params, state.actor_carry, state.timestep
        )
        action = dist.mode()
        critic_carry_next, _ = self._apply_critic(
            state.critic_params, state.critic_carry, state.timestep
        )
        aux = {
            "log_prob": dist.log_prob(action),
            "entropy": dist.entropy(),
            "actor_carry_next": actor_carry_next,
            "critic_carry_next": critic_carry_next,
        }
        return state, action, aux

    def _step(
        self, state: RecurrentACLambdaState, key: Key, *, policy: Callable
    ) -> tuple[RecurrentACLambdaState, Transition]:
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
                actor_carry=self._reset_carry(aux["actor_carry_next"], done),
                critic_carry=self._reset_carry(aux["critic_carry_next"], done),
                env_state=env_state,
            ),
            transition,
        )

    def _update_step(
        self, state: RecurrentACLambdaState, key: Key, *, policy: Callable
    ) -> tuple[RecurrentACLambdaState, None]:
        state, transition = self._step(state, key, policy=policy)
        state = self._update(state, transition)
        return state.replace(update_step=state.update_step + 1), None

    def _update(
        self,
        state: RecurrentACLambdaState,
        transition: Transition,
    ) -> RecurrentACLambdaState:
        aux = transition.aux
        log_probs = aux["log_prob"]
        entropy_values = aux["entropy"]
        log_prob_grads = aux["log_prob_grads"]
        entropy_grads = aux["entropy_grads"]
        critic_value = aux["critic_value"]
        critic_grads = aux["critic_grads"]
        critic_carry_next = aux["critic_carry_next"]

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

        def bootstrap_value(params):
            _, next_value = self._apply_critic(
                params, critic_carry_next, transition.second
            )
            return next_value.squeeze(-1)

        not_done = 1.0 - transition.second.done.astype(jnp.float32)
        next_value, curvature = self.critic_optimizer.bootstrap(
            state.critic_optimizer_state,
            state.critic_params,
            critic_grads,
            critic_trace,
            bootstrap_value,
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
            state.actor_optimizer_state, actor_grads, actor_trace, td_error,
        )

        critic_updates, critic_optimizer_state = self.critic_optimizer.update(
            state.critic_optimizer_state,
            critic_grads,
            critic_trace,
            td_error,
            curvature,
        )

        actor_params = jax.tree.map(
            lambda p, u: p + u, state.actor_params, actor_updates
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

    def init(self, key: Key) -> RecurrentACLambdaState:
        env_key, actor_key, critic_key, actor_carry_key, critic_carry_key = (
            jax.random.split(key, 5)
        )
        obs, env_state = self.env.reset(env_key, self.env_params)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape, dtype=canonicalize_dtype(action_space.dtype)
        )
        timestep = Timestep(
            obs=obs, action=action, reward=jnp.float32(0.0), done=jnp.bool_(True)
        )

        actor_carry = self.actor_network.initialize_carry(actor_carry_key)
        critic_carry = self.critic_network.initialize_carry(critic_carry_key)
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
            timestep.action,
            timestep.reward,
            timestep.done,
        )

        actor_optimizer_state = self.actor_optimizer.init(actor_params)
        critic_optimizer_state = self.critic_optimizer.init(critic_params)

        actor_trace = jax.tree.map(jnp.zeros_like, actor_params)
        critic_trace = jax.tree.map(jnp.zeros_like, critic_params)

        state = dict(
            step=0,
            update_step=0,
            timestep=timestep,
            actor_carry=actor_carry,
            critic_carry=critic_carry,
            env_state=env_state,
            actor_params=actor_params,
            critic_params=critic_params,
            actor_trace=actor_trace,
            critic_trace=critic_trace,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
        )

        return RecurrentACLambdaState(**state)

    def warmup(
        self, key: Key, state: RecurrentACLambdaState, num_steps: int
    ) -> RecurrentACLambdaState:
        return state

    def train(
        self, key: Key, state: RecurrentACLambdaState, num_steps: int
    ) -> RecurrentACLambdaState:
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
        state: RecurrentACLambdaState,
        num_steps: int,
        deterministic: bool = True,
    ) -> RecurrentACLambdaState:
        reset_key, actor_carry_key, critic_carry_key, eval_key = jax.random.split(
            key, 4
        )
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
            actor_carry=self.actor_network.initialize_carry(actor_carry_key),
            critic_carry=self.critic_network.initialize_carry(critic_carry_key),
            env_state=env_state,
        )

        policy = (
            self._deterministic_action if deterministic else self._stochastic_action
        )
        state, _ = jax.lax.scan(
            partial(self._step, policy=policy),
            state,
            jax.random.split(eval_key, num_steps),
        )
        return state
