from dataclasses import dataclass
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
    aux_actor_loss: Callable | None = None
    aux_critic_loss: Callable | None = None
    aux_actor_coefficient: float = 1e-3
    aux_critic_coefficient: float = 1e-3

    def _env_step(
        self, state: ACLambdaState, key: Key, temperature: Array
    ) -> tuple[ACLambdaState, Transition]:
        action_key, step_key = jax.random.split(key)

        def log_prob_and_entropy(params):
            dist = self.actor_network.apply(params, state.timestep.obs)
            action, _ = dist.sample_and_log_prob(seed=action_key)
            action = jnp.where(temperature == 0.0, dist.mode(), action)
            action = jax.lax.stop_gradient(action)
            return (dist.log_prob(action), dist.entropy()), action

        (log_prob, entropy), actor_vjp, action = jax.vjp(
            log_prob_and_entropy, state.actor_params, has_aux=True
        )
        (log_prob_grads,) = actor_vjp(
            (jnp.ones_like(log_prob), jnp.zeros_like(entropy))
        )
        (entropy_grads,) = actor_vjp((jnp.zeros_like(log_prob), jnp.ones_like(entropy)))

        next_obs, env_state, reward, done, info = self.env.step(
            step_key, state.env_state, action, self.env_params
        )
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            aux={
                "log_prob": log_prob,
                "entropy": entropy,
                "log_prob_grads": log_prob_grads,
                "entropy_grads": entropy_grads,
            },
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
        self,
        state: ACLambdaState,
        transition: Transition,
    ) -> ACLambdaState:
        log_prob_grads = transition.aux["log_prob_grads"]
        entropy_grads = transition.aux["entropy_grads"]

        critic_value, critic_vjp = jax.vjp(
            lambda params: self.critic_network.apply(
                params, transition.first.obs
            ).squeeze(-1),
            state.critic_params,
        )
        (critic_grads,) = critic_vjp(jnp.ones_like(critic_value))

        critic_trace = jax.tree.map(
            lambda trace, grad: self.cfg.gamma * self.cfg.trace_lambda * trace + grad,
            state.critic_trace,
            critic_grads,
        )

        next_value, curvature = self.critic_optimizer.bootstrap(
            state.critic_optimizer_state,
            state.critic_params,
            critic_grads,
            critic_trace,
            lambda params: self.critic_network.apply(
                params, transition.second.obs
            ).squeeze(-1),
            self.cfg.gamma,
            1.0 - transition.second.done.astype(jnp.float32),
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
        actor_trace = jax.tree.map(
            lambda trace, grad: self.cfg.gamma * self.cfg.trace_lambda * trace + grad,
            state.actor_trace,
            actor_grads,
        )

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

        if self.aux_actor_loss is not None:
            aux_actor_grads = jax.grad(self.aux_actor_loss)(actor_params, transition)
            actor_params = jax.tree.map(
                lambda p, g: p - self.aux_actor_coefficient * g,
                actor_params,
                aux_actor_grads,
            )

        if self.aux_critic_loss is not None:
            aux_critic_grads = jax.grad(self.aux_critic_loss)(
                critic_params, transition
            )
            critic_params = jax.tree.map(
                lambda p, g: p - self.aux_critic_coefficient * g,
                critic_params,
                aux_critic_grads,
            )

        actor_trace = jax.tree.map(
            lambda t: jnp.where(transition.second.done, jnp.zeros_like(t), t),
            actor_trace,
        )
        critic_trace = jax.tree.map(
            lambda t: jnp.where(transition.second.done, jnp.zeros_like(t), t),
            critic_trace,
        )

        target = critic_value + td_error
        explained_variance = 1 - jnp.var(td_error) / (jnp.var(target) + 1e-8)
        lox.log(
            {
                "critic/value": critic_value.mean(),
                "critic/td_error": td_error.mean(),
                "critic/absolute_td_error": jnp.abs(td_error).mean(),
                "critic/explained_variance": explained_variance,
                "actor/log_prob": transition.aux["log_prob"].mean(),
                "actor/entropy": transition.aux["entropy"].mean(),
            }
        )

        return state.replace(
            actor_params=actor_params,
            critic_params=critic_params,
            actor_trace=actor_trace,
            critic_trace=critic_trace,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
        )

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

        return ACLambdaState(
            step=0,
            timestep=timestep,
            env_state=env_state,
            actor_params=actor_params,
            critic_params=critic_params,
            actor_trace=actor_trace,
            critic_trace=critic_trace,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
        )

    def train(self, key: Key, state: ACLambdaState, num_steps: int) -> ACLambdaState:
        def step(state, key):
            state, transition = self._env_step(state, key, 1.0)
            return self._update_step(state, transition), None

        state, _ = jax.lax.scan(
            step,
            state,
            jax.random.split(key, num_steps),
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

        def step(state, key):
            state, _ = self._env_step(state, key, 0.0)
            return state, None

        state, _ = jax.lax.scan(step, state, jax.random.split(eval_key, num_steps))
        return state
