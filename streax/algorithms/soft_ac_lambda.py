from dataclasses import dataclass
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
from flax import core, struct

from streax.optimizers import Optimizer
from streax.utils import Timestep, Transition, canonicalize_dtype
from streax.utils.axes import remove_feature_axis
from streax.utils.typing import Array, Environment, EnvParams, EnvState, Key, PyTree


@struct.dataclass(frozen=True)
class SoftACLambdaConfig:
    gamma: float
    trace_lambda: float
    tau: float = 0.1
    entropy_coefficient: float = 0.01
    target_entropy_ratio: float = 0.0
    temperature_lr: float = 1e-4
    max_temperature: float = 1.0
    unroll: int = struct.field(pytree_node=False, default=4)


@struct.dataclass(frozen=True)
class SoftACLambdaState:
    step: int
    timestep: Timestep
    env_state: EnvState
    actor_params: core.FrozenDict[str, Any]
    critic_params: core.FrozenDict[str, Any]
    actor_trace: PyTree
    critic_trace: PyTree
    actor_optimizer_state: PyTree
    critic_optimizer_state: PyTree
    log_temperature: Array


@dataclass
class SoftACLambda:
    """Streaming soft actor-critic, AC(lambda) form.

    Maximum-entropy AC(lambda) with eligibility traces and ObGD. The change vs.
    :class:`ACLambda` is that the per-step reward is augmented with
    ``tau * H(pi(.|s))``, so the critic learns the entropy-regularized soft value
    ``V(s) = E[sum_t gamma^t (r_t + tau * H(pi(.|s_t)))]`` and the augmented TD
    error drives both actor and critic. The actor keeps its entropy-gradient
    bonus (``entropy_coefficient``). ``tau = 0`` recovers plain AC(lambda).
    """

    cfg: SoftACLambdaConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    actor_optimizer: Optimizer
    critic_optimizer: Optimizer
    aux_actor_loss: Callable | None = None
    aux_critic_loss: Callable | None = None

    def __post_init__(self):
        assert 0.0 <= self.cfg.gamma <= 1.0, (
            f"gamma must be in [0, 1], got {self.cfg.gamma}."
        )
        assert 0.0 <= self.cfg.trace_lambda <= 1.0, (
            f"trace_lambda must be in [0, 1], got {self.cfg.trace_lambda}."
        )
        assert self.cfg.tau >= 0.0, f"tau must be >= 0, got {self.cfg.tau}."
        assert self.cfg.entropy_coefficient >= 0.0, (
            f"entropy_coefficient must be >= 0, got {self.cfg.entropy_coefficient}."
        )
        assert self.cfg.unroll >= 1, f"unroll must be >= 1, got {self.cfg.unroll}."
        action_space = self.env.action_space(self.env_params)
        self.max_entropy = (
            float(jnp.log(action_space.n)) if hasattr(action_space, "n") else 1.0
        )

    def _env_step(
        self, state: SoftACLambdaState, key: Key, temperature: Array
    ) -> tuple[SoftACLambdaState, Transition]:
        action_key, step_key = jax.random.split(key)

        def log_prob_and_entropy(params):
            dist, intermediates = self.actor_network.apply(
                params, state.timestep.obs, mutable=["intermediates"]
            )
            action, _ = dist.sample_and_log_prob(seed=action_key)
            action = jnp.where(temperature == 0.0, dist.mode(), action)
            action = jax.lax.stop_gradient(action)
            return (dist.log_prob(action), dist.entropy(), intermediates), action

        (log_prob, entropy, intermediates), actor_vjp, action = jax.vjp(
            log_prob_and_entropy, state.actor_params, has_aux=True
        )
        (log_prob_grads,) = actor_vjp(
            (
                jnp.ones_like(log_prob),
                jnp.zeros_like(entropy),
                jax.tree.map(jnp.zeros_like, intermediates),
            )
        )
        (entropy_grads,) = actor_vjp(
            (
                jnp.zeros_like(log_prob),
                jnp.ones_like(entropy),
                jax.tree.map(jnp.zeros_like, intermediates),
            )
        )

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
                "intermediates": intermediates,
                "actor_vjp": actor_vjp,
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
        state: SoftACLambdaState,
        transition: Transition,
    ) -> SoftACLambdaState:
        log_prob = transition.aux["log_prob"]
        entropy = transition.aux["entropy"]
        log_prob_grads = transition.aux["log_prob_grads"]
        entropy_grads = transition.aux["entropy_grads"]
        actor_intermediates = transition.aux["intermediates"]
        actor_vjp = transition.aux["actor_vjp"]

        (critic_value, critic_intermediates), critic_vjp = jax.vjp(
            lambda params: self.critic_network.apply(
                params, transition.first.obs, mutable=["intermediates"]
            ),
            state.critic_params,
        )
        (critic_grads,) = critic_vjp(
            (jnp.ones_like(critic_value), jax.tree.map(jnp.zeros_like, critic_intermediates))
        )

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
            lambda params: remove_feature_axis(
                self.critic_network.apply(params, transition.second.obs)
            ),
            self.cfg.gamma,
            1.0 - transition.second.done.astype(jnp.float32),
        )
        temperature = jnp.exp(state.log_temperature)
        soft_reward = transition.second.reward + temperature * entropy
        td_error = (
            soft_reward
            + self.cfg.gamma * (1.0 - transition.second.done) * next_value
            - remove_feature_axis(critic_value)
        )

        controller_on = self.cfg.target_entropy_ratio > 0.0
        entropy_scale = jnp.where(
            controller_on, temperature, self.cfg.entropy_coefficient
        )
        actor_grads = jax.tree.map(
            lambda lpg, eg: lpg + jnp.sign(td_error) * entropy_scale * eg,
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
            _, next_intermediates = self.actor_network.apply(
                state.actor_params, transition.second.obs, mutable=["intermediates"]
            )
            actor_transition = transition.replace(
                aux={"intermediates": actor_intermediates, "next_intermediates": next_intermediates}
            )
            cotangents = jax.grad(
                lambda i: self.aux_actor_loss(
                    actor_transition.replace(aux={**actor_transition.aux, "intermediates": i})
                )
            )(actor_intermediates)
            (aux_actor_grads,) = actor_vjp(
                (jnp.zeros_like(log_prob), jnp.zeros_like(entropy), cotangents)
            )
            actor_params = jax.tree.map(
                lambda p, g: p - g,
                actor_params,
                aux_actor_grads,
            )

        if self.aux_critic_loss is not None:
            _, next_intermediates = self.critic_network.apply(
                state.critic_params, transition.second.obs, mutable=["intermediates"]
            )
            critic_transition = transition.replace(
                aux={"intermediates": critic_intermediates, "next_intermediates": next_intermediates}
            )
            cotangents = jax.grad(
                lambda i: self.aux_critic_loss(
                    critic_transition.replace(aux={**critic_transition.aux, "intermediates": i})
                )
            )(critic_intermediates)
            (aux_critic_grads,) = critic_vjp((jnp.zeros_like(critic_value), cotangents))
            critic_params = jax.tree.map(
                lambda p, g: p - g,
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

        target_entropy = self.cfg.target_entropy_ratio * self.max_entropy
        log_temperature = jnp.where(
            controller_on,
            jnp.clip(
                state.log_temperature
                + self.cfg.temperature_lr * (target_entropy - entropy),
                jnp.log(1e-6),
                jnp.log(self.cfg.max_temperature),
            ),
            state.log_temperature,
        )

        td_target = remove_feature_axis(critic_value) + td_error
        explained_variance = 1 - jnp.var(td_error) / (jnp.var(td_target) + 1e-8)
        lox.log(
            {
                "critic/value": critic_value.mean(),
                "critic/td_error": td_error.mean(),
                "critic/absolute_td_error": jnp.abs(td_error).mean(),
                "critic/explained_variance": explained_variance,
                "actor/log_prob": transition.aux["log_prob"].mean(),
                "actor/entropy": transition.aux["entropy"].mean(),
                "actor/temperature": temperature.mean(),
                "actor/soft_reward_bonus": (temperature * entropy).mean(),
            }
        )

        return state.replace(
            actor_params=actor_params,
            critic_params=critic_params,
            actor_trace=actor_trace,
            critic_trace=critic_trace,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
            log_temperature=log_temperature,
        )

    def init(self, key: Key) -> SoftACLambdaState:
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

        return SoftACLambdaState(
            step=0,
            timestep=timestep,
            env_state=env_state,
            actor_params=actor_params,
            critic_params=critic_params,
            actor_trace=actor_trace,
            critic_trace=critic_trace,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
            log_temperature=jnp.log(jnp.maximum(self.cfg.tau, 1e-8)),
        )

    def train(
        self, key: Key, state: SoftACLambdaState, num_steps: int
    ) -> SoftACLambdaState:
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
        state: SoftACLambdaState,
        num_steps: int,
    ) -> SoftACLambdaState:
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
