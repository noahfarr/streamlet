from dataclasses import dataclass
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
from flax import core, struct

from streamlet.optimizers import Optimizer
from streamlet.utils import Timestep, Transition, TDErrorScalerState, canonicalize_dtype
from streamlet.utils.axes import remove_feature_axis
from streamlet.utils.typing import (
    Array,
    Box,
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
    timestep: Timestep
    env_state: EnvState
    actor_params: core.FrozenDict[str, Any]
    actor_optimizer_state: PyTree
    critic_params: core.FrozenDict[str, Any]
    critic_optimizer_state: PyTree
    critic_trace: PyTree
    td_scaler: TDErrorScalerState


@dataclass
class AVGLambda:
    cfg: AVGLambdaConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    actor_optimizer: Optimizer
    critic_optimizer: Optimizer
    aux_actor_loss: Callable | None = None
    aux_critic_loss: Callable | None = None

    def __post_init__(self):
        action_space = self.env.action_space(self.env_params)
        assert isinstance(action_space, Box), (
            "AVGLambda requires a continuous (Box) action space, got "
            f"{type(action_space).__name__}."
        )
        assert 0.0 <= self.cfg.gamma <= 1.0, (
            f"gamma must be in [0, 1], got {self.cfg.gamma}."
        )
        assert 0.0 <= self.cfg.trace_lambda <= 1.0, (
            f"trace_lambda must be in [0, 1], got {self.cfg.trace_lambda}."
        )
        assert self.cfg.alpha >= 0.0, (
            f"alpha (entropy temperature) must be >= 0, got {self.cfg.alpha}."
        )
        assert self.cfg.unroll >= 1, (
            f"unroll must be >= 1, got {self.cfg.unroll}."
        )

    def _env_step(
        self, state: AVGLambdaState, key: Key, temperature: Array
    ) -> tuple[AVGLambdaState, Transition]:
        sample_key, step_key, next_action_key = jax.random.split(key, 3)

        dist = self.actor_network.apply(state.actor_params, state.timestep.obs)
        action, log_prob = dist.sample_and_log_prob(seed=sample_key)
        mode = dist.mode()
        action = jnp.where(temperature == 0.0, mode, action)
        log_prob = jnp.where(temperature == 0.0, dist.log_prob(mode), log_prob)

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
                "sample_key": sample_key,
                "next_action_key": next_action_key,
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
        self, state: AVGLambdaState, transition: Transition
    ) -> AVGLambdaState:
        log_prob = transition.aux["log_prob"]
        sample_key = transition.aux["sample_key"]
        next_action_key = transition.aux["next_action_key"]

        next_dist = self.actor_network.apply(
            jax.lax.stop_gradient(state.actor_params), transition.second.obs
        )
        next_action, next_log_prob = next_dist.sample_and_log_prob(
            seed=next_action_key
        )
        next_q_value = remove_feature_axis(
            self.critic_network.apply(
                jax.lax.stop_gradient(state.critic_params),
                transition.second.obs,
                next_action,
            )
        )
        next_value = next_q_value - self.cfg.alpha * next_log_prob

        entropy_reward = transition.second.reward - self.cfg.alpha * log_prob
        td_scaler = state.td_scaler.update(
            entropy_reward, transition.second.done, self.cfg.gamma
        )
        sigma = td_scaler.sigma()

        q_value, q_grads = jax.value_and_grad(
            lambda params: remove_feature_axis(
                self.critic_network.apply(
                    params, transition.first.obs, transition.second.action
                )
            )
        )(state.critic_params)
        td_error = (
            transition.second.reward
            + (1.0 - transition.second.done.astype(jnp.float32))
            * self.cfg.gamma
            * next_value
            - q_value
        ) / sigma

        def compute_actor_loss(actor_params):
            dist = self.actor_network.apply(actor_params, transition.first.obs)
            sampled_action, sampled_log_prob = dist.sample_and_log_prob(
                seed=sample_key
            )
            sampled_q = remove_feature_axis(
                self.critic_network.apply(
                    jax.lax.stop_gradient(state.critic_params),
                    transition.first.obs,
                    sampled_action,
                )
            )
            return self.cfg.alpha * sampled_log_prob - sampled_q

        actor_loss, actor_grads = jax.value_and_grad(compute_actor_loss)(
            state.actor_params
        )
        actor_ascent = jax.tree.map(jnp.negative, actor_grads)
        actor_updates, actor_optimizer_state = self.actor_optimizer.update(
            state.actor_optimizer_state, actor_grads, actor_ascent, jnp.float32(1.0)
        )
        actor_params = jax.tree.map(
            lambda p, u: p + u, state.actor_params, actor_updates
        )

        critic_trace = jax.tree.map(
            lambda trace, grad: self.cfg.gamma
            * self.cfg.trace_lambda
            * (1.0 - transition.first.done.astype(jnp.float32))
            * trace
            + grad,
            state.critic_trace,
            q_grads,
        )

        critic_updates, critic_optimizer_state = self.critic_optimizer.update(
            state.critic_optimizer_state, q_grads, critic_trace, td_error
        )
        critic_params = jax.tree.map(
            lambda p, u: p + u, state.critic_params, critic_updates
        )

        if self.aux_actor_loss is not None:
            (dist, intermediates), actor_vjp = jax.vjp(
                lambda params: self.actor_network.apply(
                    params, transition.first.obs, mutable=["intermediates"]
                ),
                state.actor_params,
            )
            _, next_intermediates = self.actor_network.apply(
                state.actor_params, transition.second.obs, mutable=["intermediates"]
            )
            actor_transition = transition.replace(
                aux={"intermediates": intermediates, "next_intermediates": next_intermediates}
            )
            cotangents = jax.grad(
                lambda i: self.aux_actor_loss(
                    actor_transition.replace(aux={**actor_transition.aux, "intermediates": i})
                )
            )(intermediates)
            (aux_actor_grads,) = actor_vjp((jax.tree.map(jnp.zeros_like, dist), cotangents))
            actor_params = jax.tree.map(
                lambda p, g: p - g,
                actor_params,
                aux_actor_grads,
            )

        if self.aux_critic_loss is not None:
            (critic_out, critic_intermediates), critic_vjp = jax.vjp(
                lambda params: self.critic_network.apply(
                    params, transition.first.obs, transition.second.action,
                    mutable=["intermediates"],
                ),
                state.critic_params,
            )
            _, next_intermediates = self.critic_network.apply(
                state.critic_params, transition.second.obs, transition.second.action,
                mutable=["intermediates"],
            )
            critic_transition = transition.replace(
                aux={"intermediates": critic_intermediates, "next_intermediates": next_intermediates}
            )
            cotangents = jax.grad(
                lambda i: self.aux_critic_loss(
                    critic_transition.replace(aux={**critic_transition.aux, "intermediates": i})
                )
            )(critic_intermediates)
            (aux_critic_grads,) = critic_vjp((jnp.zeros_like(critic_out), cotangents))
            critic_params = jax.tree.map(
                lambda p, g: p - g,
                critic_params,
                aux_critic_grads,
            )

        td_target = q_value + td_error
        explained_variance = 1 - jnp.var(td_error) / (jnp.var(td_target) + 1e-8)
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

        return state.replace(
            actor_params=actor_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_trace=critic_trace,
            td_scaler=td_scaler,
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

        critic_trace = jax.tree.map(jnp.zeros_like, critic_params)

        return AVGLambdaState(
            step=0,
            timestep=timestep,
            env_state=env_state,
            actor_params=actor_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_trace=critic_trace,
            td_scaler=TDErrorScalerState.init(),
        )

    def train(self, key: Key, state: AVGLambdaState, num_steps: int) -> AVGLambdaState:
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

        def step(state, key):
            state, _ = self._env_step(state, key, 0.0)
            return state, None

        state, _ = jax.lax.scan(step, state, jax.random.split(eval_key, num_steps))
        return state
