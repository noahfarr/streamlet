from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
from flax import struct

from streamlet.optimizers import Optimizer
from streamlet.utils import Timestep, Transition, canonicalize_dtype
from streamlet.utils.axes import remove_feature_axis
from streamlet.utils.typing import Array, Environment, EnvParams, EnvState, Key, PyTree


@struct.dataclass(frozen=True)
class RecurrentACLambdaConfig:
    gamma: float
    trace_lambda: float
    entropy_coefficient: float = 0.01
    unroll: int = struct.field(pytree_node=False, default=4)


@struct.dataclass(frozen=True)
class RecurrentACLambdaState:
    step: int
    timestep: Timestep
    carry: PyTree
    env_state: EnvState
    params: Array
    actor_trace: Array
    critic_trace: Array
    actor_optimizer_state: PyTree
    critic_optimizer_state: PyTree


@dataclass
class RecurrentACLambda:
    cfg: RecurrentACLambdaConfig
    env: Environment
    env_params: EnvParams
    network: nn.Module
    actor_optimizer: Optimizer
    critic_optimizer: Optimizer

    def __post_init__(self):
        assert 0.0 <= self.cfg.gamma <= 1.0, (
            f"gamma must be in [0, 1], got {self.cfg.gamma}."
        )
        assert 0.0 <= self.cfg.trace_lambda <= 1.0, (
            f"trace_lambda must be in [0, 1], got {self.cfg.trace_lambda}."
        )
        assert self.cfg.entropy_coefficient >= 0.0, (
            f"entropy_coefficient must be >= 0, got {self.cfg.entropy_coefficient}."
        )
        assert self.cfg.unroll >= 1, (
            f"unroll must be >= 1, got {self.cfg.unroll}."
        )

    def env_step(
        self, state: RecurrentACLambdaState, key: Key, temperature: Array
    ) -> tuple[RecurrentACLambdaState, Transition]:
        action_key, step_key = jax.random.split(key)

        def forward(params):
            next_carry, (dist, value) = self.network.apply(
                params, state.carry, *state.timestep
            )
            action, _ = dist.sample_and_log_prob(seed=action_key)
            action = jnp.where(temperature == 0.0, dist.mode(), action)
            action = jax.lax.stop_gradient(action)
            return (
                next_carry,
                dist.log_prob(action),
                dist.entropy(),
                value,
            ), action

        ((next_carry, log_prob, entropy, value), vjp, action) = jax.vjp(
            forward, state.params, has_aux=True
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
                "critic_value": value,
                "vjp": vjp,
                "next_carry": next_carry,
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
                carry=next_carry,
                env_state=env_state,
            ),
            transition,
        )

    def update_step(
        self,
        state: RecurrentACLambdaState,
        transition: Transition,
    ) -> RecurrentACLambdaState:
        log_prob = transition.aux["log_prob"]
        entropy = transition.aux["entropy"]
        critic_value = transition.aux["critic_value"]
        vjp = transition.aux["vjp"]
        next_carry = transition.aux["next_carry"]

        carry_bar = jax.tree.map(jnp.zeros_like, next_carry)
        (critic_grads,) = vjp(
            (
                carry_bar,
                jnp.zeros_like(log_prob),
                jnp.zeros_like(entropy),
                jnp.ones_like(critic_value),
            )
        )

        critic_trace = jax.tree.map(
            lambda trace, grad: self.cfg.gamma * self.cfg.trace_lambda * trace + grad,
            state.critic_trace,
            critic_grads,
        )

        next_value, curvature = self.critic_optimizer.bootstrap(
            state.critic_optimizer_state,
            state.params,
            critic_grads,
            critic_trace,
            lambda params: remove_feature_axis(
                self.network.apply(params, next_carry, *transition.second)[1][1]
            ),
            self.cfg.gamma,
            1.0 - transition.second.done.astype(jnp.float32),
        )
        td_error = (
            transition.second.reward
            + self.cfg.gamma * (1.0 - transition.second.done) * next_value
            - remove_feature_axis(critic_value)
        )

        (actor_grads,) = vjp(
            (
                carry_bar,
                jnp.ones_like(log_prob),
                jnp.sign(td_error) * self.cfg.entropy_coefficient
                * jnp.ones_like(entropy),
                jnp.zeros_like(critic_value),
            )
        )
        actor_trace = jax.tree.map(
            lambda trace, grad: self.cfg.gamma * self.cfg.trace_lambda * trace + grad,
            state.actor_trace,
            actor_grads,
        )

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

        params = jax.tree.map(
            lambda p, au, cu: p + au + cu,
            state.params,
            actor_updates,
            critic_updates,
        )

        actor_trace = jax.tree.map(
            lambda t: jnp.where(transition.second.done, jnp.zeros_like(t), t),
            actor_trace,
        )
        critic_trace = jax.tree.map(
            lambda t: jnp.where(transition.second.done, jnp.zeros_like(t), t),
            critic_trace,
        )

        td_target = remove_feature_axis(critic_value) + td_error
        explained_variance = 1 - jnp.var(td_error) / (jnp.var(td_target) + 1e-8)
        lox.log(
            {
                "critic/value": critic_value.mean(),
                "critic/td_error": td_error.mean(),
                "critic/absolute_td_error": jnp.abs(td_error).mean(),
                "critic/explained_variance": explained_variance,
                "actor/log_prob": log_prob.mean(),
                "actor/entropy": entropy.mean(),
            }
        )

        return state.replace(
            params=params,
            actor_trace=actor_trace,
            critic_trace=critic_trace,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
        )

    def init(self, key: Key) -> RecurrentACLambdaState:
        env_key, params_key, carry_key = jax.random.split(key, 3)
        obs, env_state = self.env.reset(env_key, self.env_params)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape, dtype=canonicalize_dtype(action_space.dtype)
        )
        timestep = Timestep(
            obs=obs, action=action, reward=jnp.float32(0.0), done=jnp.bool_(True)
        )

        carry = self.network.initialize_carry(carry_key)
        params = self.network.init(params_key, carry, *timestep)

        actor_optimizer_state = self.actor_optimizer.init(params)
        critic_optimizer_state = self.critic_optimizer.init(params)

        actor_trace = jax.tree.map(jnp.zeros_like, params)
        critic_trace = jax.tree.map(jnp.zeros_like, params)

        return RecurrentACLambdaState(
            step=0,
            timestep=timestep,
            carry=carry,
            env_state=env_state,
            params=params,
            actor_trace=actor_trace,
            critic_trace=critic_trace,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
        )

    def train(
        self, key: Key, state: RecurrentACLambdaState, num_steps: int
    ) -> RecurrentACLambdaState:
        def step(state, key):
            state, transition = self.env_step(state, key, 1.0)
            return self.update_step(state, transition), None

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
        state: RecurrentACLambdaState,
        num_steps: int,
    ) -> RecurrentACLambdaState:
        reset_key, carry_key, eval_key = jax.random.split(key, 3)
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
            carry=self.network.initialize_carry(carry_key),
            env_state=env_state,
        )

        def step(state, key):
            state, _ = self.env_step(state, key, 0.0)
            return state, None

        state, _ = jax.lax.scan(step, state, jax.random.split(eval_key, num_steps))
        return state
