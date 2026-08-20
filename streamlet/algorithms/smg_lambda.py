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


def carry_state(carry: PyTree) -> PyTree:
    while hasattr(carry, "carry"):
        carry = carry.carry
    return carry


def flatten_carry(carry: PyTree) -> Array:
    return jnp.concatenate(
        [leaf.reshape(-1) for leaf in jax.tree.leaves(carry_state(carry))]
    )


@struct.dataclass(frozen=True)
class SMGLambdaConfig:
    gamma: float
    alpha: float
    trace_lambda: float = 0.0
    memory_decay: float = 0.9
    memory_coefficient: float = 1.0
    unroll: int = struct.field(pytree_node=False, default=4)


@struct.dataclass(frozen=True)
class SMGLambdaState:
    step: int
    timestep: Timestep
    env_state: EnvState
    actor_params: core.FrozenDict[str, Any]
    actor_optimizer_state: PyTree
    actor_carry: PyTree
    critic_params: core.FrozenDict[str, Any]
    critic_optimizer_state: PyTree
    critic_carry: PyTree
    critic_trace: PyTree
    memory_trace: PyTree
    td_scaler: TDErrorScalerState


@dataclass
class SMGLambda:
    cfg: SMGLambdaConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    actor_optimizer: Optimizer
    critic_optimizer: Optimizer
    auxiliary_actor_loss: Callable | None = None
    auxiliary_critic_loss: Callable | None = None

    def __post_init__(self):
        action_space = self.env.action_space(self.env_params)
        assert isinstance(action_space, Box), (
            "SMGLambda requires a continuous (Box) action space, got "
            f"{type(action_space).__name__}."
        )
        assert 0.0 <= self.cfg.gamma <= 1.0, (
            f"gamma must be in [0, 1], got {self.cfg.gamma}."
        )
        assert 0.0 <= self.cfg.trace_lambda <= 1.0, (
            f"trace_lambda must be in [0, 1], got {self.cfg.trace_lambda}."
        )
        assert 0.0 <= self.cfg.memory_decay <= 1.0, (
            f"memory_decay must be in [0, 1], got {self.cfg.memory_decay}."
        )
        assert self.cfg.alpha >= 0.0, (
            f"alpha (entropy temperature) must be >= 0, got {self.cfg.alpha}."
        )
        assert self.cfg.unroll >= 1, f"unroll must be >= 1, got {self.cfg.unroll}."

    def env_step(
        self, state: SMGLambdaState, key: Key, temperature: Array
    ) -> tuple[SMGLambdaState, Transition]:
        sample_key, step_key, next_action_key = jax.random.split(key, 3)

        def actor_forward(params):
            return self.actor_network.apply(params, state.actor_carry, *state.timestep)

        actor_carry, dist = actor_forward(state.actor_params)
        action, log_prob = dist.sample_and_log_prob(seed=sample_key)
        greedy = dist.bijector.forward(dist.distribution.mean())
        action = jnp.where(temperature == 0.0, greedy, action)
        log_prob = jnp.where(temperature == 0.0, dist.log_prob(greedy), log_prob)

        action_jacobian = jax.jacrev(
            lambda params: actor_forward(params)[1].sample(seed=sample_key)
        )(state.actor_params)

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
                "actor_carry": state.actor_carry,
                "critic_carry": state.critic_carry,
                "next_actor_carry": actor_carry,
                "action_jacobian": action_jacobian,
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
                actor_carry=actor_carry,
            ),
            transition,
        )

    def update_step(
        self, state: SMGLambdaState, transition: Transition
    ) -> SMGLambdaState:
        log_prob = transition.aux["log_prob"]
        sample_key = transition.aux["sample_key"]
        next_action_key = transition.aux["next_action_key"]
        actor_carry = transition.aux["actor_carry"]
        critic_carry = transition.aux["critic_carry"]
        next_actor_carry = transition.aux["next_actor_carry"]
        action_jacobian = transition.aux["action_jacobian"]
        not_done = 1.0 - transition.second.done.astype(jnp.float32)

        _, next_dist = self.actor_network.apply(
            jax.lax.stop_gradient(state.actor_params),
            next_actor_carry,
            *transition.second,
        )
        next_action, next_log_prob = next_dist.sample_and_log_prob(
            seed=next_action_key
        )

        def critic_forward(params, carry, timestep, action):
            carry, value = self.critic_network.apply(
                params, carry, *timestep, action=action
            )
            return remove_feature_axis(value), carry

        (q_value, critic_carry_out), q_grads = jax.value_and_grad(
            lambda params: critic_forward(
                params, critic_carry, transition.first, transition.second.action
            ),
            has_aux=True,
        )(state.critic_params)

        next_q_value, _ = critic_forward(
            jax.lax.stop_gradient(state.critic_params),
            critic_carry_out,
            transition.second,
            next_action,
        )
        next_value = next_q_value - self.cfg.alpha * next_log_prob

        entropy_reward = transition.second.reward - self.cfg.alpha * log_prob
        td_scaler = state.td_scaler.update(
            entropy_reward, transition.second.done, self.cfg.gamma
        )
        sigma = td_scaler.sigma()

        td_error = (
            transition.second.reward + not_done * self.cfg.gamma * next_value - q_value
        ) / sigma

        def state_action_value(carry, action):
            value, _ = critic_forward(
                jax.lax.stop_gradient(state.critic_params),
                carry,
                transition.first,
                action,
            )
            return value

        def compute_actor_loss(actor_params):
            _, dist = self.actor_network.apply(
                actor_params, actor_carry, *transition.first
            )
            sampled_action, sampled_log_prob = dist.sample_and_log_prob(seed=sample_key)
            sampled_q = state_action_value(critic_carry, sampled_action)
            return self.cfg.alpha * sampled_log_prob - sampled_q

        actor_loss, actor_grads = jax.value_and_grad(compute_actor_loss)(
            state.actor_params
        )

        carry_cotangent = flatten_carry(
            jax.grad(lambda carry: state_action_value(carry, transition.second.action))(
                critic_carry
            )
        )


        memory_grads = jax.tree.map(
            lambda trace: jnp.tensordot(carry_cotangent, trace, axes=([0], [0])),
            state.memory_trace,
        )
        actor_grads = jax.tree.map(
            lambda g, m: g - self.cfg.memory_coefficient * m,
            actor_grads,
            memory_grads,
        )

        actor_ascent = jax.tree.map(jnp.negative, actor_grads)
        actor_updates, actor_optimizer_state = self.actor_optimizer.update(
            state.actor_optimizer_state, actor_grads, actor_ascent, jnp.float32(1.0)
        )
        actor_params = jax.tree.map(
            lambda p, u: p + u, state.actor_params, actor_updates
        )

        def carry_of_previous_action(previous_action):
            _, carry = critic_forward(
                jax.lax.stop_gradient(state.critic_params),
                critic_carry,
                transition.first.replace(action=previous_action),
                transition.second.action,
            )
            return flatten_carry(carry)

        action_sensitivity = jax.jacrev(carry_of_previous_action)(
            transition.first.action
        )

        memory_trace = jax.tree.map(
            lambda trace, jacobian: self.cfg.memory_decay * not_done * trace
            + jnp.tensordot(action_sensitivity, jacobian, axes=([1], [0])),
            state.memory_trace,
            action_jacobian,
        )

        critic_trace = jax.tree.map(
            lambda trace, grad: self.cfg.gamma
            * self.cfg.trace_lambda
            * not_done
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

        td_target = q_value + td_error
        explained_variance = 1 - jnp.var(td_error) / (jnp.var(td_target) + 1e-8)
        memory_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(memory_grads))
        )
        actor_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(actor_grads))
        )
        lox.log(
            {
                "actor/loss": actor_loss.mean(),
                "actor/log_prob": log_prob.mean(),
                "actor/gradient_norm": actor_norm,
                "actor/memory_gradient_norm": memory_norm,
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
            critic_carry=critic_carry_out,
            critic_trace=critic_trace,
            memory_trace=memory_trace,
            td_scaler=td_scaler,
        )

    def init(self, key: Key) -> SMGLambdaState:
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

        actor_params = self.actor_network.init(actor_key, actor_carry, *timestep)
        critic_params = self.critic_network.init(
            critic_key, critic_carry, *timestep, action=action
        )

        actor_optimizer_state = self.actor_optimizer.init(actor_params)
        critic_optimizer_state = self.critic_optimizer.init(critic_params)

        critic_trace = jax.tree.map(jnp.zeros_like, critic_params)
        carry_size = sum(
            leaf.size for leaf in jax.tree.leaves(carry_state(critic_carry))
        )
        memory_trace = jax.tree.map(
            lambda p: jnp.zeros((carry_size, *p.shape)), actor_params
        )

        return SMGLambdaState(
            step=0,
            timestep=timestep,
            env_state=env_state,
            actor_params=actor_params,
            actor_optimizer_state=actor_optimizer_state,
            actor_carry=actor_carry,
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_carry=critic_carry,
            critic_trace=critic_trace,
            memory_trace=memory_trace,
            td_scaler=TDErrorScalerState.init(),
        )

    def train(self, key: Key, state: SMGLambdaState, num_steps: int) -> SMGLambdaState:
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
        state: SMGLambdaState,
        num_steps: int,
    ) -> SMGLambdaState:
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
            env_state=env_state,
            actor_carry=self.actor_network.initialize_carry(actor_carry_key),
            critic_carry=self.critic_network.initialize_carry(critic_carry_key),
        )

        def step(state, key):
            state, _ = self.env_step(state, key, 0.0)
            return state, None

        state, _ = jax.lax.scan(step, state, jax.random.split(eval_key, num_steps))
        return state
