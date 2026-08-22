from dataclasses import dataclass
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
from flax import struct
from jax.scipy.special import logsumexp

from streamlet.optimizers import Optimizer
from streamlet.utils import Timestep, Transition, canonicalize_dtype
from streamlet.utils.typing import (
    Array,
    Discrete,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)


@struct.dataclass(frozen=True)
class SoftRecurrentQLambdaConfig:
    gamma: float
    trace_lambda: float
    tau: float = 0.03
    unroll: int = struct.field(pytree_node=False, default=4)


@struct.dataclass(frozen=True)
class SoftRecurrentQLambdaState:
    step: int
    timestep: Timestep
    carry: PyTree
    env_state: EnvState
    params: Array
    q_trace: Array
    q_optimizer_state: PyTree


@dataclass
class SoftRecurrentQLambda:
    cfg: SoftRecurrentQLambdaConfig
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    q_optimizer: Optimizer
    auxiliary_loss: Callable | None = None

    def __post_init__(self):
        action_space = self.env.action_space(self.env_params)
        assert isinstance(action_space, Discrete), (
            "SoftRecurrentQLambda requires a discrete action space, got "
            f"{type(action_space).__name__}."
        )
        assert 0.0 <= self.cfg.gamma <= 1.0, (
            f"gamma must be in [0, 1], got {self.cfg.gamma}."
        )
        assert 0.0 <= self.cfg.trace_lambda <= 1.0, (
            f"trace_lambda must be in [0, 1], got {self.cfg.trace_lambda}."
        )
        assert self.cfg.tau > 0.0, f"tau must be > 0, got {self.cfg.tau}."
        assert self.cfg.unroll >= 1, f"unroll must be >= 1, got {self.cfg.unroll}."

    def soft_value(self, q_values: Array) -> Array:
        return self.cfg.tau * logsumexp(q_values / self.cfg.tau, axis=-1)

    def log_policy(self, q_values: Array) -> Array:
        return jax.nn.log_softmax(q_values / self.cfg.tau)

    def env_step(
        self, state: SoftRecurrentQLambdaState, key: Key
    ) -> tuple[SoftRecurrentQLambdaState, Transition]:
        sample_key, step_key = jax.random.split(key, 2)

        action_space = self.env.action_space(self.env_params)

        ((next_carry, q_values), auxiliary_losses), q_vjp = jax.vjp(
            lambda params: self.q_network.apply(
                params, state.carry, *state.timestep, mutable=["auxiliary_losses"]
            ),
            state.params,
        )

        log_probabilities = self.log_policy(q_values)
        action = jax.random.categorical(sample_key, log_probabilities)
        probabilities = jnp.exp(log_probabilities)
        lox.log({
            "policy/entropy": -jnp.sum(probabilities * log_probabilities),
            "policy/max_probability": probabilities.max(),
        })

        q_value = q_values[action]
        (q_grads,) = q_vjp((
            (
                jax.tree.map(jnp.zeros_like, next_carry),
                jax.nn.one_hot(action, action_space.n, dtype=q_values.dtype),
            ),
            jax.tree.map(jnp.zeros_like, auxiliary_losses),
        ))

        next_obs, env_state, reward, done, info = self.env.step(
            step_key, state.env_state, action, self.env_params
        )
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            aux={
                "action_probability": probabilities[action],
                "q_value": q_value,
                "q_values": q_values,
                "q_grads": q_grads,
                "auxiliary_losses": auxiliary_losses,
                "q_vjp": q_vjp,
                "carry": state.carry,
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
        state: SoftRecurrentQLambdaState,
        transition: Transition,
    ) -> SoftRecurrentQLambdaState:
        action_probability = transition.aux["action_probability"]
        q_value = transition.aux["q_value"]
        q_values = transition.aux["q_values"]
        q_grads = transition.aux["q_grads"]
        auxiliary_losses = transition.aux["auxiliary_losses"]
        q_vjp = transition.aux["q_vjp"]
        next_carry = transition.aux["next_carry"]

        trace_decay = self.cfg.gamma * self.cfg.trace_lambda * action_probability
        q_trace = jax.tree.map(
            lambda trace, grad: trace_decay * trace + grad,
            state.q_trace,
            q_grads,
        )

        next_value, curvature = self.q_optimizer.bootstrap(
            state.q_optimizer_state,
            state.params,
            q_grads,
            q_trace,
            lambda params: self.soft_value(
                self.q_network.apply(params, next_carry, *transition.second)[1]
            ),
            self.cfg.gamma,
            1.0 - transition.second.done.astype(jnp.float32),
        )
        td_error = (
            transition.second.reward
            + self.cfg.gamma * next_value * (1.0 - transition.second.done)
            - q_value
        )
        q_updates, q_optimizer_state = self.q_optimizer.update(
            state.q_optimizer_state,
            q_grads,
            q_trace,
            td_error,
            curvature,
        )

        params = jax.tree.map(lambda p, u: p + u, state.params, q_updates)

        if self.auxiliary_loss is not None:
            _, next_auxiliary_losses = self.q_network.apply(
                state.params, next_carry, *transition.second, mutable=["auxiliary_losses"]
            )
            transition = transition.replace(
                aux={**transition.aux, "next_auxiliary_losses": next_auxiliary_losses}
            )
            cotangents = jax.grad(
                lambda i: self.auxiliary_loss(
                    transition.replace(aux={**transition.aux, "auxiliary_losses": i})
                )
            )(auxiliary_losses)
            (aux_grads,) = q_vjp((
                (jax.tree.map(jnp.zeros_like, next_carry), jnp.zeros_like(q_values)),
                cotangents,
            ))
            params = jax.tree.map(lambda p, g: p - g, params, aux_grads)

        q_trace = jax.tree.map(
            lambda t: jnp.where(transition.second.done, jnp.zeros_like(t), t),
            q_trace,
        )

        lox.log(
            {
                "q_network/q_value": q_value.mean(),
                "q_network/td_error": td_error.mean(),
                "q_network/absolute_td_error": jnp.abs(td_error).mean(),
                "q_network/soft_value": next_value.mean(),
                "policy/trace_decay": trace_decay,
            }
        )

        return state.replace(
            params=params,
            q_trace=q_trace,
            q_optimizer_state=q_optimizer_state,
        )

    def init(self, key: Key) -> SoftRecurrentQLambdaState:
        env_key, q_key, carry_key = jax.random.split(key, 3)
        obs, env_state = self.env.reset(env_key, self.env_params)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape,
            dtype=canonicalize_dtype(action_space.dtype),
        )
        timestep = Timestep(
            obs=obs, action=action, reward=jnp.float32(0.0), done=jnp.bool_(True)
        )

        carry = self.q_network.initialize_carry(carry_key)
        params = self.q_network.init(q_key, carry, *timestep)

        q_optimizer_state = self.q_optimizer.init(params)

        q_trace = jax.tree.map(jnp.zeros_like, params)

        return SoftRecurrentQLambdaState(
            step=0,
            timestep=timestep,
            carry=carry,
            env_state=env_state,
            params=params,
            q_trace=q_trace,
            q_optimizer_state=q_optimizer_state,
        )

    def train(
        self, key: Key, state: SoftRecurrentQLambdaState, num_steps: int
    ) -> SoftRecurrentQLambdaState:
        def step(state, key):
            state, transition = self.env_step(state, key)
            return self.update_step(state, transition), None

        state, _ = jax.lax.scan(
            step,
            state,
            jax.random.split(key, num_steps),
            unroll=self.cfg.unroll,
        )
        return state

    def evaluate(
        self, key: Key, state: SoftRecurrentQLambdaState, num_steps: int
    ) -> SoftRecurrentQLambdaState:
        reset_key, carry_key, eval_key = jax.random.split(key, 3)
        obs, env_state = self.env.reset(reset_key, self.env_params)

        action_space = self.env.action_space(self.env_params)
        state = state.replace(
            step=0,
            timestep=Timestep(
                obs=obs,
                action=jnp.zeros(
                    action_space.shape,
                    dtype=canonicalize_dtype(action_space.dtype),
                ),
                reward=jnp.float32(0.0),
                done=jnp.bool_(True),
            ),
            carry=self.q_network.initialize_carry(carry_key),
            env_state=env_state,
        )

        def step(state, key):
            state, _ = self.env_step(state, key)
            return state, None

        state, _ = jax.lax.scan(step, state, jax.random.split(eval_key, num_steps))
        return state
