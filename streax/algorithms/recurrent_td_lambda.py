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
class RecurrentTDLambdaConfig:
    gamma: float
    trace_lambda: float
    unroll: int = struct.field(pytree_node=False, default=4)


@struct.dataclass(frozen=True)
class RecurrentTDLambdaState:
    step: int
    timestep: Timestep
    carry: PyTree
    env_state: EnvState
    value_params: core.FrozenDict[str, Any]
    value_trace: PyTree
    value_optimizer_state: PyTree


@dataclass
class RecurrentTDLambda:
    cfg: RecurrentTDLambdaConfig
    env: Environment
    env_params: EnvParams
    value_network: nn.Module
    value_optimizer: Optimizer
    aux_loss: Callable | None = None

    def _env_step(
        self, state: RecurrentTDLambdaState, key: Key
    ) -> tuple[RecurrentTDLambdaState, Transition]:
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape, dtype=canonicalize_dtype(action_space.dtype)
        )

        ((next_carry, value), intermediates), value_vjp = jax.vjp(
            lambda params: self.value_network.apply(
                params, state.carry, *state.timestep, mutable=["intermediates"]
            ),
            state.value_params,
        )
        (value_grads,) = value_vjp((
            (
                jax.tree.map(jnp.zeros_like, next_carry),
                jnp.ones_like(value),
            ),
            jax.tree.map(jnp.zeros_like, intermediates),
        ))

        next_obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, action, self.env_params
        )
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            aux={
                "value": value,
                "value_grads": value_grads,
                "intermediates": intermediates,
                "value_vjp": value_vjp,
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

    def _update_step(
        self, state: RecurrentTDLambdaState, transition: Transition
    ) -> RecurrentTDLambdaState:
        value = transition.aux["value"]
        value_grads = transition.aux["value_grads"]
        intermediates = transition.aux["intermediates"]
        value_vjp = transition.aux["value_vjp"]
        next_carry = transition.aux["next_carry"]

        value_trace = jax.tree.map(
            lambda trace, grad: self.cfg.gamma * self.cfg.trace_lambda * trace + grad,
            state.value_trace,
            value_grads,
        )

        next_value, curvature = self.value_optimizer.bootstrap(
            state.value_optimizer_state,
            state.value_params,
            value_grads,
            value_trace,
            lambda params: remove_feature_axis(
                self.value_network.apply(params, next_carry, *transition.second)[1]
            ),
            self.cfg.gamma,
            1.0 - transition.second.done.astype(jnp.float32),
        )
        td_error = (
            transition.second.reward
            + self.cfg.gamma * (1.0 - transition.second.done) * next_value
            - remove_feature_axis(value)
        )
        value_updates, value_optimizer_state = self.value_optimizer.update(
            state.value_optimizer_state,
            value_grads,
            value_trace,
            td_error,
            curvature,
        )

        value_params = jax.tree.map(
            lambda p, u: p + u, state.value_params, value_updates
        )

        if self.aux_loss is not None:
            _, next_intermediates = self.value_network.apply(
                state.value_params, next_carry, *transition.second, mutable=["intermediates"]
            )
            transition = transition.replace(
                aux={**transition.aux, "next_intermediates": next_intermediates}
            )
            cotangents = jax.grad(
                lambda i: self.aux_loss(
                    transition.replace(aux={**transition.aux, "intermediates": i})
                )
            )(intermediates)
            (aux_grads,) = value_vjp((
                (jax.tree.map(jnp.zeros_like, next_carry), jnp.zeros_like(value)),
                cotangents,
            ))
            value_params = jax.tree.map(lambda p, g: p - g, value_params, aux_grads)

        value_trace = jax.tree.map(
            lambda t: jnp.where(transition.second.done, jnp.zeros_like(t), t),
            value_trace,
        )

        lox.log(
            {
                "value/value": next_value.mean(),
                "value/td_error": td_error.mean(),
                "value/absolute_td_error": jnp.abs(td_error).mean(),
                "value/cumulant": transition.second.reward.mean(),
            }
        )

        return state.replace(
            value_params=value_params,
            value_trace=value_trace,
            value_optimizer_state=value_optimizer_state,
        )

    def init(self, key: Key) -> RecurrentTDLambdaState:
        env_key, value_key, carry_key = jax.random.split(key, 3)
        obs, env_state = self.env.reset(env_key, self.env_params)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape, dtype=canonicalize_dtype(action_space.dtype)
        )
        timestep = Timestep(
            obs=obs, action=action, reward=jnp.float32(0.0), done=jnp.bool_(True)
        )

        carry = self.value_network.initialize_carry(carry_key)
        value_params = self.value_network.init(value_key, carry, *timestep)

        value_optimizer_state = self.value_optimizer.init(value_params)

        value_trace = jax.tree.map(jnp.zeros_like, value_params)

        return RecurrentTDLambdaState(
            step=0,
            timestep=timestep,
            carry=carry,
            env_state=env_state,
            value_params=value_params,
            value_trace=value_trace,
            value_optimizer_state=value_optimizer_state,
        )

    def train(
        self, key: Key, state: RecurrentTDLambdaState, num_steps: int
    ) -> RecurrentTDLambdaState:
        def step(state, key):
            state, transition = self._env_step(state, key)
            return self._update_step(state, transition), None

        state, _ = jax.lax.scan(
            step,
            state,
            jax.random.split(key, num_steps),
            unroll=self.cfg.unroll,
        )
        return state

    def evaluate(
        self, key: Key, state: RecurrentTDLambdaState, num_steps: int
    ) -> RecurrentTDLambdaState:
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
            carry=self.value_network.initialize_carry(carry_key),
            env_state=env_state,
        )

        def step(state, key):
            state, _ = self._env_step(state, key)
            return state, None

        state, _ = jax.lax.scan(step, state, jax.random.split(eval_key, num_steps))
        return state
