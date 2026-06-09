from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from streax.optimizers import Implicit, Optimizer
from streax.utils import Timestep, Transition, canonicalize_dtype
from streax.utils.typing import (
    Array,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)


@struct.dataclass(frozen=True)
class QRCLambdaConfig:
    gamma: float
    trace_lambda: float
    gradient_correction: bool
    regularization_coefficient: float
    unroll: int = struct.field(pytree_node=False, default=4)


@struct.dataclass(frozen=True)
class QRCLambdaState:
    step: int
    update_step: int
    timestep: Timestep
    env_state: EnvState
    q_params: core.FrozenDict[str, Any]
    h_params: core.FrozenDict[str, Any]
    q_optimizer_state: PyTree
    h_optimizer_state: PyTree
    q_trace: PyTree
    h_trace: PyTree
    bias_trace: Array
    aux_q_optimizer_state: PyTree
    aux_h_optimizer_state: PyTree


@dataclass
class QRCLambda:
    cfg: QRCLambdaConfig
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    h_network: nn.Module
    q_optimizer: Optimizer
    h_optimizer: Optimizer
    epsilon_schedule: Callable
    aux_q_loss: Callable = lambda params, transition: 0.0
    aux_h_loss: Callable = lambda params, transition: 0.0
    aux_q_optimizer: optax.GradientTransformation = optax.sgd(1e-3)
    aux_h_optimizer: optax.GradientTransformation = optax.sgd(1e-3)

    def _greedy_action(
        self, key: Key, state: QRCLambdaState
    ) -> tuple[QRCLambdaState, Array, dict]:
        q_values, q_vjp = jax.vjp(
            lambda params: self.q_network.apply(params, state.timestep.obs),
            state.q_params,
        )
        action = jnp.argmax(q_values, axis=-1)
        q_value = q_values[action]
        num_actions = self.env.action_space(self.env_params).n
        (q_grads,) = q_vjp(
            jax.nn.one_hot(action, num_actions, dtype=q_values.dtype)
        )
        aux = {
            "non_greedy": jnp.bool_(False),
            "q_value": q_value,
            "q_grads": q_grads,
        }
        return state, action, aux

    def _random_action(
        self, key: Key, state: QRCLambdaState
    ) -> tuple[QRCLambdaState, Array, dict]:
        action_space = self.env.action_space(self.env_params)
        action = jax.random.randint(
            key,
            action_space.shape,
            minval=0,
            maxval=action_space.n,
        )
        aux = {"non_greedy": jnp.bool_(True)}
        return state, action, aux

    def _epsilon_greedy_action(
        self, key: Key, state: QRCLambdaState
    ) -> tuple[QRCLambdaState, Array, dict]:
        random_key, _, sample_key = jax.random.split(key, 3)
        state, random_action, _ = self._random_action(random_key, state)

        q_values, q_vjp = jax.vjp(
            lambda params: self.q_network.apply(params, state.timestep.obs),
            state.q_params,
        )
        greedy_action = jnp.argmax(q_values, axis=-1)

        epsilon = self.epsilon_schedule(state.step)
        is_random = jax.random.uniform(sample_key, ()) < epsilon
        action = jnp.where(is_random, random_action, greedy_action)
        non_greedy = is_random & (random_action != greedy_action)
        q_value = q_values[action]
        num_actions = self.env.action_space(self.env_params).n
        (q_grads,) = q_vjp(
            jax.nn.one_hot(action, num_actions, dtype=q_values.dtype)
        )
        aux = {
            "non_greedy": non_greedy,
            "q_value": q_value,
            "q_grads": q_grads,
        }
        return state, action, aux

    def _step(
        self, state: QRCLambdaState, key: Key, *, policy: Callable
    ) -> tuple[QRCLambdaState, Transition]:
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
        self, state: QRCLambdaState, key: Key, *, policy: Callable
    ) -> tuple[QRCLambdaState, None]:
        state, transition = self._step(state, key, policy=policy)
        state = self._update(state, transition)
        return state.replace(update_step=state.update_step + 1), None

    def _update(self, state: QRCLambdaState, transition: Transition) -> QRCLambdaState:
        action = transition.second.action
        aux = transition.aux
        non_greedy = aux["non_greedy"]
        q_value = aux["q_value"]
        q_grads = aux["q_grads"]

        def get_next_q_value(params):
            q_values = self.q_network.apply(params, transition.second.obs)
            return q_values.max(axis=-1)

        next_q_value, nv_vjp = jax.vjp(get_next_q_value, state.q_params)
        (next_q_value_grads,) = nv_vjp(jnp.ones_like(next_q_value))

        td_error = (
            transition.second.reward
            + self.cfg.gamma * next_q_value * (1.0 - transition.second.done)
            - q_value
        )
        coef = self.cfg.gamma * (1.0 - transition.second.done.astype(jnp.float32))
        td_grads = jax.tree.map(
            lambda nvg, qg: coef * nvg - qg,
            next_q_value_grads,
            q_grads,
        )

        def compute_h(params):
            h_values = self.h_network.apply(params, transition.first.obs)
            h_value = h_values[action]
            return h_value

        h_values, h_vjp = jax.vjp(compute_h, state.h_params)
        (h_grads,) = h_vjp(jnp.ones_like(h_values))

        reset_trace = transition.second.done | non_greedy
        discount = jnp.float32(self.cfg.gamma * self.cfg.trace_lambda)

        q_trace = jax.tree.map(
            lambda t, g: (discount * t + g).astype(t.dtype), state.q_trace, q_grads
        )
        h_trace = jax.tree.map(
            lambda t, g: (discount * t + g).astype(t.dtype), state.h_trace, h_grads
        )
        bias_trace = discount * state.bias_trace + h_values

        h_updates = jax.tree.map(
            lambda t, g, p: td_error * t
            - h_values * g
            - self.cfg.regularization_coefficient * p,
            h_trace,
            h_grads,
            state.h_params,
        )
        h_grads_final = jax.tree.map(lambda x: -x, h_updates)
        h_param_updates, h_optimizer_state = self.h_optimizer.update(
            state.h_optimizer_state, h_grads_final
        )
        h_params = jax.tree.map(lambda p, u: p + u, state.h_params, h_param_updates)

        _, aux_h_grads = jax.value_and_grad(self.aux_h_loss)(h_params, transition)
        aux_h_updates, aux_h_optimizer_state = self.aux_h_optimizer.update(
            aux_h_grads, state.aux_h_optimizer_state, h_params
        )
        h_params = optax.apply_updates(h_params, aux_h_updates)

        if isinstance(self.q_optimizer, Implicit):
            assert self.cfg.gradient_correction, (
                "QRCLambda with the Implicit q-optimizer requires gradient_correction=True; "
                "the implicit step is derived for the full direction delta z - h g - e_h b."
            )
            rho_trace = self.q_optimizer.precondition(
                state.q_optimizer_state, q_trace
            )
            curvature = -sum(
                jnp.sum(b * z)
                for b, z in zip(
                    jax.tree.leaves(td_grads), jax.tree.leaves(rho_trace)
                )
            )
            q_param_updates, q_optimizer_state = self.q_optimizer.update(
                state.q_optimizer_state,
                q_grads,
                q_trace,
                td_error,
                curvature,
                td_error_grad=td_grads,
                h_value=h_values,
                bias_trace=bias_trace,
            )
        else:
            q_updates = jax.tree.map(
                lambda td_g: -bias_trace * td_g, td_grads
            )

            if self.cfg.gradient_correction:
                q_updates = jax.tree.map(
                    lambda update, t, g: update
                    + td_error * t
                    - h_values * g,
                    q_updates,
                    q_trace,
                    q_grads,
                )

            q_grads_final = jax.tree.map(lambda x: -x, q_updates)
            q_param_updates, q_optimizer_state = self.q_optimizer.update(
                state.q_optimizer_state, q_grads_final
            )

        q_params = jax.tree.map(lambda p, u: p + u, state.q_params, q_param_updates)

        _, aux_q_grads = jax.value_and_grad(self.aux_q_loss)(q_params, transition)
        aux_q_updates, aux_q_optimizer_state = self.aux_q_optimizer.update(
            aux_q_grads, state.aux_q_optimizer_state, q_params
        )
        q_params = optax.apply_updates(q_params, aux_q_updates)

        new_q_trace = jax.tree.map(
            lambda t: jnp.where(reset_trace, jnp.zeros_like(t), t),
            q_trace,
        )
        new_h_trace = jax.tree.map(
            lambda t: jnp.where(reset_trace, jnp.zeros_like(t), t),
            h_trace,
        )
        new_bias_trace = jnp.where(reset_trace, jnp.zeros_like(bias_trace), bias_trace)

        target_q_value = q_value + td_error
        explained_variance = 1 - jnp.var(td_error) / (jnp.var(target_q_value) + 1e-8)
        lox.log(
            {
                "q_network/q_value": q_value.mean(),
                "q_network/td_error": td_error.mean(),
                "q_network/absolute_td_error": jnp.abs(td_error).mean(),
                "q_network/explained_variance": explained_variance,
                "h_network/h_value": h_values.mean(),
                "h_network/bias_trace": bias_trace.mean(),
                "training/epsilon": self.epsilon_schedule(state.step),
            }
        )

        return state.replace(
            q_params=q_params,
            h_params=h_params,
            q_optimizer_state=q_optimizer_state,
            h_optimizer_state=h_optimizer_state,
            q_trace=new_q_trace,
            h_trace=new_h_trace,
            bias_trace=new_bias_trace,
            aux_q_optimizer_state=aux_q_optimizer_state,
            aux_h_optimizer_state=aux_h_optimizer_state,
        )

    def init(self, key: Key) -> QRCLambdaState:
        env_key, q_key, h_key = jax.random.split(key, 3)
        obs, env_state = self.env.reset(env_key, self.env_params)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape, dtype=canonicalize_dtype(action_space.dtype)
        )
        timestep = Timestep(obs=obs, action=action, reward=0.0, done=True)

        q_params = self.q_network.init(q_key, obs)
        h_params = self.h_network.init(h_key, obs)
        q_optimizer_state = self.q_optimizer.init(q_params)
        h_optimizer_state = self.h_optimizer.init(h_params)
        aux_q_optimizer_state = self.aux_q_optimizer.init(q_params)
        aux_h_optimizer_state = self.aux_h_optimizer.init(h_params)

        q_trace = jax.tree.map(jnp.zeros_like, q_params)
        h_trace = jax.tree.map(jnp.zeros_like, h_params)
        bias_trace = jnp.float32(0.0)

        return QRCLambdaState(
            step=0,
            update_step=0,
            timestep=timestep,
            env_state=env_state,
            q_params=q_params,
            h_params=h_params,
            q_optimizer_state=q_optimizer_state,
            h_optimizer_state=h_optimizer_state,
            q_trace=q_trace,
            h_trace=h_trace,
            bias_trace=bias_trace,
            aux_q_optimizer_state=aux_q_optimizer_state,
            aux_h_optimizer_state=aux_h_optimizer_state,
        )

    def warmup(self, key: Key, state: QRCLambdaState, num_steps: int) -> QRCLambdaState:
        step_keys = jax.random.split(key, num_steps)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._random_action),
            state,
            step_keys,
            unroll=self.cfg.unroll,
        )
        return state

    def train(self, key: Key, state: QRCLambdaState, num_steps: int) -> QRCLambdaState:
        keys = jax.random.split(key, num_steps)
        state, _ = jax.lax.scan(
            partial(self._update_step, policy=self._epsilon_greedy_action),
            state,
            keys,
            unroll=self.cfg.unroll,
        )
        return state

    def evaluate(
        self, key: Key, state: QRCLambdaState, num_steps: int
    ) -> QRCLambdaState:
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
            partial(self._step, policy=self._greedy_action),
            state,
            jax.random.split(eval_key, num_steps),
        )
        return state
