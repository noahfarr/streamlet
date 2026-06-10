from dataclasses import dataclass
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
from flax import core, struct

from streax.optimizers import Implicit, Optimizer
from streax.utils import Timestep, Transition, canonicalize_dtype
from streax.utils.typing import Array, Environment, EnvParams, EnvState, Key, PyTree


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
    timestep: Timestep
    env_state: EnvState
    q_params: core.FrozenDict[str, Any]
    h_params: core.FrozenDict[str, Any]
    q_optimizer_state: PyTree
    h_optimizer_state: PyTree
    q_trace: PyTree
    h_trace: PyTree
    bias_trace: Array


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
    aux_q_loss: Callable | None = None
    aux_h_loss: Callable | None = None

    def _env_step(
        self, state: QRCLambdaState, key: Key, epsilon: Array
    ) -> tuple[QRCLambdaState, Transition]:
        random_key, sample_key, step_key = jax.random.split(key, 3)

        action_space = self.env.action_space(self.env_params)
        random_action = jax.random.randint(
            random_key,
            action_space.shape,
            minval=0,
            maxval=action_space.n,
        )

        (q_values, intermediates), q_vjp = jax.vjp(
            lambda params: self.q_network.apply(
                params, state.timestep.obs, mutable=["intermediates"]
            ),
            state.q_params,
        )
        greedy_action = jnp.argmax(q_values, axis=-1)

        explore = jax.random.uniform(sample_key, ()) < epsilon
        action = jnp.where(explore, random_action, greedy_action)
        non_greedy = explore & (random_action != greedy_action)

        q_value = q_values[action]
        (q_grads,) = q_vjp(
            (
                jax.nn.one_hot(action, action_space.n, dtype=q_values.dtype),
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
                "non_greedy": non_greedy,
                "q_value": q_value,
                "q_values": q_values,
                "q_grads": q_grads,
                "intermediates": intermediates,
                "q_vjp": q_vjp,
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
        self, state: QRCLambdaState, transition: Transition
    ) -> QRCLambdaState:
        action = transition.second.action
        non_greedy = transition.aux["non_greedy"]
        q_value = transition.aux["q_value"]
        q_values = transition.aux["q_values"]
        q_grads = transition.aux["q_grads"]
        intermediates = transition.aux["intermediates"]
        q_vjp = transition.aux["q_vjp"]

        next_q_value, nv_vjp = jax.vjp(
            lambda params: self.q_network.apply(params, transition.second.obs).max(
                axis=-1
            ),
            state.q_params,
        )
        (next_q_value_grads,) = nv_vjp(jnp.ones_like(next_q_value))

        td_error = (
            transition.second.reward
            + self.cfg.gamma * next_q_value * (1.0 - transition.second.done)
            - q_value
        )
        td_grads = jax.tree.map(
            lambda next_grad, grad: self.cfg.gamma
            * (1.0 - transition.second.done.astype(jnp.float32))
            * next_grad
            - grad,
            next_q_value_grads,
            q_grads,
        )

        h_value, h_vjp = jax.vjp(
            lambda params: self.h_network.apply(params, transition.first.obs)[action],
            state.h_params,
        )
        (h_grads,) = h_vjp(jnp.ones_like(h_value))

        q_trace = jax.tree.map(
            lambda trace, grad: self.cfg.gamma * self.cfg.trace_lambda * trace + grad,
            state.q_trace,
            q_grads,
        )
        h_trace = jax.tree.map(
            lambda trace, grad: self.cfg.gamma * self.cfg.trace_lambda * trace + grad,
            state.h_trace,
            h_grads,
        )
        bias_trace = (
            self.cfg.gamma * self.cfg.trace_lambda * state.bias_trace + h_value
        )

        h_updates = jax.tree.map(
            lambda trace, grad, param: td_error * trace
            - h_value * grad
            - self.cfg.regularization_coefficient * param,
            h_trace,
            h_grads,
            state.h_params,
        )
        h_param_updates, h_optimizer_state = self.h_optimizer.update(
            state.h_optimizer_state, jax.tree.map(lambda u: -u, h_updates)
        )
        h_params = jax.tree.map(lambda p, u: p + u, state.h_params, h_param_updates)

        if self.aux_h_loss is not None:
            aux_h_grads = jax.grad(self.aux_h_loss)(h_params, transition)
            h_params = jax.tree.map(
                lambda p, g: p - g, h_params, aux_h_grads
            )

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
                h_value=h_value,
                bias_trace=bias_trace,
            )
        else:
            q_updates = jax.tree.map(
                lambda td_g: -bias_trace * td_g, td_grads
            )

            if self.cfg.gradient_correction:
                q_updates = jax.tree.map(
                    lambda update, trace, grad: update
                    + td_error * trace
                    - h_value * grad,
                    q_updates,
                    q_trace,
                    q_grads,
                )

            q_param_updates, q_optimizer_state = self.q_optimizer.update(
                state.q_optimizer_state, jax.tree.map(lambda u: -u, q_updates)
            )

        q_params = jax.tree.map(lambda p, u: p + u, state.q_params, q_param_updates)

        if self.aux_q_loss is not None:
            _, next_intermediates = self.q_network.apply(
                state.q_params, transition.second.obs, mutable=["intermediates"]
            )
            transition = transition.replace(
                aux={**transition.aux, "next_intermediates": next_intermediates}
            )
            cotangents = jax.grad(
                lambda i: self.aux_q_loss(
                    transition.replace(aux={**transition.aux, "intermediates": i})
                )
            )(intermediates)
            (aux_q_grads,) = q_vjp((jnp.zeros_like(q_values), cotangents))
            q_params = jax.tree.map(
                lambda p, g: p - g, q_params, aux_q_grads
            )

        target_q_value = q_value + td_error
        explained_variance = 1 - jnp.var(td_error) / (jnp.var(target_q_value) + 1e-8)
        lox.log(
            {
                "q_network/q_value": q_value.mean(),
                "q_network/td_error": td_error.mean(),
                "q_network/absolute_td_error": jnp.abs(td_error).mean(),
                "q_network/explained_variance": explained_variance,
                "h_network/h_value": h_value.mean(),
                "h_network/bias_trace": bias_trace.mean(),
            }
        )

        q_trace = jax.tree.map(
            lambda t: jnp.where(
                transition.second.done | non_greedy, jnp.zeros_like(t), t
            ),
            q_trace,
        )
        h_trace = jax.tree.map(
            lambda t: jnp.where(
                transition.second.done | non_greedy, jnp.zeros_like(t), t
            ),
            h_trace,
        )
        bias_trace = jnp.where(
            transition.second.done | non_greedy, jnp.zeros_like(bias_trace), bias_trace
        )

        return state.replace(
            q_params=q_params,
            h_params=h_params,
            q_optimizer_state=q_optimizer_state,
            h_optimizer_state=h_optimizer_state,
            q_trace=q_trace,
            h_trace=h_trace,
            bias_trace=bias_trace,
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

        q_trace = jax.tree.map(jnp.zeros_like, q_params)
        h_trace = jax.tree.map(jnp.zeros_like, h_params)
        bias_trace = jnp.float32(0.0)

        return QRCLambdaState(
            step=0,
            timestep=timestep,
            env_state=env_state,
            q_params=q_params,
            h_params=h_params,
            q_optimizer_state=q_optimizer_state,
            h_optimizer_state=h_optimizer_state,
            q_trace=q_trace,
            h_trace=h_trace,
            bias_trace=bias_trace,
        )

    def train(self, key: Key, state: QRCLambdaState, num_steps: int) -> QRCLambdaState:
        def step(state, key):
            epsilon = self.epsilon_schedule(state.step)
            lox.log({"training/epsilon": epsilon})
            state, transition = self._env_step(state, key, epsilon)
            return self._update_step(state, transition), None

        state, _ = jax.lax.scan(
            step,
            state,
            jax.random.split(key, num_steps),
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

        def step(state, key):
            state, _ = self._env_step(state, key, 0.0)
            return state, None

        state, _ = jax.lax.scan(step, state, jax.random.split(eval_key, num_steps))
        return state
