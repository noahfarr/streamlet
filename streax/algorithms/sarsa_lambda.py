from dataclasses import dataclass
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from streax.optimizers import Optimizer
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
class SARSALambdaConfig:
    gamma: float
    trace_lambda: float
    unroll: int = struct.field(pytree_node=False, default=4)


@struct.dataclass(frozen=True)
class SARSALambdaState:
    step: int
    update_step: int
    timestep: Timestep
    next_action: Array
    env_state: EnvState
    q_params: core.FrozenDict[str, Any]
    q_trace: PyTree
    q_optimizer_state: PyTree
    aux_optimizer_state: PyTree


@dataclass
class SARSALambda:
    cfg: SARSALambdaConfig
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    epsilon_schedule: Callable
    q_optimizer: Optimizer
    aux_loss: Callable = lambda params, transition: 0.0
    aux_optimizer: optax.GradientTransformation = optax.sgd(1e-3)

    def _sample_action(
        self, key: Key, q_params: PyTree, obs: Array, step: Array
    ) -> tuple[Array, Array]:
        random_key, sample_key = jax.random.split(key)
        action_space = self.env.action_space(self.env_params)
        random_action = jax.random.randint(
            random_key,
            action_space.shape,
            minval=0,
            maxval=action_space.n,
        )
        q_values = self.q_network.apply(q_params, obs)
        greedy_action = jnp.argmax(q_values, axis=-1)

        epsilon = self.epsilon_schedule(step)
        is_random = jax.random.uniform(sample_key, ()) < epsilon
        action = jnp.where(is_random, random_action, greedy_action)
        value = q_values[action]
        return action, value

    def _step(self, state: SARSALambdaState, key: Key) -> tuple[SARSALambdaState, Transition]:
        sample_key, step_key = jax.random.split(key)

        action = state.next_action

        next_obs, env_state, reward, done, info = self.env.step(
            step_key, state.env_state, action, self.env_params
        )
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        next_action, _ = self._sample_action(
            sample_key, state.q_params, next_obs, state.step + 1
        )

        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=next_obs, action=action, reward=reward, done=done),
            aux={"next_action": next_action},
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
                next_action=next_action,
                env_state=env_state,
            ),
            transition,
        )

    def _update_step(
        self, state: SARSALambdaState, key: Key
    ) -> tuple[SARSALambdaState, None]:
        state, transition = self._step(state, key)
        state = self._update(state, transition)
        return state.replace(update_step=state.update_step + 1), None

    def _update(
        self, state: SARSALambdaState, transition: Transition
    ) -> SARSALambdaState:
        action = transition.second.action
        next_action = transition.aux["next_action"]

        q_values, q_vjp = jax.vjp(
            lambda params: self.q_network.apply(params, transition.first.obs),
            state.q_params,
        )
        q_value = q_values[action]
        num_actions = self.env.action_space(self.env_params).n
        (q_grads,) = q_vjp(jax.nn.one_hot(action, num_actions, dtype=q_values.dtype))

        reset = transition.second.done
        discount = jnp.float32(self.cfg.gamma * self.cfg.trace_lambda)

        q_trace = jax.tree.map(
            lambda t, g: (discount * t + g).astype(t.dtype), state.q_trace, q_grads
        )

        def get_next_q_value(params):
            q_values = self.q_network.apply(params, transition.second.obs)
            return q_values[next_action]

        not_done = 1.0 - transition.second.done.astype(jnp.float32)
        next_q_value, curvature = self.q_optimizer.bootstrap(
            state.q_optimizer_state,
            state.q_params,
            q_grads,
            q_trace,
            get_next_q_value,
            self.cfg.gamma,
            not_done,
        )
        td_error = (
            transition.second.reward
            + self.cfg.gamma * next_q_value * (1.0 - transition.second.done)
            - q_value
        )
        q_updates, q_optimizer_state = self.q_optimizer.update(
            state.q_optimizer_state, q_grads, q_trace, td_error, curvature,
        )

        q_params = jax.tree.map(lambda p, u: p + u, state.q_params, q_updates)

        _, aux_grads = jax.value_and_grad(self.aux_loss)(q_params, transition)
        aux_updates, aux_optimizer_state = self.aux_optimizer.update(
            aux_grads, state.aux_optimizer_state, q_params
        )
        q_params = optax.apply_updates(q_params, aux_updates)

        new_q_trace = jax.tree.map(
            lambda t: jnp.where(reset, jnp.zeros_like(t), t), q_trace
        )

        log_dict = {
            "q_network/q_value": q_value.mean(),
            "q_network/td_error": td_error.mean(),
            "q_network/absolute_td_error": jnp.abs(td_error).mean(),
            "training/epsilon": self.epsilon_schedule(state.step),
        }
        lox.log(log_dict)

        return state.replace(
            q_params=q_params,
            q_trace=new_q_trace,
            q_optimizer_state=q_optimizer_state,
            aux_optimizer_state=aux_optimizer_state,
        )

    def init(self, key: Key) -> SARSALambdaState:
        env_key, q_key, action_key = jax.random.split(key, 3)
        obs, env_state = self.env.reset(env_key, self.env_params)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape, dtype=canonicalize_dtype(action_space.dtype)
        )
        timestep = Timestep(obs=obs, action=action, reward=0.0, done=True)
        q_params = self.q_network.init(q_key, obs)

        q_optimizer_state = self.q_optimizer.init(q_params)
        aux_optimizer_state = self.aux_optimizer.init(q_params)

        q_trace = jax.tree.map(jnp.zeros_like, q_params)

        next_action, _ = self._sample_action(action_key, q_params, obs, jnp.int32(0))

        return SARSALambdaState(
            step=0,
            update_step=0,
            timestep=timestep,
            next_action=next_action,
            env_state=env_state,
            q_params=q_params,
            q_trace=q_trace,
            q_optimizer_state=q_optimizer_state,
            aux_optimizer_state=aux_optimizer_state,
        )

    def train(
        self, key: Key, state: SARSALambdaState, num_steps: int
    ) -> SARSALambdaState:
        keys = jax.random.split(key, num_steps)
        state, _ = jax.lax.scan(
            self._update_step, state, keys, unroll=self.cfg.unroll
        )
        return state

    def evaluate(
        self, key: Key, state: SARSALambdaState, num_steps: int
    ) -> SARSALambdaState:
        reset_key, eval_key = jax.random.split(key)
        obs, env_state = self.env.reset(reset_key, self.env_params)

        action_space = self.env.action_space(self.env_params)
        first_action = jnp.argmax(self.q_network.apply(state.q_params, obs), axis=-1)
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
            next_action=first_action,
            env_state=env_state,
        )

        def greedy_step(state: SARSALambdaState, key: Key):
            next_obs, env_state, reward, done, info = self.env.step(
                key, state.env_state, state.next_action, self.env_params
            )
            next_action = jnp.argmax(
                self.q_network.apply(state.q_params, next_obs), axis=-1
            )
            return (
                state.replace(
                    timestep=Timestep(
                        obs=next_obs,
                        action=state.next_action,
                        reward=jnp.asarray(reward, dtype=jnp.float32),
                        done=jnp.asarray(done, dtype=jnp.bool_),
                    ),
                    next_action=next_action,
                    env_state=env_state,
                ),
                None,
            )

        state, _ = jax.lax.scan(
            greedy_step, state, jax.random.split(eval_key, num_steps)
        )
        return state
