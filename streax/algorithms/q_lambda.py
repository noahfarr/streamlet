from dataclasses import dataclass
from functools import partial
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
class QLambdaConfig:
    gamma: float
    trace_lambda: float
    unroll: int = struct.field(pytree_node=False, default=2)


@struct.dataclass(frozen=True)
class QLambdaState:
    step: int
    update_step: int
    timestep: Timestep
    env_state: EnvState
    q_params: core.FrozenDict[str, Any]
    q_trace: PyTree
    q_optimizer_state: PyTree


@dataclass
class QLambda:
    cfg: QLambdaConfig
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    epsilon_schedule: Callable
    q_optimizer: Optimizer

    def _greedy_action(
        self, key: Key, state: QLambdaState
    ) -> tuple[QLambdaState, Array, dict]:
        q_values, q_vjp = jax.vjp(
            lambda params: self.q_network.apply(params, state.timestep.obs),
            state.q_params,
        )
        action = jnp.argmax(q_values, axis=-1)
        q_value = q_values[action]
        (q_grads,) = q_vjp(
            jax.nn.one_hot(action, q_values.shape[-1], dtype=q_values.dtype)
        )
        aux = {
            "non_greedy": jnp.bool_(False),
            "q_value": q_value,
            "q_grads": q_grads,
        }
        return state, action, aux

    def _random_action(
        self, key: Key, state: QLambdaState
    ) -> tuple[QLambdaState, Array, dict]:
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
        self, key: Key, state: QLambdaState
    ) -> tuple[QLambdaState, Array, dict]:
        random_key, sample_key = jax.random.split(key)
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
        (q_grads,) = q_vjp(
            jax.nn.one_hot(action, q_values.shape[-1], dtype=q_values.dtype)
        )
        aux = {"non_greedy": non_greedy, "q_value": q_value, "q_grads": q_grads}
        return state, action, aux

    def _step(
        self, state: QLambdaState, key: Key, *, policy: Callable
    ) -> tuple[QLambdaState, Transition]:
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
        self, state: QLambdaState, key: Key, *, policy: Callable
    ) -> tuple[QLambdaState, None]:
        state, transition = self._step(state, key, policy=policy)
        state = self._update(state, transition)
        return state.replace(update_step=state.update_step + 1), None

    def _update(
        self,
        state: QLambdaState,
        transition: Transition,
    ) -> QLambdaState:
        aux = transition.aux
        non_greedy = aux["non_greedy"]
        q_value = aux["q_value"]
        q_grads = aux["q_grads"]

        reset = transition.second.done | non_greedy
        discount = jnp.float32(self.cfg.gamma * self.cfg.trace_lambda)

        q_trace = jax.tree.map(
            lambda t, g: discount * t + g, state.q_trace, q_grads
        )

        def bootstrap_value(params):
            return self.q_network.apply(params, transition.second.obs).max(axis=-1)

        not_done = 1.0 - transition.second.done.astype(jnp.float32)
        next_value, curvature = self.q_optimizer.bootstrap(
            state.q_optimizer_state,
            state.q_params,
            q_grads,
            q_trace,
            bootstrap_value,
            self.cfg.gamma,
            not_done,
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

        q_params = jax.tree.map(lambda p, u: p + u, state.q_params, q_updates)

        new_q_trace = jax.tree.map(
            lambda t: jnp.where(reset, jnp.zeros_like(t), t), q_trace
        )

        log_dict = {
            "q_network/q_value": q_value.mean(),
            "q_network/td_error": td_error.mean(),
            "training/epsilon": self.epsilon_schedule(state.step),
        }
        lox.log(log_dict)

        new_state = dict(
            q_params=q_params,
            q_trace=new_q_trace,
            q_optimizer_state=q_optimizer_state,
        )

        return state.replace(**new_state)

    def init(self, key: Key) -> QLambdaState:
        env_key, q_key = jax.random.split(key)
        obs, env_state = self.env.reset(env_key, self.env_params)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            action_space.shape,
            dtype=canonicalize_dtype(action_space.dtype),
        )
        timestep = Timestep(obs=obs, action=action, reward=0.0, done=True)
        q_params = self.q_network.init(q_key, obs)

        q_optimizer_state = self.q_optimizer.init(q_params)

        q_trace = jax.tree.map(jnp.zeros_like, q_params)

        state = dict(
            step=0,
            update_step=0,
            timestep=timestep,
            env_state=env_state,
            q_params=q_params,
            q_trace=q_trace,
            q_optimizer_state=q_optimizer_state,
        )

        return QLambdaState(**state)

    def warmup(self, key: Key, state: QLambdaState, num_steps: int) -> QLambdaState:
        step_keys = jax.random.split(key, num_steps)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._random_action),
            state,
            step_keys,
            unroll=self.cfg.unroll,
        )
        return state

    def train(self, key: Key, state: QLambdaState, num_steps: int) -> QLambdaState:
        keys = jax.random.split(key, num_steps)
        state, _ = jax.lax.scan(
            partial(self._update_step, policy=self._epsilon_greedy_action),
            state,
            keys,
            unroll=self.cfg.unroll,
        )
        return state

    def evaluate(self, key: Key, state: QLambdaState, num_steps: int) -> QLambdaState:
        reset_key, eval_key = jax.random.split(key)
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
