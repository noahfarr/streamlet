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
class RecurrentQLambdaConfig:
    gamma: float
    trace_lambda: float
    unroll: int = struct.field(pytree_node=False, default=4)


@struct.dataclass(frozen=True)
class RecurrentQLambdaState:
    step: int
    update_step: int
    timestep: Timestep
    carry: PyTree
    env_state: EnvState
    q_params: core.FrozenDict[str, Any]
    q_trace: PyTree
    q_optimizer_state: PyTree


@dataclass
class RecurrentQLambda:
    cfg: RecurrentQLambdaConfig
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    epsilon_schedule: Callable
    q_optimizer: Optimizer

    def _apply(
        self, params: PyTree, carry: PyTree, timestep: Timestep
    ) -> tuple[PyTree, Array, dict]:
        return self.q_network.apply(
            params,
            carry,
            timestep.obs,
            timestep.action,
            timestep.reward,
            timestep.done,
        )

    def _reset_carry(self, carry: PyTree, done: Array) -> PyTree:
        return jax.tree.map(
            lambda leaf: jnp.where(done, jnp.zeros_like(leaf), leaf), carry
        )

    def _value_and_grad(
        self, params: PyTree, carry: PyTree, timestep: Timestep, action: Array
    ) -> tuple[Array, Array, PyTree]:
        (carry_next, q_values, aux), q_vjp = jax.vjp(
            lambda p: self._apply(p, carry, timestep), params
        )
        q_value = q_values[action]
        num_actions = self.env.action_space(self.env_params).n
        (q_grads,) = q_vjp((
            jax.tree.map(jnp.zeros_like, carry_next),
            jax.nn.one_hot(action, num_actions, dtype=q_values.dtype),
            jax.tree.map(jnp.zeros_like, aux),
        ))
        return carry_next, q_value, q_grads

    def _greedy_action(
        self, key: Key, state: RecurrentQLambdaState
    ) -> tuple[RecurrentQLambdaState, Array, dict]:
        carry_next, q_values, _ = self._apply(state.q_params, state.carry, state.timestep)
        action = jnp.argmax(q_values, axis=-1)
        _, q_value, q_grads = self._value_and_grad(
            state.q_params, state.carry, state.timestep, action
        )
        aux = {
            "non_greedy": jnp.bool_(False),
            "q_value": q_value,
            "q_grads": q_grads,
            "carry_next": carry_next,
        }
        return state, action, aux

    def _random_action(
        self, key: Key, state: RecurrentQLambdaState
    ) -> tuple[RecurrentQLambdaState, Array, dict]:
        carry_next, _, _ = self._apply(state.q_params, state.carry, state.timestep)
        action_space = self.env.action_space(self.env_params)
        action = jax.random.randint(
            key,
            action_space.shape,
            minval=0,
            maxval=action_space.n,
        )
        aux = {"non_greedy": jnp.bool_(True), "carry_next": carry_next}
        return state, action, aux

    def _epsilon_greedy_action(
        self, key: Key, state: RecurrentQLambdaState
    ) -> tuple[RecurrentQLambdaState, Array, dict]:
        random_key, sample_key = jax.random.split(key)
        action_space = self.env.action_space(self.env_params)
        random_action = jax.random.randint(
            random_key,
            action_space.shape,
            minval=0,
            maxval=action_space.n,
        )

        carry_next, q_values, _ = self._apply(state.q_params, state.carry, state.timestep)
        greedy_action = jnp.argmax(q_values, axis=-1)

        epsilon = self.epsilon_schedule(state.step)
        is_random = jax.random.uniform(sample_key, ()) < epsilon
        action = jnp.where(is_random, random_action, greedy_action)
        non_greedy = is_random & (random_action != greedy_action)

        _, q_value, q_grads = self._value_and_grad(
            state.q_params, state.carry, state.timestep, action
        )
        aux = {
            "non_greedy": non_greedy,
            "q_value": q_value,
            "q_grads": q_grads,
            "carry_next": carry_next,
        }
        return state, action, aux

    def _step(
        self, state: RecurrentQLambdaState, key: Key, *, policy: Callable
    ) -> tuple[RecurrentQLambdaState, Transition]:
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
                carry=self._reset_carry(aux["carry_next"], done),
                env_state=env_state,
            ),
            transition,
        )

    def _update_step(
        self, state: RecurrentQLambdaState, key: Key, *, policy: Callable
    ) -> tuple[RecurrentQLambdaState, None]:
        state, transition = self._step(state, key, policy=policy)
        state = self._update(state, transition)
        return state.replace(update_step=state.update_step + 1), None

    def _update(
        self,
        state: RecurrentQLambdaState,
        transition: Transition,
    ) -> RecurrentQLambdaState:
        aux = transition.aux
        non_greedy = aux["non_greedy"]
        q_value = aux["q_value"]
        q_grads = aux["q_grads"]
        carry_next = aux["carry_next"]

        reset = transition.second.done | non_greedy
        discount = jnp.float32(self.cfg.gamma * self.cfg.trace_lambda)

        q_trace = jax.tree.map(
            lambda t, g: discount * t + g, state.q_trace, q_grads
        )

        def bootstrap_value(params):
            _, next_q_values, _ = self._apply(params, carry_next, transition.second)
            return next_q_values.max(axis=-1)

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
            "q_network/absolute_td_error": jnp.abs(td_error).mean(),
            "training/epsilon": self.epsilon_schedule(state.step),
        }
        lox.log(log_dict)

        return state.replace(
            q_params=q_params,
            q_trace=new_q_trace,
            q_optimizer_state=q_optimizer_state,
        )

    def init(self, key: Key) -> RecurrentQLambdaState:
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
        q_params = self.q_network.init(
            q_key,
            carry,
            timestep.obs,
            timestep.action,
            timestep.reward,
            timestep.done,
        )

        q_optimizer_state = self.q_optimizer.init(q_params)

        q_trace = jax.tree.map(jnp.zeros_like, q_params)

        return RecurrentQLambdaState(
            step=0,
            update_step=0,
            timestep=timestep,
            carry=carry,
            env_state=env_state,
            q_params=q_params,
            q_trace=q_trace,
            q_optimizer_state=q_optimizer_state,
        )

    def warmup(
        self, key: Key, state: RecurrentQLambdaState, num_steps: int
    ) -> RecurrentQLambdaState:
        keys = jax.random.split(key, num_steps)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._random_action),
            state,
            keys,
            unroll=self.cfg.unroll,
        )
        return state

    def train(
        self, key: Key, state: RecurrentQLambdaState, num_steps: int
    ) -> RecurrentQLambdaState:
        keys = jax.random.split(key, num_steps)
        state, _ = jax.lax.scan(
            partial(self._update_step, policy=self._epsilon_greedy_action),
            state,
            keys,
            unroll=self.cfg.unroll,
        )
        return state

    def evaluate(
        self, key: Key, state: RecurrentQLambdaState, num_steps: int
    ) -> RecurrentQLambdaState:
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

        state, _ = jax.lax.scan(
            partial(self._step, policy=self._greedy_action),
            state,
            jax.random.split(eval_key, num_steps),
        )
        return state
