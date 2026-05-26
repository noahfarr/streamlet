from typing import Any, TypeAlias

import gymnax
import jax

Key: TypeAlias = jax.Array
Array: TypeAlias = jax.Array

Environment: TypeAlias = gymnax.environments.environment.Environment
EnvParams: TypeAlias = gymnax.EnvParams
EnvState: TypeAlias = gymnax.EnvState
Discrete: TypeAlias = gymnax.environments.spaces.Discrete
Box: TypeAlias = gymnax.environments.spaces.Box

PyTree: TypeAlias = Any
