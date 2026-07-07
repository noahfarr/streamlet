import jax.numpy as jnp
from flax import struct

from streamlet.utils.typing import Array


@struct.dataclass(frozen=True)
class RunningStats:
    n: Array
    m: Array
    s: Array

    @classmethod
    def init(cls, shape: tuple = ()) -> "RunningStats":
        return cls(
            n=jnp.zeros(shape, dtype=jnp.int32),
            m=jnp.zeros(shape, dtype=jnp.float32),
            s=jnp.zeros(shape, dtype=jnp.float32),
        )

    def push(self, x: Array, mask: Array) -> "RunningStats":
        x = jnp.asarray(x, dtype=jnp.float32)
        mask = jnp.asarray(mask, dtype=jnp.bool_)
        new_n = self.n + mask.astype(self.n.dtype)
        denom = jnp.maximum(new_n, 1).astype(jnp.float32)
        delta = jnp.where(mask, x - self.m, 0.0)
        new_m = self.m + delta / denom
        new_s = self.s + jnp.where(mask, (x - self.m) * (x - new_m), 0.0)
        return RunningStats(n=new_n, m=new_m, s=new_s)

    @property
    def variance(self) -> Array:
        denom = jnp.maximum(self.n, 1).astype(jnp.float32)
        return jnp.where(self.n > 0, self.s / denom, 0.0)


@struct.dataclass(frozen=True)
class TDErrorScalerState:
    reward_rms: RunningStats
    gamma_rms: RunningStats
    return_rms: RunningStats
    return_sq_rms: RunningStats
    G: Array

    @classmethod
    def init(cls) -> "TDErrorScalerState":
        shape = ()
        return cls(
            reward_rms=RunningStats.init(shape),
            gamma_rms=RunningStats.init(shape),
            return_rms=RunningStats.init(shape),
            return_sq_rms=RunningStats.init(shape),
            G=jnp.zeros(shape, dtype=jnp.float32),
        )

    def update(self, r_ent: Array, done: Array, gamma: float) -> "TDErrorScalerState":
        done = jnp.asarray(done, dtype=jnp.bool_)
        r_ent = jnp.asarray(r_ent, dtype=jnp.float32)
        new_G = self.G + r_ent
        gamma_step = jnp.where(done, 0.0, gamma)
        observed = jnp.ones_like(done)
        return TDErrorScalerState(
            reward_rms=self.reward_rms.push(r_ent, observed),
            gamma_rms=self.gamma_rms.push(gamma_step, observed),
            return_rms=self.return_rms.push(new_G, done),
            return_sq_rms=self.return_sq_rms.push(jnp.square(new_G), done),
            G=jnp.where(done, 0.0, new_G),
        )

    def sigma(self) -> Array:
        variance = (
            self.reward_rms.variance
            + self.gamma_rms.variance * self.return_sq_rms.m
        )
        variance = jnp.maximum(variance, 1e-4)
        return jnp.where(self.return_sq_rms.n > 0, jnp.sqrt(variance), 1.0)
