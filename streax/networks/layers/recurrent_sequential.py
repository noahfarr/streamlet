import flax.linen as nn
from streax.utils.typing import Key, PyTree


class RecurrentSequential(nn.Sequential):
    """An ``nn.Sequential`` that also exposes ``initialize_carry``.

    Lets a recurrent network be written as a plain layer list -- a multi-input
    first layer, an ``nn.RNNCellBase`` cell (threaded through via tuple returns),
    and a head -- while still satisfying the recurrent algorithms' contract,
    which calls ``network.initialize_carry(rng, num_envs)``. The carry is taken
    from the first ``nn.RNNCellBase`` among the layers.
    """

    @nn.nowrap
    def initialize_carry(self, rng: Key, num_envs: int) -> PyTree:
        for layer in self.layers:
            if isinstance(layer, nn.RNNCellBase):
                return layer.initialize_carry(rng, (num_envs, 1))
        raise ValueError(
            "RecurrentSequential.initialize_carry: no nn.RNNCellBase among layers."
        )
