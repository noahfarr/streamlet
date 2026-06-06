import flax.linen as nn

from streax.utils.typing import Key, PyTree


class RecurrentSequential(nn.Sequential):
    """An ``nn.Sequential`` that also exposes ``initialize_carry``.

    Lets a recurrent network be written as a plain layer list -- a multi-input
    first layer, an ``nn.RNNCellBase`` cell (threaded through via tuple returns),
    and a head -- while still satisfying the recurrent algorithms' contract:

        (carry, obs, action, reward, done) -> (carry, q_values)

    The recurrent algorithms run a single stream (parallelism comes from an
    outer ``vmap`` over seeds), so ``initialize_carry`` builds an unbatched
    carry taken from the first ``nn.RNNCellBase`` among the layers.
    """

    @nn.nowrap
    def initialize_carry(self, rng: Key) -> PyTree:
        for layer in self.layers:
            if isinstance(layer, nn.RNNCellBase):
                return layer.initialize_carry(rng, ())
        raise ValueError(
            "RecurrentSequential.initialize_carry: no nn.RNNCellBase among layers."
        )
