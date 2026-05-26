import distrax
import flax.linen as nn
from stremax.utils.typing import Array


class Categorical(nn.Module):
    action_dim: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> distrax.Categorical:
        logits = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        return distrax.Categorical(logits=logits)
