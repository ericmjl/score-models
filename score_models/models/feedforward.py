"""Feed Forward Neural Network Models."""
import equinox as eqx
from jax import nn
from jax import numpy as np
from jax import random


class FeedForwardModel(eqx.Module):
    """Feed-forward NN model."""

    mlp: eqx.Module

    def __init__(
        self, in_size=1, out_size=1, width_size=4096, depth=1, activation=nn.relu
    ):
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=random.PRNGKey(45),
        )

    @eqx.filter_jit
    def __call__(self, x):
        """Forward pass.

        :param x: Data. Should be of shape (1, :),
            as the model is intended to be vmapped over batches of data.
        :returns: Estimated score of a Gaussian.
        """
        return np.expand_dims(self.mlp(x), -1)
