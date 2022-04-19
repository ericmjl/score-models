import equinox as eqx
from jax import nn
from jax import numpy as np
from jax import random


class FeedForwardModel(eqx.Module):
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
        # return np.expand_dims(self.mlp(x), -1)
        return self.mlp(x)


# def FeedForwardModel():
#     return eqx.nn.MLP(
#         in_size=1,
#         out_size=1,
#         width_size=1024,
#         depth=1,
#         key=random.PRNGKey(45),
#     )


# def squeezify(apply_fun):
#     @wraps(apply_fun)
#     def inner(params, x):
#         return apply_fun(params, x).squeeze()

#     return inner


# def nn_model(
#     output_dim: int = 1,
#     hidden_dim: int = 1024,
#     nonlin=stax.Relu,
#     num_dense_blocks: int = 1,
# ) -> Tuple[Callable, Callable]:
#     """Simple neural network model for predicting score from.

#     Example:

#         >>> from score_models.models import nn_score_func
#         >>> init_fun, apply_fun = nn_score_func()

#     :param output_dim: Number of dimensions for the model to output.
#     :param nonlinearity: Nonlinearity function to use.
#     :returns: (init_fun, apply_fun) pair.
#     """
#     dense_layers = [stax.Dense(hidden_dim), nonlin] * num_dense_blocks
#     init_fun, apply_fun = stax.serial(
#         *dense_layers,
#         stax.Dense(output_dim),
#     )
#     return init_fun, squeezify(apply_fun)
