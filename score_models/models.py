"""Custom model code for score-models."""
from typing import Callable, Tuple

import jax.numpy as np
from jax import jacfwd, random
from jax.example_libraries import stax
from jax.scipy.stats import norm


def gaussian_model() -> Tuple[Callable, Callable]:
    """Gaussian model.

    :returns: An (init_fun, apply_fun) pair.
    """

    def init_fun(key: random.PRNGKey, **kwargs):
        """Initialization function.

        :param key: JAX PRNGKey.
        :param kwargs: Unused by the function.
            Just present to make function match API of stax.
        :returns: A tuple of (None, [mu, log_sigma]).
        """
        mu, log_sigma = random.normal(key, shape=(2, 1))
        return None, (mu, log_sigma)

    def apply_fun(params, x):
        """Apply function.

        :param params: Neural network params from init_fun.
        :param x: one sample of data.
        :returns: Score function w.r.t. x.
        """
        gaussian_score_func = jacfwd(norm.logpdf)
        mu, log_sigma = params
        return gaussian_score_func(x, loc=mu, scale=np.exp(log_sigma)).squeeze()

    return init_fun, apply_fun


def nn_model(output_dim: int = 1) -> Tuple[Callable, Callable]:
    """Simple neural network model for predicting score from.

    Example:

        >>> from score_models.models import nn_score_func
        >>> init_fun, apply_fun = nn_score_func()

    :param output_dim: Number of dimensions for the model to output.
    :returns: (init_fun, apply_fun) pair.
    """
    init_fun, apply_fun = stax.serial(
        stax.Dense(1024), stax.Relu, stax.Dense(output_dim)
    )
    return init_fun, apply_fun
