"""Custom model code for score-models."""
from functools import partial
from typing import Callable, Tuple

import jax.numpy as np
from jax import grad
from jax.example_libraries import stax
from jax.scipy.stats import norm


def gaussian_score_func(params: Tuple[float, float], draw: float) -> float:
    """Gaussian score function for 1 draw.

    Example:

        >>> from score_models.models import gaussian_score_func
        >>> params = (3.0, 1.0)  # true_mu, true_sigma
        >>> gaussian_score_func(params, 3.0)
        DeviceArray(-0., dtype=float32)

    :param params: Tuple of `mu` and `log_sigma`.
    :param draw: One observed value.
    :returns: Data score (conditioned on `params`).
    """
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    logp_func = partial(norm.logpdf, loc=mu, scale=sigma)
    return grad(logp_func)(draw)


def nn_model() -> Tuple[Callable, Callable]:
    """Simple neural network model for predicting score from.

    Example:

        >>> from score_models.models import nn_model
        >>> init_fun, apply_fun = nn_model()

    :returns: (init_fun, apply_fun) pair.
    """
    init_fun, apply_fun = stax.serial(stax.Dense(1024), stax.Relu, stax.Dense(1))
    return init_fun, apply_fun


def nn_score_func(params, draw: float):
    """Scalar version of the score function.

    A hack to get around the shape issues that plague tensor math :).
    `grad` requires the output to be a scalar, so we need to hack and squeeze scores.

    `batch` should be a scalar.

    Example:

        >>> from score_models.models import nn_model, nn_score_func
        >>> from jax.random import PRNGKey
        >>> init_fun, apply_fun = nn_model()
        >>> key = PRNGKey(44)
        >>> _, params = init_fun(key, input_shape=(None, 1))
        >>> nn_score_func(params, 0.5)
        DeviceArray(-0.01079839, dtype=float32)

    :param params: Parameters generated from `nn_model()`'s initialization function.
    :param draw: One observed value.
    :return: Predicted score, given parameters.
    """
    _, apply_fun = nn_model()
    # The reshape(-1, 1) is necessary to interface with stax' Dense layers.
    apply_fun = partial(apply_fun, params)
    draw = np.reshape(draw, (-1, 1))
    scores = apply_fun(draw).squeeze()
    return scores


# def score_func_haiku(params, batch):
#     pass
