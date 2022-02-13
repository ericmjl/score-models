"""Utilities for score-models."""
from functools import partial
from typing import Callable

import numpy as np
from jax import grad, random, vmap


def batchify_and_grad(
    score_func: Callable, params: tuple, batch: np.ndarray
) -> tuple[Callable, Callable]:
    """Utility function to batchify a function.

    This convenience function is useful because
    the score function is defined as
    the partial derivative w.r.t. each dimension of our data.
    In calculating the loss, we need the gradient w.r.t. each scalar in our data,
    thus we need to be able to vmap the function over all of our dimensions.

    :param score_func: The score function.
    :param params: Params to the score function (and its corresponding gradient).
    :param batch: Data of shape (batch, :),
        where : is any arbitrary number of dimensions >= 1.
    :returns: A callable that will operate on an entire batch of data.
    """
    ndims = len(batch.shape)
    score_func = partial(score_func, params)
    dscore_func = grad(score_func)
    for _ in range(ndims):
        score_func = vmap(score_func)
        dscore_func = vmap(dscore_func)
    return score_func, dscore_func


def generate_mixture_2d(key) -> tuple[np.ndarray, random.PRNGKey]:
    """Generate a mixture 2D gaussian.

    :param key: A JAX PRNGKey.
    :returns: A 2-tuple of (data, PRNGKey).
    """
    k1, k2, k3, k4 = random.split(key, 4)

    mu1 = np.array([10, 10])
    cov1 = np.eye(2)
    mix1 = random.multivariate_normal(k1, mean=mu1, cov=cov1, shape=(100,))

    mu2 = np.array([-10, -10])
    cov2 = np.eye(2)
    mix2 = random.multivariate_normal(k2, mean=mu2, cov=cov2, shape=(100,))

    mu3 = np.array([-10, 10])
    cov3 = np.eye(2)
    mix3 = random.multivariate_normal(k3, mean=mu3, cov=cov3, shape=(100,))

    data = np.concatenate([mix1, mix2, mix3])
    return data, k4
