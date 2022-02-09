"""Utilities for score-models."""
from functools import partial
from typing import Callable

import numpy as np
from jax import grad, vmap


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
