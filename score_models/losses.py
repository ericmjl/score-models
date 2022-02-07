"""Implementation of loss function for score matching."""
from functools import partial

from jax import grad
from jax import numpy as np
from jax import vmap


def score_matching_loss(params, score_func, batch) -> float:
    """Score matching loss function.

    :param params: The parameters to the score function.
    :param score_func: Score function with signature `func(params, batch)`,
        which returns a scalar.
    :param batch: A batch of data. Should be of shape (batch, :),
        where `:` refers to at least 1 more dimension,
        and should return a scalar float.
        We `vmap` the `scorefunc` and `grad(scorefunc)` over the batch dimension.
    :returns: Score matching loss, a float.
    """
    score_func = partial(score_func, params)
    dscorefunc = grad(score_func)
    term1 = vmap(dscorefunc)(batch)
    term2 = 0.5 * vmap(score_func)(batch) ** 2
    inner_term = term1 + term2
    return np.mean(inner_term)
