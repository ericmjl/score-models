"""Implementation of loss function for score matching."""
from functools import partial
from typing import Callable

from jax import jacfwd
from jax import numpy as np
from jax import vmap
from jax.tree_util import tree_flatten, tree_map


def l2_norm(params) -> float:
    """Return the sum of square of weights.

    Allows for L2 norm-based regularization of weight parameter.
    Intended to be used as part of a loss function.

    Example:

    >>> from score_models.losses import l2_norm
    >>> param = [(0.5, 0.3), (0.1, -1.3)]
    >>> l2_norm(param)
    DeviceArray(2.0399997, dtype=float32)

    :param params: Any PyTree of parameters.
    :returns: L2 norm.
    """
    # Test: tree-map np.sum to get weight regularization term
    squared = tree_map(lambda x: np.power(x, 2), params)
    summed = tree_map(np.sum, squared)
    flattened, _ = tree_flatten(summed)
    return np.sum(np.array(flattened))


def score_matching_loss(params, score_func: Callable, batch: np.ndarray) -> float:
    """Score matching loss function.

    This is taken from (Hyvärinen, 2005) (JMLR)
    and https://yang-song.github.io/blog/2019/ssm/.

    :param params: The parameters to the score function.
    :param score_func: Score function with signature `func(params, batch)`,
        which returns a scalar.
    :param batch: A batch of data. Should be of shape (batch, :),
        where `:` refers to at least 1 more dimension.
    :returns: Score matching loss, a float.
    """
    score_func = partial(score_func, params)
    dscore_func = jacfwd(score_func)

    # Jacobian of score function (i.e. dlogp estimator function).
    # In the literature, this is also called the Hessian of the logp.
    # (Recall that the Hessian is the 2nd derivative, while the Jacobian is the 1st.)
    # The Jacobian shape is: `(i, i)`,
    # where `i` is the number of dimensions of the input data,
    # or the number of random variables.
    # Here, we want the diagonals instead, which is of shape (i,)
    term1 = vmap(dscore_func)(batch)
    term1 = vmap(np.diagonal)(term1)

    # Discretized integral of score function.
    term2 = 0.5 * vmap(score_func)(batch) ** 2
    term2 = np.reshape(term2, term1.shape)

    # Summation over the inner term, by commutative property of addition,
    # automagically gives us the trace of the Jacobian of the score function.
    # Yang Song's blog post refers to the trace
    # (final equation in the section
    # "Learning unnormalized models with score matching"),
    # while Hyvärinen's JMLR paper uses an explicit summation in Equation 4.
    inner_term = term1 + term2
    summed_by_dims = vmap(np.sum)(inner_term)
    return np.mean(summed_by_dims) + 0.1 * l2_norm(params)
