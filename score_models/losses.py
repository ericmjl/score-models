"""Implementation of loss function for score matching."""

import equinox as eqx
from jax import jacfwd
from jax import numpy as np
from jax import vmap
from jax.tree_util import tree_flatten, tree_map


def l2_norm(model: eqx.Module, batch: np.ndarray, scale: float = 0.1) -> float:
    """Return the sum of square of weights.

    Allows for L2 norm-based regularization of weight parameter.
    Intended to be used as part of a loss function.

    Example:

    >>> from score_models.losses import l2_norm
    >>> param = [(0.5, 0.3), (0.1, -1.3)]
    >>> l2_norm(param)
    DeviceArray(2.0399997, dtype=float32)

    :param model: An Equinox Module.
    :param batch: Present only for compatibility with the rest of the loss functions.
    :param scale: The scale of L2 regularization to apply. Defaults to 0.1
    :returns: L2 norm.
    """
    # Test: tree-map np.sum to get weight regularization term
    model = eqx.filter(model, eqx.is_array_like)
    squared = tree_map(lambda x: np.power(x, 2), model)
    summed = tree_map(np.sum, squared)
    flattened, _ = tree_flatten(summed)
    return np.sum(np.array(flattened)) * scale


def score_matching_loss(model: eqx.Module, batch: np.ndarray) -> float:
    """Score matching loss function.

    This is taken from (Hyvärinen, 2005) (JMLR)
    and https://yang-song.github.io/blog/2019/ssm/.

    :param model: An Equinox Module.
    :param batch: A batch of data. Should be of shape (batch, :),
        where `:` refers to at least 1 more dimension.
    :returns: Score matching loss, a float.
    """
    dmodel = jacfwd(model)

    # Jacobian of score function (i.e. dlogp estimator function).
    # In the literature, this is also called the Hessian of the logp.
    # (Recall that the Hessian is the 2nd derivative, while the Jacobian is the 1st.)
    # The Jacobian shape is: `(i, i)`,
    # where `i` is the number of dimensions of the input data,
    # or the number of random variables.
    # Here, we want the diagonals instead, which when extracted out, is of shape (i,)
    term1 = vmap(dmodel)(batch)
    if len(term1.shape) == 2 and term1.shape[-1] == 1:
        term1 = np.expand_dims(term1, -1)
    term1 = vmap(np.diagonal)(term1)

    # Discretized integral of score function.
    term2 = 0.5 * vmap(model)(batch) ** 2
    term2 = np.reshape(term2, term1.shape)

    # Summation over the inner term, by commutative property of addition,
    # automagically gives us the trace of the Jacobian of the score function.
    # Yang Song's blog post refers to the trace
    # (final equation in the section
    # "Learning unnormalized models with score matching"),
    # while Hyvärinen's JMLR paper uses an explicit summation in Equation 4.
    inner_term = term1 + term2
    summed_by_dims = vmap(np.sum)(inner_term)
    return np.mean(summed_by_dims)


def chain(*loss_funcs):
    """Chain loss functions together.

    All loss funcs must have the signature loss(model, batch).

    :param loss_funcs: Loss functions to chain together.
    :returns: A closure.
    """

    def chained(model, batch):
        """Chained loss function.

        This loss simply adds up the losses computed by
        the loss functions defined in the outer function.

        :param model: Equinox model.
        :param batch: A batch of data.
        :returns: Total loss across all loss functions.
        """
        loss_score = 0
        for loss in loss_funcs:
            loss_score += loss(model, batch)
        return loss_score

    return chained


def joint_score_matching_loss(
    models: list[eqx.Module], datas: np.ndarray, scales: np.ndarray
) -> float:
    """Joint score matching loss.

    :param models: A list of Equinox models.
    :param datas: A collection of noised up data.
        Leading axis should be of length n_noise_scales,
        or in an SDE case, the number of time steps.
    :param scales: A NumPy array of scales.
        In the SDE case, this would be an array of diffusion values.
    :returns: A float.
    """
    loss = 0
    for model, data, scale in zip(models, datas, scales):
        loss = loss + score_matching_loss(model, data) * scale
    return loss


def sde_joint_score_matching_loss(
    models: list[eqx.Module], datas: np.ndarray, scales: np.ndarray, ts: np.ndarray
) -> float:
    """Joint score matching loss.

    :param models: A list of Equinox models.
    :param datas: A collection of noised up data.
        Leading axis should be of length n_noise_scales,
        or in an SDE case, the number of time steps.
    :param scales: A NumPy array of scales.
        In the SDE case, this would be an array of diffusion values.
    :param ts: Timepoints at which loss is being evaluated.
    :returns: A float.
    """
    loss = 0
    for model, data, scale, t in zip(models, datas, scales, ts):
        loss = loss + score_matching_loss(model, data, t) * scale
    return loss
