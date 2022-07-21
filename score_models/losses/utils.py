"""Utility loss functions."""
import equinox as eqx
import jax.numpy as np
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
