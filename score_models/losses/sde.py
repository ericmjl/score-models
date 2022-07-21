"""Score matching loss functions for probability flow score models."""
from functools import partial
from typing import Callable, Union

import equinox as eqx
import jax.numpy as np
from jax import jacfwd, vmap

from .utils import l2_norm


def score_matching_loss(
    model: Union[eqx.Module, Callable], noised_data: np.ndarray, t: float
) -> float:
    """Score matching loss for SDE-based score models.

    :param model: Equinox model.
    :param noised_data: Batch of data from 1 noise scale of shape (batch, n_data_dims).
    :param t: Time in SDE at which the noise scale was evaluated.
    :returns: Score matching loss for one batch of data.
    """
    model = partial(model, t=t)
    dmodel = jacfwd(model, argnums=0)
    term1 = vmap(dmodel)(noised_data)
    if term1.ndim > 1:
        term1 = vmap(np.diagonal)(term1)
    term2 = 0.5 * vmap(model)(noised_data) ** 2
    inner_term = term1 + term2
    summed_by_dims = vmap(np.sum)(inner_term)
    return np.mean(summed_by_dims)


@eqx.filter_jit
def joint_score_matching_loss(
    model: Union[eqx.Module, Callable], noised_data_all: np.ndarray, ts: np.ndarray
):
    """Joint score matching loss.

    :param model: An equinox model.
    :param noised_data_all: An array of shape (time, batch, n_data_dims).
    :param ts: An array of shape (time,).
    :returns: Score matching loss, summed across all noise scales.
    """
    loss_score = 0
    for noise_batch, t in zip(noised_data_all, ts):
        scale = t
        loss_score += score_matching_loss(model, noise_batch, t) * scale
        loss_score += l2_norm(model, noise_batch)
    return loss_score
