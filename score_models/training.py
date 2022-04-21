"""Training loop defined here."""
from typing import Callable, List, Tuple

import equinox as eqx
import optax
from jax import numpy as np
from tqdm.auto import tqdm


def fit(
    model: eqx.Module,
    data: np.ndarray,
    loss: Callable,
    optimizer: optax.GradientTransformation,
    steps: int = 1_000,
    progress_bar: bool = True,
) -> Tuple[eqx.Module, List]:
    """Fit model to data.

    :param model: An Equinox Module.
    :param data: Data to fit to of shape (batch, :)
    :param loss: Loss function.
    :param optimizer: The optimizer to use.
    :param steps: Number of steps to train for.
    :param progress_bar: Whether or not to show a progress bar.
        Defaults to True.
    :returns: A tuple of updated model + training loss history.
    """
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    dloss = eqx.filter_jit(eqx.filter_value_and_grad(loss))

    @eqx.filter_jit
    def step(model, data, opt_state):
        """One step of training loop.

        This closure is jitted to make training run really fast.

        :param model: An Equinox Module.
        :param data: Data to fit to of shape (batch, :)
        :param opt_state: State of optimizer.
        :returns: Stuff. (TODO)
        """
        loss_score, grads = dloss(model, data)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_score

    loss_history = []
    iterator = range(steps)
    if progress_bar:
        iterator = tqdm(iterator)
    for _ in iterator:
        model, opt_state, loss_score = step(model, data, opt_state)
        loss_history.append(loss_score)
    return model, loss_history
