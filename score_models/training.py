"""Training loop defined here."""
from typing import Callable, List, Tuple

import equinox as eqx
import optax
from jax import numpy as np
from jax import value_and_grad
from tqdm.auto import tqdm


def default_optimizer():
    return optax.chain(
        optax.clip(0.01),
        optax.sgd(learning_rate=5e-2),
    )


def adam_optimizer():
    return optax.adam(learning_rate=5e-2)


def fit(
    model: eqx.Module,
    data: np.ndarray,
    loss: Callable,
    optimizer: optax.GradientTransformation,
    steps: int = 1_000,
) -> Tuple[eqx.Module, List]:
    """Fit model to data.

    :param model: An Equinox Module.
    :param data: Data to fit to of shape (batch, :)
    :param loss: Loss function.
    :param optimizer: The optimizer to use.
    :param steps: Number of steps to train for.
    :returns: A tuple of updated model + training loss history.
    """
    opt_state = optimizer.init(model)
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
    for _ in tqdm(range(steps)):
        model, opt_state, loss_score = step(model, data, opt_state)
        loss_history.append(loss_score)
    return model, loss_history
