"""Langevin dynamics samplers."""
from typing import Optional

import equinox as eqx
from jax import jit, lax
from jax import numpy as np
from jax import random, vmap


class LangevinDynamicsChain(eqx.Module):
    """Langevin dynamics chain."""

    gradient_func: eqx.Module
    n_samples: int = 1000
    epsilon: float = 5e-3

    @eqx.filter_jit
    def __call__(self, x, key: random.PRNGKey):
        """Callable implementation for sampling.

        :param x: Data of shape (batch, :).
        :param key: PRNGKey for random draws.
        :returns: A tuple of final draw and historical draws."""

        def langevin_step(prev_x, key):
            """Scannable langevin dynamics step.

            :param prev_x: Previous value of x in langevin dynamics step.
            :param key: PRNGKey for random draws.
            :returns: A tuple of new x and previous x.
            """
            draw = random.normal(key, shape=x.shape)
            new_x = (
                prev_x
                + self.epsilon * vmap(self.gradient_func)(prev_x)
                + np.sqrt(2 * self.epsilon) * draw
            )
            return new_x, prev_x

        keys = random.split(key, self.n_samples)
        final_xs, xs = lax.scan(langevin_step, init=x, xs=keys)
        return final_xs, np.vstack(xs)


def langevin_dynamics(
    n_chains: int,
    n_samples: int,
    key: random.PRNGKey,
    epsilon: float,
    score_func: eqx.Module,
    init_scale: float,
    starter_xs: Optional[np.ndarray] = None,
    sample_shape: Optional[tuple] = None,
):
    """MCMC with Langevin dynamics to sample from the data generating distribution.

    Example:

        >>> from score_models.sampler import langevin_dynamics
        >>> chain_samples = langevin_dynamics(
        ...     n_chains=4000,
        ...     n_samples=2000,
        ...     key=key,
        ...     epsilon=epsilon,
        ...     score_func=nn_score_func,  # an eqx.Module score function model.
        ...     init_scale=5,
        ...     sample_shape=(None, 2)
        ... )

    :param n_chains: Number of chains to run for sampling.
    :param n_samples: Number of samples to generate from each chain.
    :param key: JAX PRNGKey.
    :param epsilon: A small number.
        A sane default probably is on the order of
        1/1000th of the magnitude of the data.
    :param score_func: An Equinox module that gives the score function of the data.
        Can be, for example, a neural network function.
    :param init_scale: Scale parameter for the Gaussian
        from which chains are initialized.
    :param starter_xs: Starting values of each chain.
        Its shape should be similar to the observed data;
        instead of `(batch, :)`,
        where `:` refers to arbitrary numbers of dimensions,
        `starter_xs` should be of shape `(n_chains, :)`.
    :param sample_shape: The shape of one observation in the chain.
        Used to initialize the shape of a sample.
    :returns: An array of samples of shape (n_chains, n_samples).
    :raises ValueError: if `starter_xs` and `sample_shape` are both None
    """
    # Defensive check on starter_xs and sample_shape
    if starter_xs is None and sample_shape is None:
        raise ValueError("`starter_xs` and `sample_shape` cannot both be None!")

    if sample_shape is None:
        sample_shape = (None, *starter_xs.shape[1:])
    if starter_xs is None:
        starter_xs = (
            random.normal(key, shape=(n_chains, *sample_shape[1:])) * init_scale
        )

    @jit
    def langevin_dynamics_one_chain(
        x: float,
        key: random.PRNGKey,
    ):
        """One chain of Langevin dynamics sampling.

        Used for sampling from the data generating distribution.

        :param x: One sample from the data generating distribution.
        :param key: JAX PRNGKey.
        :returns: Final states and samples from one chain of Langevin dynamics sampling.
        """

        def inner(prev_x, key):
            """Scannable closure for one step of Langevin dynamics sampling.

            :param prev_x: The previously sampled value.
            :param key: JAX PRNGKey.
            :returns: A tuple of a (new_draw, prev_draw) from the sampler.
            """
            draw = random.normal(key, shape=sample_shape[1:])
            new_x = prev_x + epsilon * score_func(prev_x) + np.sqrt(2 * epsilon) * draw
            return new_x, prev_x

        keys = random.split(key, n_samples)
        final_xs, xs = lax.scan(inner, init=x, xs=keys)
        return final_xs, np.vstack(xs)

    keys = random.split(key, num=n_chains)
    final_samples, samples = vmap(langevin_dynamics_one_chain)(starter_xs, keys)
    return starter_xs, final_samples, samples
