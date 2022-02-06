"""Langevin dynamics samplers."""
from typing import Callable

from jax import lax
from jax import numpy as np
from jax import random, vmap


def langevin_dynamics(
    n_chains: int,
    n_samples: int,
    key: random.PRNGKey,
    epsilon: float,
    score_func: Callable,
    params: tuple,
    init_scale: float,
):
    """MCMC with Langevin dynamics to sample from the data generating distribution.

    Example:

        >>> from score_models.sampler import langevin_dynamics
        >>> chain_samples = langevin_dynamics(
        ...     n_chains=4000,
        ...     n_samples=2000,
        ...     key=key,
        ...     epsilon=epsilon,
        ...     score_func=nn_score_func,  # a score function model.
        ...     params=result.params,      # result of optimization
        ...     init_scale=5
        ... )

    :param n_chains: Number of chains to run for sampling.
    :param n_samples: Number of samples to generate from each chain.
    :param key: JAX PRNGKey.
    :param epsilon: A small number.
        A sane default probably is on the order of
        1/1000th of the magnitude of the data.
    :param score_func: Callable that gives the score function of the data.
        Can be, for example, a neural network function.
    :param params: Parameters to the score function.
        Can be, for example, parameters to a neural network function.
    :param init_scale: Scale parameter for the Gaussian
        from which chains are initialized.
    :returns: An array of samples of shape (n_chains, n_samples).
    """

    def langevin_dynamics_one_chain(
        x: float,
        key: random.PRNGKey,
    ):
        """One chain of Langevin dynamics sampling.

        Used for sampling from the data generating distribution.

        :param x: One sample from the data generating distribution.
        :param key: JAX PRNGKey.
        :returns: Samples from one chain of Langevin dynamics sampling.
        """

        def inner(prev_x, key):
            """Scannable closure for one step of Langevin dynamics sampling.

            :param prev_x: The previously sampled value.
            :param key: JAX PRNGKey.
            :returns: A tuple of a (new_draw, prev_draw) from the sampler.
            """
            draw = random.normal(key, shape=(1,))
            new_x = (
                prev_x
                + epsilon * score_func(params, prev_x)
                + np.sqrt(2 * epsilon) * draw
            )
            return new_x, prev_x

        keys = random.split(key, n_samples)
        _, xs = lax.scan(inner, init=x, xs=keys)
        return np.concatenate(xs)

    starter_xs = random.normal(key, shape=(n_chains,)).reshape(-1, 1) * init_scale
    keys = random.split(key, num=n_chains)
    samples = vmap(langevin_dynamics_one_chain)(starter_xs, keys)
    return samples
