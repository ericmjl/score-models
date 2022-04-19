"""Custom model code for score-models."""
from typing import Callable, Tuple

import jax.numpy as np
from jax import jacfwd, random
from jax.scipy.stats import norm


def gaussian_model() -> Tuple[Callable, Callable]:
    """Gaussian model in stax form.

    :returns: An (init_fun, apply_fun) pair.
    """

    def init_fun(key: random.PRNGKey, *args, **kwargs):
        """Initialization function.

        :param key: JAX PRNGKey.
        :param args: Unused by the function.
            Just present to make function match API of stax.
        :param kwargs: Unused by the function.
            Just present to make function match API of stax.
        :returns: A tuple of (None, [mu, log_sigma]).
        """
        mu, log_sigma = random.normal(key, shape=(2, 1))
        return None, (mu, log_sigma)

    def apply_fun(params, x):
        """Apply function.

        :param params: Neural network params from init_fun.
        :param x: one sample of data.
        :returns: Score function w.r.t. x.
        """
        gaussian_score_func = jacfwd(norm.logpdf)
        mu, log_sigma = params
        return gaussian_score_func(x, loc=mu, scale=np.exp(log_sigma)).squeeze()

    return init_fun, apply_fun
