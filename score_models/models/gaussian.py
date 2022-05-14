"""Gaussian score function implementations."""
from functools import partial

import equinox as eqx
from jax import jacfwd
from jax import numpy as np
from jax import vmap
from jax.scipy.stats import norm


class GaussianModel(eqx.Module):
    """Univariate Gaussian score function."""

    mu: np.array = np.array(0.0)
    log_sigma: np.array = np.array(0.0)

    @eqx.filter_jit
    def __call__(self, x):
        """Forward pass.

        :param x: Data. Should be of shape (1, :),
            as the model is intended to be vmapped over batches of data.
        :returns: Score of a Gaussian conditioned on a `mu` and `log_sigma`.
        """
        gaussian_score_func = jacfwd(norm.logpdf)
        return gaussian_score_func(x, loc=self.mu, scale=np.exp(self.log_sigma))


class SDEGaussianModel(eqx.Module):
    r"""Time-dependent Gaussian score function.

    Key trick here is to make the parameters $\mu$ and $\log(\sigma)$ depend on $t$,
    i.e. the time at which a Gaussian was sampled.
    """

    mu_w: np.array = np.array(0.0)
    mu_b: np.array = np.array(0.0)
    log_sigma_w: np.array = np.array(0.0)
    log_sigma_b: np.array = np.array(0.0)

    @eqx.filter_jit
    def __call__(self, x, t):
        """Forward pass.

        :param x: Data. Should be of shape (1, :),
            as the model is intended to be vmapped over batches of data.
        :param t: Time step at which evaluation is happening.
        :returns: Score of a Gaussian conditioned on a `mu` and `log_sigma`.
        """
        gaussian_score_func = jacfwd(norm.logpdf)
        mu = self.mu_w * t + self.mu_b
        log_sigma = self.log_sigma_w * t + self.log_sigma_b
        return gaussian_score_func(x, loc=mu, scale=np.exp(log_sigma))

    # @property
    # def mu(self, t):
    #     return self.mu_w * t + self.mu_b

    # @property
    # def log_sigma(self, t):
    #     return self.log_sigma_w * t + self.log_sigma_b


class MixtureGaussian(eqx.Module):
    """Mixture Gaussian score function."""

    mus: np.array
    log_sigmas: np.array
    ws: np.array

    def __init__(self, mus, log_sigmas, ws):
        self.mus = mus
        self.log_sigmas = log_sigmas
        self.ws = ws

        # Check that mus, log_sigmas, and ws are of the same length.
        lengths = set(map(len, [mus, log_sigmas, ws]))
        if len(lengths) != 1:
            raise ValueError(
                "`mus`, `log_sigmas` and `ws` must all be of the same length!"
            )

    @eqx.filter_jit
    def __call__(self, x):
        """Forward pass.

        :param x: Data. Should be of shape (1, :),
            as the model is intended to be vmapped over batches of data.
        :returns: Score of a Gaussian conditioned on a `mu` and `log_sigma`.
        """
        return partial(
            dmixture_logpdf,
            mus=self.mus,
            sigmas=np.exp(self.log_sigmas),
            ws=self.ws,
        )(x)


def mixture_pdf(x, mus, sigmas, ws):
    """Mixture likelihood.

    :param x: Data. Should be of shape (1, :),
        as the model is intended to be vmapped over batches of data.
    :param mus: Mixture component locations.
    :param sigmas: Mixture component scales.
    :param ws: Mixture component weights.
    :returns: The likelihood of a mixture PDF evaluated at `x`.
    """
    component_pdfs = vmap(partial(norm.pdf, x))(mus, sigmas)  # 2, n_draws)
    scaled_component_pdfs = vmap(np.multiply)(component_pdfs, ws)
    total_pdf = np.sum(scaled_component_pdfs, axis=0)
    return total_pdf


def mixture_logpdf(x, mus, sigmas, ws):
    """Mixture loglikelihood.

    :param x: Data. Should be of shape (1, :),
        as the model is intended to be vmapped over batches of data.
    :param mus: Mixture component locations.
    :param sigmas: Mixture component scales.
    :param ws: Mixture component weights.
    :returns: The log likelihood of a mixture PDF evaluated at `x`.
    """
    return np.log(mixture_pdf(x, mus, sigmas, ws))


dmixture_logpdf = jacfwd(mixture_logpdf, argnums=0)
