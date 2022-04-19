import equinox as eqx
from jax import jacfwd, jit
from jax import numpy as np
from jax.scipy.stats import norm


class GaussianModel(eqx.Module):
    mu: float = 0.0
    log_sigma: float = 0.0

    @jit
    def __call__(self, x):
        gaussian_score_func = jacfwd(norm.logpdf)
        return gaussian_score_func(x, loc=self.mu, scale=np.exp(self.log_sigma))
