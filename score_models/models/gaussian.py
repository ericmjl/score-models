import equinox as eqx
from jax import jacfwd, jit
from jax import numpy as np
from jax.scipy.stats import norm


class GaussianModel(eqx.Module):
    mu: np.array = np.array(0.0)
    log_sigma: np.array = np.array(0.0)

    @eqx.filter_jit
    def __call__(self, x):
        gaussian_score_func = jacfwd(norm.logpdf)
        return gaussian_score_func(x, loc=self.mu, scale=np.exp(self.log_sigma))
