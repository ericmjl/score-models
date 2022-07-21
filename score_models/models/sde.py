"""SDE score models and classes."""
import equinox as eqx
import jax.numpy as np
from diffrax import (
    ControlTerm,
    Heun,
    MultiTerm,
    ODETerm,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
)
from jax import nn, random, vmap


class SDE(eqx.Module):
    """Equinox SDE module.

    Wraps a very common SDE code pattern into a single object.
    """

    drift: callable
    diffusion: callable

    def __call__(self, ts: np.ndarray, y0: float, key: random.PRNGKey) -> np.ndarray:
        """Solve an SDE model.

        :param ts: Time steps to follow through.
        :param y0: Initial value.
        :param key: PRNG key for reproducibility purposes.
        :returns: The trajectory starting from y0.
        """
        brownian_motion = VirtualBrownianTree(
            ts[0], ts[-1], tol=1e-3, shape=(), key=key
        )
        terms = MultiTerm(
            ODETerm(self.drift), ControlTerm(self.diffusion, brownian_motion)
        )
        solver = Heun()
        saveat = SaveAt(t0=True, ts=ts, dense=True)
        sol = diffeqsolve(
            terms, solver, t0=ts[0], t1=ts[-1], dt0=ts[1] - ts[0], y0=y0, saveat=saveat
        )
        return vmap(sol.evaluate)(ts)


class SDEScoreModel(eqx.Module):
    """Time-dependent score model.

    We choose an MLP here with 2 inputs (`x` and `t` concatenated),
    and output a scalar which is the estimated score.
    """

    mlp: eqx.Module

    def __init__(
        self,
        data_dims=2,
        width_size=256,
        depth=2,
        activation=nn.softplus,
        key=random.PRNGKey(45),
    ):
        """Initialize module.

        :param data_dims: The number of data dimensions.
            For example, 2D Gaussian data would have data_dims = 2.
        :param width_size: Width of the hidden layers.
        :param depth: Number of hidden layers.
        :param activation: Activation function.
            Should be passed in uncalled.
        :param key: jax Random key value pairs.
        """
        self.mlp = eqx.nn.MLP(
            in_size=data_dims + 1,  # +1 for the time dimension
            out_size=data_dims,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=key,
        )

    @eqx.filter_jit
    def __call__(self, x: np.ndarray, t: float):
        """Forward pass.

        :param x: Data. Should be of shape (1, :),
            as the model is intended to be vmapped over batches of data.
        :param t: Time in the SDE.
        :returns: Estimated score of a Gaussian.
        """
        t = np.array([t])
        x = np.concatenate([x, t])
        return self.mlp(x)
