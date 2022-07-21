"""Neural ODE models and associated classes."""
import equinox as eqx
import jax.numpy as np
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve
from jax import vmap


class ODE(eqx.Module):
    """Equinox ODE module.

    Wraps a very common ODE code pattern into a single object.
    """

    drift: callable

    def __call__(self, ts: np.ndarray, y0: float) -> np.ndarray:
        """Solve an ODE model.

        :param ts: Time steps to follow through.
        :param y0: Initial value.
        :returns: The trajectory starting from y0.
        """
        term = ODETerm(self.drift)
        solver = Tsit5()
        saveat = SaveAt(ts=ts, dense=True)
        sol = diffeqsolve(
            term, solver, t0=ts[0], t1=ts[-1], dt0=ts[1] - ts[0], y0=y0, saveat=saveat
        )
        return vmap(sol.evaluate)(ts)
