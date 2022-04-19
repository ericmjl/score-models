"""Data generators."""
from jax import random


def make_gaussian():
    """Make Gaussian data.

    :returns: A numpy array of shape (1000, 1).
    """
    data = random.normal(key=random.PRNGKey(44), shape=(1000, 1)) * 1.5 + 3
    return data
