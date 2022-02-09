"""Tests for score-models's machine learning models."""
from functools import partial

from jax import random, vmap

from score_models.models import gaussian_model, nn_model


def test_gaussian_model():
    """Test for Gaussian model."""
    key = random.PRNGKey(23)
    init_fun, apply_fun = gaussian_model()
    _, params = init_fun(key)

    data = random.normal(key, shape=(3,))
    out = vmap(partial(apply_fun, params))(data)
    assert out.shape == (3,)


def test_nn_model():
    """Test for NN model."""
    key = random.PRNGKey(49)
    init_fun, apply_fun = nn_model(output_dim=1)
    _, params = init_fun(key, (None, 1))

    data = random.normal(key, shape=(3, 1))

    out = vmap(partial(apply_fun, params))(data)
    assert out.shape == (3, 1)
