"""Fast integration tests for training."""
import optax
import pytest

from score_models.data import make_gaussian
from score_models.losses import score_matching_loss
from score_models.models.feedforward import FeedForwardModel
from score_models.models.gaussian import GaussianModel
from score_models.training import fit


@pytest.mark.parametrize("model", [GaussianModel(), FeedForwardModel()])
def test_model_fitting(model):
    data = make_gaussian()
    optimizer = optax.adam(learning_rate=5e-3)
    model, history = fit(model, data, score_matching_loss, optimizer, steps=3)
