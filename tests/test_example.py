"""Example test for score_models."""
from score_models.example import hello_world


def test_hello_world():
    """Test for the hello world function.

    Place intent of test here.
    """
    assert hello_world() == "Hello, world!"
