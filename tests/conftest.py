import jax
import pytest


@pytest.fixture
def rng():
    init_rng = jax.random.PRNGKey(0)
    return init_rng
