import pytest

from protein_classification import __main__


@pytest.fixture
def mock_main_env(monkeypatch):
    to_mock = [
        ('protein_classification.__main__.DATA_DIR', ''),
        ('protein_classification.__main__.NUM_EPOCHS', 1)
    ]
    for key, val in to_mock:
        monkeypatch.setattr(key, val)


@pytest.fixture
def mock_path(monkeypatch):
    from io import TextIOWrapper
    
    class MockIO:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self, *args, **kwargs):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def write(self, *args, **kwargs):
            pass
    
    class MockPath:
        def __init__(self, *args, **kwargs):
            pass

        def __truediv__(self, other):
            return MockPath()

        def exists(self):
            return True

        def open(self, *args, **kwargs):
            return MockIO(*args, **kwargs)

    monkeypatch.setattr(
        'protein_classification.__main__.Path',
        MockPath
    )


def test_main(mock_wandb, mock_main_env, mock_path, mock_jnp_load_2):
    __main__.main()
