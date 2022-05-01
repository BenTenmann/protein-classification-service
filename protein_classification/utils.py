from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import srsly
from flax import serialization

from .model import ResNet
from .tokenize import (
    Tokenizer
)

__all__ = [
    'load_tokenizer',
    'load_model'
]


def load_tokenizer(path: Path or str) -> Tokenizer:
    """
    Load a Tokenizer object by specifying a target token map.

    Parameters
    ----------
    path: Path or str
        Path specifying the target token map in `.json` format.

    Returns
    -------
    out: Tokenizer
        The initialised Tokenizer object.
    """
    token_map = srsly.read_json(path)
    out = Tokenizer(token_map)
    return out


def load_model(params_path: str, config_map_path: str, sequence_length: int) -> Callable:
    """
    Load the ResNet model from a parameters file and a model manifest file. It is a convenience function wrapping model
    init and the model apply, such that the model can be used as a callable, much like PyTorch modules.

    Parameters
    ----------
    params_path: str
        A path to a binary file containing the stored model weights and parameters.
    config_map_path: str
        Filepath to a YAML specifying the model hyperparameters.
    sequence_length: int
        The maximum sequence length for a model input sequence. Is used for initializing the Flax model.

    Returns
    -------
    predict_fn: Callable
        Returns a function which wraps the `model.apply` method. This callable takes a DeviceArray as input and returns
        the model logits.
    """
    def load_params(variables):
        file = Path(params_path)
        byte_str = file.read_bytes()
        params_ = serialization.from_bytes(variables, byte_str)
        return params_

    conf = srsly.read_yaml(config_map_path)
    model = ResNet(**conf)
    batch = jnp.ones((1, sequence_length))
    init_rng = jax.random.PRNGKey(0)
    var = model.init(init_rng, batch)
    params = load_params(var)

    def predict_fn(x: jnp.ndarray) -> jnp.ndarray:
        # wrapper around model.apply to remove need for handling model parameters in service
        y_hat = model.apply(params, x)
        return y_hat
    return predict_fn
