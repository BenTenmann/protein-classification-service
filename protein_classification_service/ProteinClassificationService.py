import logging
import os
import traceback
from functools import wraps
from typing import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import srsly
from flax import serialization
from pydantic import BaseModel, Field

from protein_classification.model import ResNet
from protein_classification.utils import load_tokenizer

# ----- Logging Setup ------------------------------------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(asctime)s.%(msecs)03d - %(levelname)s - %(module)s.%(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ----- Environment Variables ---------------------------------------------------------------------------------------- #
CONFIG_MAP = os.environ.get('CONFIG_MAP')
LOGIT_MAP = os.environ.get('LOGIT_MAP')
MODEL_WEIGHTS = os.environ.get('MODEL_WEIGHTS')
TOKEN_MAP = os.environ.get('TOKEN_MAP')
SEQUENCE_MAX_LEN = os.environ.get('SEQUENCE_MAX_LEN', 256)


# ----- Error Handling ----------------------------------------------------------------------------------------------- #
def catch_and_log_errors(fn):
    @wraps(fn)
    def _method(*args, **kwargs):
        try:
            out = fn(*args, **kwargs)
            return out

        except Exception as ex:
            logging.error(ex)
            logging.error(traceback.print_exc())
            raise ex
    return _method


# ----- Typing ------------------------------------------------------------------------------------------------------- #
class Payload(BaseModel):
    sequence: Sequence[str] = Field(..., example={'sequence': ['EIKKMISEIDKDGSGTIDFEEFLTMMTA']})


class Prediction(BaseModel):
    family_id: str
    family_accession: str
    family_description: str
    score: float


class Response(BaseModel):
    prediction: Sequence[Prediction]


# ----- Helpers ------------------------------------------------------------------------------------------------------ #
def ensure_sequence_integrity(predict_method):
    def is_correctly_formatted(sequence: str) -> bool:
        # ensure that the sequence conforms to expected format
        correctly_formatted = (
                sequence.isupper() and sequence.isalpha()
        )
        return correctly_formatted

    @wraps(predict_method)
    def guarded_predict_method(self, msg: Payload):
        payload = dict(msg)
        incorrectly_formatted = [(pos, sequence) for pos, sequence in enumerate(payload['sequence'])
                                 if not is_correctly_formatted(sequence)]
        if incorrectly_formatted:
            message = '\n'.join(f'  at [{pos}]: {repr(seq)}' for pos, seq in incorrectly_formatted)
            raise ValueError(f'invalid sequences supplied to service:\n{message}')
        response = predict_method(self, msg)
        return response

    return guarded_predict_method


def load_model(sequence_length: int) -> Callable:
    def load_params(variables):
        with open(MODEL_WEIGHTS, 'rb') as file:
            byte_str = file.read()
            params_ = serialization.from_bytes(variables, byte_str)
        return params_

    conf = srsly.read_yaml(CONFIG_MAP)
    model = ResNet(**conf)
    batch = jnp.ones((1, sequence_length))
    init_rng = jax.random.PRNGKey(0)
    var = model.init(init_rng, batch)
    params = load_params(var)

    def predict_fn(x: jnp.ndarray) -> jnp.ndarray:
        y_hat = model.apply(params, x)
        return y_hat
    return predict_fn


# ----- Service------------------------------------------------------------------------------------------------------- #
class ProteinClassificationService:
    tokenizer_args: dict = {
        'padding': 'max_length',
        'truncation': True,
        'max_length': int(SEQUENCE_MAX_LEN)
    }

    def __init__(self):
        self.logit_map = None
        self.model = None
        self.tokenizer = None

    def _tokenize(self, sequence: str) -> jnp.ndarray:
        tok = self.tokenizer(sequence, **self.tokenizer_args)
        out = jnp.array(tok, dtype=jnp.float32)
        return out

    def _prediction_to_output(self, prediction: jnp.ndarray) -> dict:
        predicted_class = prediction.argmax(axis=-1).item()
        predicted_class = str(predicted_class)

        out = self.logit_map[predicted_class]
        logits = nn.softmax(prediction, axis=-1)
        out['score'] = float(jnp.max(logits, axis=-1).item())
        return out

    @catch_and_log_errors
    @ensure_sequence_integrity
    def predict_raw(self, msg: Payload) -> Response:
        payload = dict(msg)
        token_arrays = jnp.stack([self._tokenize(sequence) for sequence in payload['sequence']])
        prediction = self.model(token_arrays)
        formatted_output = map(self._prediction_to_output, prediction)
        response = dict(prediction=list(formatted_output))
        return response

    @catch_and_log_errors
    def load(self):
        logging.info(f'Loading service dependencies (pid {os.getpid()})')
        self.logit_map = srsly.read_json(LOGIT_MAP)
        self.model = load_model(self.tokenizer_args['max_length'])
        self.tokenizer = load_tokenizer(TOKEN_MAP)
