import logging
import os
import traceback
from functools import wraps
from typing import Sequence

import srsly
import torch
from torch import nn
from pydantic import BaseModel, Field

import protein_classification.model as mdl
from protein_classification.constants import DEVICE
from protein_classification.dataset import Tokenizer

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
SEQUENCE_MAX_LEN = os.environ.get('SEQUENCE_MAX_LEN', 512)


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


# ----- Service------------------------------------------------------------------------------------------------------- #
class ProteinClassificationService:
    _tokenizer_args: dict = {
        'padding': 'max_length',
        'truncation': True,
        'max_length': int(SEQUENCE_MAX_LEN)
    }

    def __init__(self):
        self.logit_map = None
        self.model = None
        self.tokenizer = None

    def _tokenize(self, sequence: str) -> torch.Tensor:
        tok = self.tokenizer(sequence, **self._tokenizer_args)
        out = torch.tensor(tok, dtype=torch.float32, device=DEVICE)
        return out

    def _prediction_to_output(self, prediction: torch.Tensor) -> dict:
        predicted_class = prediction.argmax(dim=-1).item()
        predicted_class = str(predicted_class)

        out = self.logit_map[predicted_class]
        logits = torch.softmax(prediction, dim=-1)
        out['score'] = float(torch.amax(logits, dim=-1).item())
        return out

    @catch_and_log_errors
    def predict_raw(self, msg: Payload) -> Response:
        payload = dict(msg)
        token_tensors = torch.stack([self._tokenize(sequence) for sequence in payload['sequence']])
        prediction = self.model(token_tensors)
        formatted_output = map(self._prediction_to_output, prediction)
        response = dict(prediction=list(formatted_output))
        return response

    @staticmethod
    def _load_tokenizer() -> Tokenizer:
        token_map = srsly.read_json(TOKEN_MAP)
        tokenizer = Tokenizer(token_map)
        return tokenizer

    @staticmethod
    def _load_model() -> nn.Module:
        conf = srsly.read_yaml(CONFIG_MAP)
        model = getattr(mdl, conf.get('name', 'MLP'))
        param = conf.get('param', {})
        out = model(**param)

        out.load_state_dict(torch.load(MODEL_WEIGHTS))
        out.to(DEVICE)
        return out

    def load(self):
        self.logit_map = srsly.read_json(LOGIT_MAP)
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
