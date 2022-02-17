"""
protein_classification
=====================

This package contains the model and tokenizer objects for the protein classification service. It can also be run as a
module from the command line, like so:

$ python3 -m protein_classification

In this case, it acts as the training and benchmarking script for potential models. To do so, a number of environment
variables need to be set.

Parameters
----------
CONFIG_MAP: str
    Path to training config map (`.yml`). The config map defines model parameters, number of epochs, learning rate etc. See
    `${REPO}/manifest` for examples.
TRAIN_DATA: str
    Path to `.jsonl` file containing the training examples.
DEV_DATA: str
    Path to `.jsonl` file containing the development (validation) examples.
TEST_DATA: str
    Path to `.jsonl` file containing the testing examples.
LABEL_MAP: str
    Path to label map (`.json`). The label map is a mapping from class label (e.g. `family_accession`) to numeric index.
LABEL_WEIGHTS: Optional[str]
    Path to label weights (`.json`). A json object with a single property (`weights`), where the property is an array of
    floats of length == number of labels. The order is significant, where the label map index and the position in the
    array need to correspond for any given label. The weights are then used in the loss to re-weigh the terms.
TOKEN_MAP: str
    Path to token map (`.json`). The token map is a mapping from sequence token (e.g. amino acid letters) to numeric
    representation. This can either be a single integer (i.e. index) or an n-length array.
SAVE_PATH: str
    Path to save directory.
SOURCE_COLUMN: Optional[str]
    Column name of the model inputs (default `sequence`).
TARGET_COLUMN: Optional[str]
    Column name of the model targets / labels (default `family_accession`).
SOURCE_TYPE: Optional[str]
    The tensor type for the model input; one of: `FLOAT`, `LONG`. (default "FLOAT")
TARGET_TYPE: Optional[str]
    The tensor type for the model targets; one of: `FLOAT`, `LONG`. (default `LONG`)
WANDB_PROJECT: str
    The wandb project identifier for experiment tracking.
WANDB_ENTITY: str
    The wandb entity to use for experiment tracking.
WANDB_API_KEY: str
    The wandb API key to use for experiment tracking.
"""

import re
from pathlib import Path

_repo_root = Path(__file__).parent.parent
try:
    changelog = (_repo_root / 'CHANGELOG.md').read_text()
    __version__, *_ = re.findall(r"\[([0-9.]+)]", changelog)
except (FileNotFoundError, ValueError):
    __version__ = '0.1.0'
