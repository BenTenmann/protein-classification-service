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
DATA_DIR: str
    Path to data directory.
MODEL_DIR: str
    Path to directory for saving the model weights.
RESUME_FROM: Optional[str]
    An optional path to model weights from which to resume the model training.
WANDB_PROJECT_NAME: str
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
