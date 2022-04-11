"""
protein_classification
=====================

This package contains the model and tokenizer objects for the protein classification service.
"""

import re
from pathlib import Path

_repo_root = Path(__file__).parent.parent
try:
    changelog = (_repo_root / 'CHANGELOG.md').read_text()
    __version__, *_ = re.findall(r"\[([0-9.]+)]", changelog)
except (FileNotFoundError, ValueError):
    __version__ = '0.1.0'
