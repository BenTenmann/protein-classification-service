"""
This script takes the jsonl formatted pfam seed random data splits and saves them as a set of jax numpy arrays in a
nested subdir structure (one dir per split). The script requires a number of environment variables.

Parameters
----------
LABEL_MAP: Path
    Filepath to the label map, which is a JSON file containing a key for each string label, with a integer index as a
    value. It will be used to map the string labels to their numeric index representations.
TOKEN_MAP: Path
    Filepath to the token map, which is a JSON file containing a key for each amino acid and a corresponding integer
    index value. The integer index needs to start at 1, as 0 is reserved as the pad token. The token map will be used
    to tokenize the protein sequences into integer vectors.
DATA_DIR: Path
    The directory containing the train, dev and test JSONL files.
OUTPUT_DIR: Path
    The directory into which the output will be stored.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd
try:
    # in true `explosion` fashion, `srsly` can break easily with new updates; so we guard against that. However, srsly
    # is still faster than the methods below
    from srsly import read_json, read_jsonl
except (ModuleNotFoundError, ImportError):
    import json
    from functools import wraps

    logging.warning('`srsly` import failed. Using slow JSON read methods.')

    def ensure_path(fn):
        @wraps(fn)
        def _type_safe(filename: str or Path):
            if isinstance(filename, str):
                filename = Path(filename)
            out = fn(filename)
            return out
        return _type_safe

    @ensure_path
    def read_json(filename: Path) -> dict:
        txt = filename.read_text()
        obj = json.loads(txt)
        return obj

    @ensure_path
    def read_jsonl(filename: Path):
        for line in filename.open(mode='r'):
            obj = json.loads(line)
            yield obj

logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(asctime)s.%(msecs)03d - %(levelname)s - %(module)s.%(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

LABEL_MAP = os.environ.get('LABEL_MAP')
TOKEN_MAP = os.environ.get('TOKEN_MAP')
DATA_DIR = os.environ.get('DATA_DIR')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR')

DATA_FILES = [
    ('train.jsonl', 'train'),
    ('dev.jsonl', 'dev'),
    ('test.jsonl', 'test')
]
SOURCE_COLUMN = 'sequence'
TARGET_COLUMN = 'family_accession'
MAX_SEQUENCE_LENGTH = 256
PAD_TOKEN = 0
PAD_CHAR = '_'


def main():
    label_map = read_json(LABEL_MAP)
    token_map = read_json(TOKEN_MAP)
    token_map.setdefault(PAD_CHAR, PAD_TOKEN)
    token_map = defaultdict(lambda: PAD_TOKEN, token_map)

    def normalize_sequence(sequence: str) -> str:
        sequence_length = len(sequence)
        n_pad_tokens = max(MAX_SEQUENCE_LENGTH - sequence_length, 0)
        sequence = sequence[:MAX_SEQUENCE_LENGTH] + (PAD_CHAR * n_pad_tokens)
        return sequence

    def tokenize_sequence(sequence: str) -> Sequence[int]:
        tokenized = [token_map[char] for char in sequence]
        return tokenized

    def process_dataset(path: str or Path) -> tuple:
        lines = read_jsonl(path)
        df = pd.DataFrame(lines)
        normalized_sequences = map(normalize_sequence, df[SOURCE_COLUMN])
        tokenized_sequences = map(tokenize_sequence, normalized_sequences)
        label_indices = map(label_map.get, df[TARGET_COLUMN])

        src = np.array(list(tokenized_sequences), dtype=np.int8)
        tgt = np.array(list(label_indices), dtype=np.int16)
        return src, tgt

    input_dir = Path(DATA_DIR)
    output_dir = Path(OUTPUT_DIR)
    for filename, split in DATA_FILES:
        logging.info(f'processing: {filename}')
        src_, tgt_ = process_dataset(input_dir / filename)
        source = jnp.array(src_, dtype=jnp.int8)
        target = jnp.array(tgt_, dtype=jnp.int16)
        split_dir = (output_dir / split)
        split_dir.mkdir(parents=True, exist_ok=True)
        jnp.save(str(split_dir / 'source'), source)
        jnp.save(str(split_dir / 'target'), target)


if __name__ == '__main__':
    main()
