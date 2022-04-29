"""
This script reformats the pfam-seed random split dataset downloaded from kaggle [1] from csv format into jsonl format.
The original data has each data split as a subdirectory of multiple csv files, hence we make them into one jsonl file
per split for ease of use. Each csv is made into a jsonl, which can then easily be concatenated in the shell.

This script is used by `$REPO_ROOT/scripts/reformat-pfam-dataset.sh`.

References
----------
[1] PFAM seed random split: https://www.kaggle.com/datasets/googleai/pfam-seed-random-split
"""

import logging
import os
from pathlib import Path

import pandas as pd
import srsly

logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(asctime)s.%(msecs)03d - %(levelname)s - %(module)s.%(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

INPUT_DIR = os.environ.get('INPUT_DIR')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR')


def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)

    logging.debug(f'input path: {input_path}')
    logging.debug(f'output path: {output_path}')

    subdir = output_path / input_path.name
    subdir.mkdir(parents=True, exist_ok=True)
    for file in input_path.iterdir():
        df = pd.read_csv(file)
        logging.debug(f'processing file: {file}')

        df['split'] = input_path.name
        df['filename'] = file.name

        out = df.to_dict(orient='records')
        srsly.write_jsonl(subdir / f'{file.stem}.jsonl', out)


if __name__ == '__main__':
    main()
