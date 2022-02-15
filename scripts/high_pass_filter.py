import os
from pathlib import Path

import pandas as pd
import srsly

DATA_DIR = os.environ.get('DATA_DIR')
OUTPUT = os.environ.get('OUTPUT')

CLASS_COLUMN = os.environ.get('CLASS_COLUMN', 'family_accession')
NUMBER_THRESHOLD = os.environ.get('NUMBER_THRESHOLD', 25)


def read_dataframe(path: Path) -> pd.DataFrame:
    lines = srsly.read_jsonl(path)
    df = pd.DataFrame(lines)
    return df


def to_jsonl(df: pd.DataFrame, directory: Path) -> None:
    split = df.split.iloc[0]
    out = df.to_dict(orient='records')
    srsly.write_jsonl(directory / f'{split}.jsonl', out)


def main():
    data_dir = Path(DATA_DIR)
    output_dir = Path(OUTPUT)

    df = pd.concat([read_dataframe(file) for file in data_dir.glob('*.jsonl')])
    counts = df[CLASS_COLUMN].value_counts()

    threshold = int(NUMBER_THRESHOLD)
    to_keep = counts.loc[(counts >= threshold)].index

    df = df.loc[df[CLASS_COLUMN].isin(to_keep)]
    df.groupby('split').apply(to_jsonl, directory=output_dir)


if __name__ == '__main__':
    main()
