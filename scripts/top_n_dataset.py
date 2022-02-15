import os
from pathlib import Path

import pandas as pd
import srsly

INPUT = os.environ.get('INPUT')
OUTPUT = os.environ.get('OUTPUT')

CLASS_COLUMN = os.environ.get('CLASS_COLUMN', 'family_accession')
TOP_N = os.environ.get('TOP_N', 1_000)


def to_jsonl(df: pd.DataFrame, directory: Path) -> None:
    split = df.split.iloc[0]
    out = df.to_dict(orient='records')
    srsly.write_jsonl(directory / f'{split}-top-{TOP_N}.jsonl', out)


def main():
    input_path = Path(INPUT)
    output_path = Path(OUTPUT)

    files = [srsly.read_jsonl(_p) for _p in input_path.glob('*.jsonl')]
    df = pd.concat([pd.DataFrame(file) for file in files])

    n = int(TOP_N)
    families = df[CLASS_COLUMN].value_counts().sort_values(ascending=False).index[:n]

    df = df.loc[df[CLASS_COLUMN].isin(families)]
    df.groupby('split').apply(to_jsonl, directory=output_path)


if __name__ == '__main__':
    main()
