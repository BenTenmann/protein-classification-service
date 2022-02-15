import os
from pathlib import Path

import pandas as pd
import srsly
from imblearn.under_sampling import RandomUnderSampler

INPUT = os.environ.get('INPUT')
OUTPUT = os.environ.get('OUTPUT')

CLASS_COLUMN = os.environ.get('CLASS_COLUMN', 'family_accession')


def main():
    input_path = Path(INPUT)
    output_path = Path(OUTPUT)

    lines = srsly.read_jsonl(input_path)
    df = pd.DataFrame(lines)

    sampler = RandomUnderSampler(random_state=42)
    df, _ = sampler.fit_resample(df, df[CLASS_COLUMN])

    srsly.write_jsonl(output_path / f'{input_path.stem}-resampled.jsonl', df.to_dict(orient='records'))


if __name__ == '__main__':
    main()
