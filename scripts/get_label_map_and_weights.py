import os
from pathlib import Path

import numpy as np
import pandas as pd
import srsly

INPUT = os.environ.get('INPUT')
OUTPUT = os.environ.get('OUTPUT')
CLASS_COLUMN = os.environ.get('CLASS_COLUMN', 'family_accession')
BETA = os.environ.get('BETA', 0.99)


def effective_number_of_samples(class_counts: np.ndarray, beta: float) -> np.ndarray:
    ens = (1 - beta ** class_counts) / (1 - beta)
    out = 1 / ens
    return out


def main():
    input_path = Path(INPUT)
    output_path = Path(OUTPUT)

    files = [srsly.read_jsonl(file) for file in input_path.glob('*.jsonl')]
    df = pd.concat([pd.DataFrame(file) for file in files])

    counts = df.groupby(CLASS_COLUMN).size()
    beta = float(BETA)
    ens = effective_number_of_samples(counts.to_numpy(), beta)

    label_weights = {'weights': ens.tolist()}
    num_classes = len(counts)
    label_map = dict(zip(counts.index, range(num_classes)))

    srsly.write_json(output_path / 'label-map.json', label_map)
    srsly.write_json(output_path / 'label-weights.json', label_weights)


if __name__ == '__main__':
    main()
