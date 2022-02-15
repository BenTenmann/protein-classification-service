import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import srsly
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# ----- Logging Setup ------------------------------------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(asctime)s.%(msecs)03d - %(levelname)s - %(module)s.%(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ----- Environment Variables ---------------------------------------------------------------------------------------- #
INPUT = os.environ.get('INPUT')
OUTPUT = os.environ.get('OUTPUT')

KMER = os.environ.get('KMER', 3)
N_COMPONENTS = os.environ.get('N_COMPONENTS', 16)


# ----- Helpers ------------------------------------------------------------------------------------------------------ #
def time_it(fn):
    @wraps(fn)
    def _method(*args, **kwargs):
        logging.info(f'running: {fn.__name__}')
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        t1 = time.perf_counter()
        logging.info(f'time taken: {t1 - t0:.3f}s')

        return out
    return _method


@time_it
def read_dataframe(path: Path) -> pd.DataFrame:
    files = [srsly.read_jsonl(_p) for _p in path.glob('*.jsonl')]
    df = pd.concat([pd.DataFrame(file) for file in files])
    return df


@time_it
def extract_frequency_features(sequences: List[str] or pd.Series) -> sparse.csr_matrix:
    k = int(KMER)
    logging.debug(f'value of k: {k}')
    model = TfidfVectorizer(tokenizer=list, ngram_range=(k, k))

    out = model.fit_transform(sequences)
    return out


@time_it
def compute_principal_components(features: sparse.csr_matrix) -> np.ndarray:
    n_components = int(N_COMPONENTS)
    svd = TruncatedSVD(n_components)
    out = svd.fit_transform(features)

    logging.info(f'cumulative explained variance ratio of {n_components} components: '
                 f'{svd.explained_variance_ratio_[:n_components].sum():.3f}')
    return out


@time_it
def to_jsonl(df: pd.DataFrame, path: Path) -> None:
    split = df.split.iloc[0]
    out = df.to_dict(orient='records')
    srsly.write_jsonl(path / f'tf-idf-{split}.jsonl', out)


# ----- Script ------------------------------------------------------------------------------------------------------- #
def main():
    input_path = Path(INPUT)
    output_path = Path(OUTPUT)

    logging.debug(f'input path: {input_path}')
    logging.debug(f'output path: {output_path}')

    df = read_dataframe(input_path)

    features = extract_frequency_features(df.sequence)
    logging.info(f'TF-IDF array shape: {features.shape}')

    components = compute_principal_components(features)

    n_components = int(N_COMPONENTS)
    df['tf_idf'] = components[:, :n_components].tolist()
    df.groupby('split').apply(to_jsonl, path=output_path)


if __name__ == '__main__':
    main()
