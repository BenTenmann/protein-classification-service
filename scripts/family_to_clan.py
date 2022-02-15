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

DATA_DIR = os.environ.get('DATA_DIR')
CLAN_FILE = os.environ.get('CLAN_FILE')
OUTPUT = os.environ.get('OUTPUT')


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
    clan_file = Path(CLAN_FILE)
    output_dir = Path(OUTPUT)

    data = pd.concat([read_dataframe(file) for file in data_dir.glob('*.jsonl')])
    clan_db = pd.read_table(clan_file, sep='\t')

    data['unversioned_accession'] = data.family_accession.str.replace(r"\.[0-9]{1,2}", "")
    data.set_index('unversioned_accession', inplace=True)
    clan_db.set_index('pfamA_acc', inplace=True)

    n_fam = data.family_accession.nunique()
    data = data.join(clan_db[['clan_acc']]).reset_index(drop=True)
    logging.info(f'number of sequences dropped: {data.clan_acc.isna().sum()}')
    data = data.dropna(subset=['clan_acc'])
    logging.info(f'number of families dropped: {n_fam - data.family_accession.nunique()}')
    data.groupby('split').apply(to_jsonl, directory=output_dir)


if __name__ == '__main__':
    main()
