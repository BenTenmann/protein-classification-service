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

INPUT = os.environ.get('INPUT')
OUTPUT = os.environ.get('OUTPUT')


def main():
    input_path = Path(INPUT)
    output_path = Path(OUTPUT)

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
