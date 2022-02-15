import os
import uuid
from pathlib import Path

import pandas as pd
import srsly
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

INPUT = os.environ.get('INPUT')
OUTPUT = os.environ.get('OUTPUT')

CLASS_COLUMN = os.environ.get('CLASS_COLUMN', 'family_accession')
SOURCE_COLUMN = os.environ.get('SOURCE_COLUMN', 'aligned_sequence')


def read_dataframe(path: Path) -> pd.DataFrame:
    lines = srsly.read_jsonl(path)
    df = pd.DataFrame(lines)
    return df


def create_msa(family: pd.DataFrame) -> AlignIO.MultipleSeqAlignment:
    records = [
        SeqRecord(Seq(sequence), id=str(uuid.uuid4())) for sequence in family[SOURCE_COLUMN]
    ]
    out = AlignIO.MultipleSeqAlignment(records)
    return out


def main():
    input_path = Path(INPUT)
    output_path = Path(OUTPUT)

    df = read_dataframe(input_path)
    multiple_seq_alignments = df.groupby(CLASS_COLUMN).apply(create_msa)

    for idx, msa in multiple_seq_alignments.iteritems():
        AlignIO.write(msa, output_path / f'{idx}.sto', 'stockholm')


if __name__ == '__main__':
    main()
