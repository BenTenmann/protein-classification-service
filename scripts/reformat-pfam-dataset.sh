#!/bin/bash

# set the `DATA_ROOT_DIR` environment variable to point at the directory which contains the pfam-seed random split
# subdirectories. By default the script will assume that the untarred folders have been placed in the $REPO_ROOT/data
# directory and will use the `$REPO_ROOT/data/random_split/random_split` subdir.

set -e

REPO_ROOT="$(dirname "$0")"/..

PYTHONENV="$REPO_ROOT"/venv
if [[ -d $PYTHONENV ]]; then
  echo error: no python3 virtualenv detected.
  exit 1
fi

export PATH="$PYTHONENV"/bin:"$PATH"
SCRIPT="$REPO_ROOT"/scripts/reformat_pfam.py

export OUTPUT_DIR="${OUTPUT_DIR:-"$REPO_ROOT"/data}"
for subdir in "${DATA_ROOT_DIR:-"$REPO_ROOT"/data/random_split/random_split}"/*
do
    export INPUT_DIR="$subdir"
    python3 "$SCRIPT"

    cat "$subdir"/*.jsonl > "$OUTPUT_DIR/$(basename "$subdir")".jsonl
    rm "$subdir"/*.jsonl
done
