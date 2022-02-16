#!/bin/bash

set -e

if ! command -v python3 &> /dev/null; then
  echo "error: python3 not found"
  exit 127
fi

PROJECT=protein-classification
PYPATH="${PWD}"/venv/bin
DATA_DIR="${PWD}"/data/random_split/top-n

export CONFIG_MAP="${PWD}"/manifest/mlp-one-hot.yml
export TRAIN_DATA="${DATA_DIR}"/train-top-1000-resampled.jsonl
export DEV_DATA="${DATA_DIR}"/dev-top-1000.jsonl
export TEST_DATA="${DATA_DIR}"/test-top-1000.jsonl
export LABEL_MAP="${DATA_DIR}"/label-map.json
export TOKEN_MAP="${PWD}"/data/oh-token-map.json
export SAVE_PATH="${PWD}"/model-weights

export SOURCE_COLUMN="sequence"
export TARGET_COLUMN="family_accession"
export SOURCE_TYPE="LONG"
export TARGET_TYPE="LONG"

export WANDB_ENTITY=bentenmann
export WANDB_PROJECT="${PROJECT}-mlp"

if [[ -z ${WANDB_API_KEY} ]]; then
  echo "WARNING: api key not set"
fi

"${PYPATH}"/python3 -m "${PROJECT//-/_}"
