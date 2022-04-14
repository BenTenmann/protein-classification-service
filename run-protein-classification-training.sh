#!/bin/bash

set -e

if ! command -v python3 &> /dev/null; then
  echo "error: python3 not found"
  exit 127
fi

PROJECT=protein-classification
DATA_DIR="${PWD}"/data/dump-10k

export CONFIG_MAP="${PWD}"/manifest/language-model-classifier.yml
export TRAIN_DATA="${DATA_DIR}"/train-resampled.jsonl
export DEV_DATA="${DATA_DIR}"/dev.jsonl
export TEST_DATA="${DATA_DIR}"/test.jsonl
export LABEL_MAP="${DATA_DIR}"/label-map.json
export TOKEN_MAP="${DATA_DIR}"/kd-token-map.json
export SAVE_PATH="${PWD}"/model-weights

if [[ $2 == "weight" ]]; then
  export LABEL_WEIGHTS="${DATA_DIR}"/label-weights.json
fi

export SOURCE_COLUMN="sequence"
export TARGET_COLUMN="family_accession"
export SOURCE_TYPE="LONG"
export TARGET_TYPE="LONG"

export WANDB_ENTITY=bentenmann
export WANDB_PROJECT="${PROJECT}-development"

if [[ -z ${WANDB_API_KEY} ]]; then
  echo "WARNING: api key not set"
fi

python3 -m "${PROJECT//-/_}"
