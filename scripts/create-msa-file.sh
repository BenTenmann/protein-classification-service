#!/bin/bash

if ! command -v jq &> /dev/null; then
  echo "error: jq not found"
  exit 127
fi

DIR="$1"
if [[ -z ${DIR} ]]; then
  DIR="${PWD}"/data/random_split
fi

cat "${DIR}"/*.jsonl | jq -R -r '. | fromjson | .aligned_sequence'