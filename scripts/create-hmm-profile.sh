#!/bin/bash

if ! command -v hmmbuild &> /dev/null; then
  echo "error: hmmbuild not found"
  exit 127
fi

DIR="$1"
if [[ -z ${DIR} ]]; then
  DIR="${PWD}/data/msa"
fi

OUT="$2"
if [[ -z ${OUT} ]]; then
  OUT="${PWD}/data/hmm"
fi

for file in "${DIR}"/*
do
  name="$(basename "${file}")"
  hmmbuild "${OUT}/${name%.sto}".hmm "${file}" &> /dev/null
  echo "processed: ${name}"
done
