#!/bin/bash

set -e

DIR="$1"
OUT="$2"

if [[ -z ${DIR} ]]; then
    DIR="${PWD}/data"
fi

if [[ -z ${OUT} ]]; then
    OUT="${DIR}"
fi

PYPATH="${PWD}"/venv/bin
SCRIPT="${PWD}"/scripts/reformat_pfam.py

export OUTPUT="${OUT}"
for subdir in "${DIR}"/*
do
    export INPUT="${subdir}"
    "${PYPATH}"/python3 "${SCRIPT}"

    cat "${subdir}"/*.jsonl > "${OUT}"/"$(basename "${subdir}")".jsonl
    rm "${subdir}"/*.jsonl
done
