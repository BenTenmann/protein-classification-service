#!/bin/bash

set -e

for prog in "wget" "make"
do
  if ! command -v ${prog} &> /dev/null; then
    echo "error: ${prog} not found"
    exit 127
  fi
done

URL=http://eddylab.org/software/hmmer/hmmer.tar.gz
TARFILE=${URL#*hmmer/}
wget ${URL}
tar -zxf ${TARFILE}

cd hmmer* || exit
./configure --prefix . && make && make install

# we will be using hmmer as a baseline (phmmer) and as a method for oversampling (hmmbuild)
