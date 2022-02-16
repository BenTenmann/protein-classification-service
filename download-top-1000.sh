#!/bin/bash

for file in requirements*.txt
do
    pip install -r ${file}
done

gdown "https://drive.google.com/uc?id=1cfYrg9myxh5w_YRdc48KPxR35_HSmKmv"
tar -xvf dump.tar.gz
