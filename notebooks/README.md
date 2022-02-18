# report

The pre-compiled HTML of the report can be downloaded [here](https://drive.google.com/uc?export=download&id=1R1RgzynOvY3YTfTbGQNrr3TxyMWexNF5). Alternatively, it can be re-run after setting up the environment:

```bash

export NOTEBOOK_ROOT=$(realpath ..)

DATA_DIR=${NOTEBOOK_ROOT}/data

DATA_URL="https://drive.google.com/uc?id=1Lmco9VmVTwjDUvJ7ebt3dgvs6ChtujcU"
TARFILE=${DATA_DIR}/data.tar.gz
gdown ${DATA_URL} --output ${TARFILE}

tar -xvf ${TARFILE} -C ${DATA_DIR} --strip-components 1
R -s -e 'install.packages(c("tidyverse", "rjson"))'
```

Here, it is assumed that you have `R` and `Rstudio` (for report compilation) installed. 
