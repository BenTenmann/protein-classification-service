# report

The pre-compiled HTML of the report can be downloaded [here](). Alternatively, it can be re-run after setting up the environment:

```bash
export NOTEBOOK_ROOT=$(realpath ..)

DATA_DIR=${NOTEBOOK_ROOT}/data/random_split
mkdir -p ${DATA_DIR}

TARFILE=${DATA_DIR}/split.tar.gz
gdown ${DATA_URL} --output ${TARFILE}

tar -xvf ${TARFILE} -C ${DATA_DIR}
R -s -e 'install.packages(c("tidyverse", "rjson"))'
```

Here, it is assumed that you have `R` and `Rstudio` (for report compilation) installed. 