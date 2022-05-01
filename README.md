# protein-classification-development

[![CircleCI](https://circleci.com/gh/BenTenmann/protein-classification-service/tree/main.svg?style=shield&circle-token=3b42235dd8a2f18865d981432d09730121915ec1)](https://circleci.com/gh/BenTenmann/protein-classification-service/tree/main)
[![codecov](https://codecov.io/gh/BenTenmann/protein-classification-service/branch/main/graph/badge.svg?token=XJYMGM5ZVK)](https://codecov.io/gh/BenTenmann/protein-classification-service)

This is the development branch, containing the model training for the classification service.

## About

A protein family is a group of proteins which share function and evolutionary origin. These similarities are reflected
in their sequence similarity, i.e. their conservation in primary structure (amino acid sequence).

## Running the training

To run the model training, you will first have to download the preprocessed data and install the relevant requirements:

```bash
# download and untar preprocessed data
FILENAME=pfam-seed-random-split.tar.gz
gsutil cp gs://`basename $PWD`/$FILENAME data/ && \
  tar -xvf data/$FILENAME -C data/
  
# install the relevant requirements
pip install -r requirements${COLAB_GPU:+'-colab'}.txt
```

Once the data is downloaded and the requirements are installed, the training scripts can be run like so:

```bash
mkdir ${MODEL_DIR:=models}

DATA_DIR=data/${FILENAME%.tar.gz} \
  MODEL_DIR=$MODEL_DIR \
  WANDB_API_KEY=$YOUR_WANDB_API_KEY \
  python -m protein_classification
```

## Running the tests

To run the unit tests, create a local Python3.9 environment and run the following:
```bash
pip install -r requirements-dev.txt
python3 -m pytest -v tests --cov=protein_classification
```


## Reading the report

A link to the pre-compiled HTML report detailing the experiments can be found in `${PWD}/notebooks/README.md`.
