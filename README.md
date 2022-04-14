# protein-classification-development

[![CircleCI](https://circleci.com/gh/BenTenmann/protein-classification-service/tree/main.svg?style=shield&circle-token=3b42235dd8a2f18865d981432d09730121915ec1)](https://circleci.com/gh/BenTenmann/protein-classification-service/tree/main)
[![codecov](https://codecov.io/gh/BenTenmann/protein-classification-service/branch/main/graph/badge.svg?token=XJYMGM5ZVK)](https://codecov.io/gh/BenTenmann/protein-classification-service)

This is the development branch, containing the model training for the classification service.

## About

A protein family is a group of proteins which share function and evolutionary origin. These similarities are reflected
in their sequence similarity, i.e. their conservation in primary structure (amino acid sequence).

## Running the training

To run the model training, run the following:

```bash
WANDB_API_KEY=$YOUR_WANDB_API_KEY ./run-protein-classification-training.sh
```

## Running the tests

To run the unit tests, create a local Python3.9 environment and run the following:
```bash
pip install -r requirements-dev.txt
python3 -m pytest -v tests --cov=protein_classification
```


## Reading the report

A link to the pre-compiled HTML report detailing the experiments can be found in `${PWD}/notebooks/README.md`.
