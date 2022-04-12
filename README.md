# protein-classification-service

[![CircleCI](https://circleci.com/gh/BenTenmann/protein-classification-service/tree/main.svg?style=shield&circle-token=3b42235dd8a2f18865d981432d09730121915ec1)](https://circleci.com/gh/BenTenmann/protein-classification-service/tree/main)
[![codecov](https://codecov.io/gh/BenTenmann/protein-classification-service/branch/main/graph/badge.svg?token=XJYMGM5ZVK)](https://codecov.io/gh/BenTenmann/protein-classification-service)

This service takes an unaligned protein sequence as a query and returns a potential protein family from the 1000 most abundant families in Pfam.

## About

A protein family is a group of proteins which share function and evolutionary origin. These similarities are reflected
in their sequence similarity, i.e. their conservation in primary structure (amino acid sequence).

## Running the service

Assuming `helm` is installed (plus a Kubernetes cluster being available), the service can be instantiated using:

```bash
helm install `basename $PWD` ./helm
```
 
This will start the Seldon microservice. You can now send post requests to the model to receive a classification, e.g.:

```bash
# port forward service
kubectl port-forward svc/`basename $PWD` ${PORT:=7687} &

curl -X POST localhost:${PORT}/api/v1.0/predictions \
     -H 'Content-Type: application/json' \
     -d '{"sequence": "EIKKMISEIDKDGSGTIDFEEFLTMMTA"}'
```

## Running the tests

To run the unit tests, create a local Python3.9 environment and run the following:

```bash
pip install -r requirements-dev.txt
python3 -m pytest -v tests --cov=protein_classification
```

