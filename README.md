# protein-classification-service

[![CircleCI](https://circleci.com/gh/BenTenmann/protein-classification-service/tree/main.svg?style=shield&circle-token=3b42235dd8a2f18865d981432d09730121915ec1)](https://circleci.com/gh/BenTenmann/protein-classification-service/tree/main)
[![codecov](https://codecov.io/gh/BenTenmann/protein-classification-service/branch/main/graph/badge.svg?token=XJYMGM5ZVK)](https://codecov.io/gh/BenTenmann/protein-classification-service)

This service takes an unaligned protein sequence as a query and returns a set of potential protein families.

## About

A protein family is a group of proteins which share function and evolutionary origin. These similarities are reflected in their sequence similarity, i.e. their conservation in primary structure (amino acid sequence).

## Running the service

This service assumes `docker` to be installed. To run this service, you first have to build the image:

```bash
IMAGE=$(dirname ${PWD})
TAG=$(cat .tag)
docker build -t ${IMAGE}:${TAG} .
```

Then run the image using:

```bash
docker run --rm --name ${IMAGE} -p 0.0.0.0:7687:9000/tcp ${IMAGE}:${TAG}
```
 
This will start the Seldon microservice. You can now send post requests to the model to receive a classification, e.g.:

```bash
curl -X POST localhost:7687/api/v1.0/predictions \
     -H 'Content-Type: application/json' \
     -d '{"sequence": "EIKKMISEIDKDGSGTIDFEEFLTMMTA"}'
```
