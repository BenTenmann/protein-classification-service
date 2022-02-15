# protein-classification-service

This service takes an unaligned protein sequence as a query and returns a set of potential protein families.

## About

A protein family is a group of proteins which share function and evolutionary origin. These similarities are reflected in their sequence similarity, i.e. their conservation in primary structure (amino acid sequence).

## Running the service

This service assumes `docker` to be installed. To run this service, you first have to build the image:

```bash
IMAGE=$(basename ${PWD})
TAG=$(cat .tag)
docker build -t ${IMAGE}:${TAG} .
```

Then run the image using:

```bash
docker run -d -p 80:80 ${IMAGE}:${TAG}
```
