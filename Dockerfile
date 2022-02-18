FROM python:3.9-slim AS base
FROM base AS builder

COPY requirements.txt .
RUN pip install -r requirements.txt

ARG CONFIG_NAME=mlp
ARG LOGIT_NAME=1k-fam
ARG TOKEN_NAME=kidera-factors
ARG MODEL_WEIGHTS_DIR=weights

ENV CONFIG_MAP=config-map/${CONFIG_NAME}.yml
ENV LOGIT_MAP=logit-map/${LOGIT_NAME}.json
ENV TOKEN_MAP=token-map/${TOKEN_NAME}.json
ENV MODEL_WEIGHTS=${MODEL_WEIGHTS_DIR}/kidera-factors-refactored.bin

ARG MODEL_WEIGHTS_URL=https://drive.google.com/uc?id=1PAp5bcgE3GZdAxKZb0JMNfEvq0rx81MB

COPY service-dependencies .
RUN mkdir ${MODEL_WEIGHTS_DIR} && \
    gdown ${MODEL_WEIGHTS_URL} && \
    tar -xvf model.tar.gz -C ${MODEL_WEIGHTS_DIR}

COPY protein_classification protein_classification
COPY protein_classification_service .

EXPOSE 5000
EXPOSE 9000

CMD exec seldon-core-microservice ProteinClassificationService --service-type MODEL --persistence 0
