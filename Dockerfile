FROM python:3.9-slim AS base
FROM base AS builder

COPY requirements.txt .
RUN pip install -r requirements.txt

ARG CONFIG_NAME=mlp
ARG LOGIT_NAME=1k-fam
ARG TOKEN_NAME=kidera-factors

ENV CONFIG_MAP=config-map/${CONFIG_NAME}.yml
ENV LOGIT_MAP=logit-map/${LOGIT_NAME}.json
ENV TOKEN_MAP=token-map/${TOKEN_NAME}.json
ENV MODEL_WEIGHTS=weights/state-dict.bin

ARG MODEL_WEIGHTS_URL=https://drive.google.com/uc?id=1OPEltgXxzesQQ84VnjVMbUd_r1aNDhRp

COPY service-dependencies .
RUN mkdir $(dirname ${MODEL_WEIGHTS}) && gdown ${MODEL_WEIGHTS_URL} --output ${MODEL_WEIGHTS}

COPY protein_classification protein_classification
COPY protein_classification_service .

EXPOSE 5000
EXPOSE 9000

CMD exec seldon-core-microservice ProteinClassificationService --service-type MODEL --persistence 0
