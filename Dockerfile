FROM python:3.9-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY service-dependencies .
COPY protein_classification protein_classification
COPY protein_classification_service .

EXPOSE 5000
EXPOSE 9000

CMD exec seldon-core-microservice ProteinClassificationService --service-type MODEL --persistence 0
