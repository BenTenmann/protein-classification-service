# protein-classification-service

[![CircleCI](https://circleci.com/gh/BenTenmann/protein-classification-service/tree/main.svg?style=shield&circle-token=3b42235dd8a2f18865d981432d09730121915ec1)](https://circleci.com/gh/BenTenmann/protein-classification-service/tree/main)
[![codecov](https://codecov.io/gh/BenTenmann/protein-classification-service/branch/main/graph/badge.svg?token=XJYMGM5ZVK)](https://codecov.io/gh/BenTenmann/protein-classification-service)

This service takes an unaligned protein domain sequence and returns the most likely protein family from the ~18,000
families in Pfam-32.0.

## About

A protein family is a group of proteins which share function and evolutionary origin. These similarities are reflected
in their sequence similarity, i.e. their conservation in primary structure (amino acid sequence).

The project implements a slimmed down version of the `ProtCNN` model proposed by Bileschi et al. [1] using Flax [2]. The
model was trained using the `pfam-seed-random-split` dataset, either available in raw form from
[Kaggle](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split) or in preprocessed format using:

```bash
FILENAME=pfam-seed-random-split.tar.gz
gsutil cp gs://protein-classification-service/$FILENAME data/ && \
    tar -xvf data/$FILENAME -C data/
```

The preprocessing scripts can be found on the `dev` branch of this repo (`scripts/reformat-pfam-dataset.sh` +
`raw_data_to_jax_arrays.py`). The Python script pads and tokenizes the unaligned protein sequences and casts the string
accession codes to class indexes. These arrays are then stored in `.npy` format for faster load times for training. The
preprocessed data available through Google Cloud (see above) also comes with the relevant token and label maps.

The model performance on train, dev and test spits is shown below:

| split | accuracy | macro avg recall | macro avg precision | macro avg F1 |
|-------|----------|------------------|---------------------|--------------|
| train | 0.970    | 0.814            | 0.850               | 0.825        |
| dev   | 0.950    | 0.870            | 0.878               | 0.862        |
| test  | 0.950    | 0.870            | 0.877               | 0.862        |

For details on model hyperparameters, please refer to `protein_classification/constants.py` on the `dev` branch of this
repository.

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
     -d '{"sequence": ["EIKKMISEIDKDGSGTIDFEEFLTMMTA"]}'
```

The request can be sent as a batch, where the JSON array of sequences would be passed through the model in one go. Keep
this in mind when running on GPU, as the service does not manage memory for you - i.e. it does not do mini-batches.
Hence, it can cause out-of-memory issues when sending too large requests.

A batched request:

```bash
curl -X POST localhost:${PORT}/api/v1.0/predictions \
     -H 'Content-Type: application/json' \
     -d '{"sequence": ["EIKKMISEIDKDGSGTIDFEEFLTMMTA", "IVQINEIFQVETDQFTQLLDA"]}'  # send 2 sequences for inference
```

## Running the tests

To run the unit tests, create a local Python3.9 environment and run the following:

```bash
pip install -r requirements-dev.txt
python3 -m pytest -v tests --cov=protein_classification
```

## References

1. Bileschi, M.L., Belanger, D., Bryant, D.H., Sanderson, T., Carter, B., Sculley, D., Bateman, A., DePristo, M.A. and
Colwell, L.J., 2022. Using deep learning to annotate the protein universe. Nature Biotechnology, pp.1-6.
2. Heek, J., Levskaya, A., Oliver, A., Ritter, M., Rondepierre, B., Steiner, A. and van Zee, M., Flax: A neural network
library and ecosystem for JAX, 2020. URL http://github. com/google/flax, 1.