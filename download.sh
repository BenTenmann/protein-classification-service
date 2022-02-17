#!/bin/bash

pip install wandb srsly==2.4.2

gcloud auth login
gsutil cp gs://pfam-seed-2/dump-2.tar.gz .
