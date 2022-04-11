#!/bin/bash

export TAG="$1"
yq -i e '.image.tag = env(TAG)' helm/values.yaml
yq -i e '.version = env(TAG)' helm/Chart.yaml
