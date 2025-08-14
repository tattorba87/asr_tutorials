#!/usr/bin/env bash

set -euo pipefail

source ./path.sh

python ./compute_fbank.py \
  --src-dir ~/data/AudioMNIST_processed \
  --output-dir ~/data/AudioMNIST_processed/fbank \
  --partition train \
  --perturb-speed true \
  --num-jobs 64 \
  --num-mel-bins 80

python ./compute_fbank.py \
  --src-dir ~/data/AudioMNIST_processed \
  --output-dir ~/data/AudioMNIST_processed/fbank \
  --partition test \
  --perturb-speed true \
  --num-jobs 64 \
  --num-mel-bins 80

echo "Fbank features computed and saved to ~/data/AudioMNIST_processed/fbank"
