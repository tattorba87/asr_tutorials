#!/usr/bin/env bash

set -euo pipefail

python ./compute_fbank.py \
  --src-dir ~/data/AudioMNIST_processed \
  --output-dir ~/data/AudioMNIST_processed/fbank \
  --perturb-speed true \
  --num-jobs 64 \
  --num-mel-bins 80

echo "Fbank features computed and saved to ~/data/AudioMNIST_processed/fbank"

