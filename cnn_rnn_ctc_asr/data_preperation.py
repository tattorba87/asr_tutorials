#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script is used to prepare data for training a CNN-RNN-CTC model for 
automatic speech recognition (ASR).

The data it prepares is AudioMNIST, which is a dataset of spoken digits - 
a great starting point for ASR tasks with a manageable size.

It includes functions to load audio files, extract features, and prepare the dataset.
"""


import logging
from lhotse.recipes.audio_mnist import prepare_audio_mnist
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def prepare_audiomnist(data_dir, output_dir):
    """
    Prepares the AudioMNIST dataset for ASR by loading audio files and extracting features.
    
    Args:
        data_dir (str): Directory where the AudioMNIST dataset is stored. (The repo root directory)
        output_dir (str): Directory where the prepared dataset will be saved.
    """

    # Load the AudioMNIST dataset
    manifests = prepare_audio_mnist(
        corpus_dir=data_dir,
        output_dir=output_dir,
    )

    return manifests

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare AudioMNIST dataset for ASR")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory where the AudioMNIST root repo is stored",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the prepared dataset will be saved",
    )

    args = parser.parse_args()

    manifests = prepare_audiomnist(args.data_dir, args.output_dir)
