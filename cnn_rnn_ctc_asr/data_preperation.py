#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script is used to prepare data for training a CNN-RNN-CTC model for 
automatic speech recognition (ASR).

The data it prepares is AudioMNIST, which is a dataset of spoken digits - 
a great starting point for ASR tasks with a manageable size.

It includes functions to load audio files, extract features, and prepare the dataset.
"""

import argparse
import logging
from pathlib import Path

from lhotse.recipes.audio_mnist import prepare_audio_mnist
from lhotse import RecordingSet, SupervisionSet, CutSet


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def prepare_audiomnist(
    data_dir, 
    output_dir
):
    """
    Prepares the AudioMNIST dataset for ASR by loading audio files and
    extracting features.

    Args:
        data_dir (str): Directory where the AudioMNIST dataset is stored.
        (The repo root directory)
        output_dir (str): Directory where the prepared dataset will be saved.
    """

    # if supervisions.to_file(output_dir / "audio_mnist_supervisions.jsonl.gz")
    # recordings.to_file(output_dir / "audio_mnist_recordings.jsonl.gz")
    # exists just load them from disk

    supervision_file = Path(output_dir) / "audio_mnist_supervisions.jsonl.gz"
    recordings_file = Path(output_dir) / "audio_mnist_recordings.jsonl.gz"

    if recordings_file.is_file() and supervision_file.is_file():
        logger.info("Loading existing AudioMNIST dataset...")
        recordings = RecordingSet.from_file(recordings_file)
        supervisions = SupervisionSet.from_file(supervision_file)
        manifests = {
            "recordings": recordings,
            "supervisions": supervisions
        }
        logger.info("AudioMNIST dataset loaded successfully.")
    else:
        logger.info("Preparing AudioMNIST dataset...")
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

    # Lets take this a bit further and convert it into a CutSet and then split
    cut_set = CutSet.from_manifests(
        recordings=manifests["recordings"],
        supervisions=manifests["supervisions"]
    )

    n_cuts_sample = len(cut_set)
    first = int(n_cuts_sample * 0.8)
    last = n_cuts_sample - first
    train_cuts = cut_set.subset(first=first)
    test_cuts = cut_set.subset(last=last)

    train_cuts.to_file(args.output_dir / "audio-mnist_cuts_train.jsonl.gz")
    test_cuts.to_file(args.output_dir / "audio-mnist_cuts_test.jsonl.gz")

    logger.info("AudioMNIST dataset prepared and saved to %s", args.output_dir)
