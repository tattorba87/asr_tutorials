#!/usr/bin/env python3

"""
This file computes fbank features of the LibriSpeech dataset.
It looks for manifests in the directory data/manifests.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch

from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor, str2bool

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-dir",
        type=str,
        default="data/manifests",
        help="""Path to the source directory containing the manifests.""",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/fbank",
        help="""Path to the output directory for the fbank features.""",
    )
    parser.add_argument(
        "--perturb-speed",
        type=str2bool,
        default=True,
        help="""Perturb speed with factor 0.9 and 1.1 on train subset.""",
    )
    parser.add_argument(
        "--num-mel-bins",
        type=int,
        default=80,
        help="""Number of mel bins for the fbank features.""",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=15,
        help="""Number of parallel jobs to use for feature extraction.""",
    )

    return parser.parse_args()


def compute_fbank_audiomnist(
    src_dir: Path,
    output_dir: Path,
    perturb_speed: Optional[bool] = True,
    num_mel_bins: int = 80,
    num_jobs: int = 15,
):
    src_dir = Path(src_dir)
    output_dir = Path(output_dir)
    num_jobs = min(num_jobs, os.cpu_count())
    num_mel_bins = int(num_mel_bins)

    prefix = "audio_mnist"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=None,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None
    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:  # Initialize the executor only once.
        for partition, m in manifests.items():
            cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
            if (output_dir / cuts_filename).is_file():
                logging.info(f"{partition} already exists - skipping.")
                continue
            logging.info(f"Processing {partition}")
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )

            if "train" in partition:
                if perturb_speed:
                    logging.info("Performing speed perturbation.")
                    cut_set = (
                        cut_set
                        + cut_set.perturb_speed(0.9)
                        + cut_set.perturb_speed(1.1)
                    )

            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                # when an executor is specified, make more partitions
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )
            cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":

    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    compute_fbank_audiomnist(
        src_dir=args.src_dir,
        output_dir=args.output_dir,
        perturb_speed=args.perturb_speed,
        num_mel_bins=args.num_mel_bins,
        num_jobs=args.num_jobs,
    )
    logging.info("Done computing fbank features.")
