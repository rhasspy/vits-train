#!/usr/bin/env python3
import argparse
import logging
import os
import random
import typing
from pathlib import Path

import torch
import torch.multiprocessing
from phonemes2ids import load_phoneme_ids
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from vits_train import setup_discriminator, setup_model
from vits_train.checkpoint import load_checkpoint
from vits_train.config import TrainingConfig
from vits_train.dataset import PhonemeIdsAndMelsDataset, UtteranceCollate, load_dataset
from vits_train.models import MultiPeriodDiscriminator, SynthesizerTrn
from vits_train.train import train

_LOGGER = logging.getLogger("vits_train")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="vits-train")
    parser.add_argument(
        "--output", required=True, help="Directory to store model artifacts"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        nargs=3,
        action="append",
        default=[],
        metavar=("dataset_name", "metadata_dir", "audio_dir"),
        help="Speaker id, phonemes CSV, and directory with audio files",
    )
    parser.add_argument(
        "--config", action="append", help="Path to JSON configuration file(s)"
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size (default: use config)"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Evaluation batch size (default: 16)",
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs to run (default: use config)"
    )
    parser.add_argument("--checkpoint", help="Path to restore checkpoint")
    parser.add_argument("--git-commit", help="Git commit to store in config")
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        default=1,
        help="Number of epochs between checkpoints",
    )
    parser.add_argument(
        "--no-restore-optimizers",
        action="store_true",
        help="Don't load the optimizer states from checkpoint",
    )
    parser.add_argument(
        "--no-restore-schedulers",
        action="store_true",
        help="Don't load the scheduler states from checkpoint",
    )
    parser.add_argument(
        "--cache",
        help="Directory to store cached audio/spectrograms (default: <output>/cache",
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="Rank for multi-GPU training"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # -------------------------------------------------------------------------

    assert torch.cuda.is_available(), "GPU is required for training"

    local_rank = os.environ.get("LOCAL_RANK", args.local_rank)
    if local_rank is not None:
        local_rank = int(local_rank)

    is_distributed = local_rank is not None

    if is_distributed:
        _LOGGER.info("Setting up distributed run (rank=%s)", local_rank)
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # -------------------------------------------------------------------------

    # Convert to paths
    args.output = Path(args.output)
    args.dataset = [
        (dataset_name, Path(phonemes_path), Path(audio_dir))
        for dataset_name, phonemes_path, audio_dir in args.dataset
    ]

    if args.config:
        args.config = [Path(p) for p in args.config]
    else:
        output_config_path = args.output / "config.json"
        assert (
            output_config_path.is_file()
        ), f"No config file found at {output_config_path}"

        args.config = [output_config_path]

    if args.checkpoint:
        args.checkpoint = Path(args.checkpoint)

    if args.cache:
        args.cache = Path(args.cache)
    else:
        args.cache = args.output / "cache"

    # Load configuration
    config = TrainingConfig()
    if args.config:
        _LOGGER.debug("Loading configuration(s) from %s", args.config)
        config = TrainingConfig.load_and_merge(config, args.config)

    config.git_commit = args.git_commit

    if not config.phonemes.phoneme_to_id:
        phonemes_path = args.output / "phonemes.txt"
        if phonemes_path.is_file():
            _LOGGER.debug("Loading phonemes from %s", phonemes_path)
            with open(phonemes_path, "r", encoding="utf-8") as phonemes_file:
                config.phonemes.phoneme_to_id = load_phoneme_ids(phonemes_file)

    _LOGGER.debug(config)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    _LOGGER.debug("Setting random seed to %s", config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    if args.epochs is not None:
        # Use command-line option
        config.epochs = args.epochs

    num_speakers = config.model.n_speakers
    if num_speakers > 1:
        assert (
            config.model.gin_channels > 0
        ), "Multispeaker model must have gin_channels > 0"

    assert (
        len(args.dataset) <= num_speakers
    ), "More datasets than speakers in model config"

    datasets = []
    speaker_id_map: typing.Dict[str, int] = {}
    for dataset_name, metadata_dir, audio_dir in args.dataset:
        metadata_dir = Path(metadata_dir)
        audio_dir = Path(audio_dir)

        datasets.append(
            load_dataset(
                config=config,
                dataset_name=dataset_name,
                metadata_dir=metadata_dir,
                audio_dir=audio_dir,
                speaker_id_map=speaker_id_map,
            )
        )

    # Create data loader
    batch_size = config.batch_size if args.batch_size is None else args.batch_size
    train_dataset = PhonemeIdsAndMelsDataset(
        config, datasets, split="train", cache_dir=args.cache
    )
    val_dataset = PhonemeIdsAndMelsDataset(
        config, datasets, split="val", cache_dir=args.cache
    )
    collate_fn = UtteranceCollate()

    train_loader = DataLoader(
        train_dataset,
        shuffle=(not is_distributed),
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=DistributedSampler(train_dataset) if is_distributed else None,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.eval_batch_size,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    model_g: typing.Optional[SynthesizerTrn] = None
    model_d: typing.Optional[MultiPeriodDiscriminator] = None
    optimizer_g: typing.Optional[torch.optim.Optimizer] = None
    optimizer_d: typing.Optional[torch.optim.Optimizer] = None
    scheduler_g: typing.Optional[torch.optim.lr_scheduler._LRScheduler] = None
    scheduler_d: typing.Optional[torch.optim.lr_scheduler._LRScheduler] = None

    if args.checkpoint:
        _LOGGER.debug(
            "Loading checkpoint from %s (optimizers=%s, schedulers=%s)",
            args.checkpoint,
            not args.no_restore_optimizers,
            not args.no_restore_schedulers,
        )
        checkpoint = load_checkpoint(
            args.checkpoint,
            config,
            load_optimizers=(not args.no_restore_optimizers),
            load_schedulers=(not args.no_restore_schedulers),
        )
        model_g, model_d, optimizer_g, optimizer_d, scheduler_g, scheduler_d = (
            checkpoint.model_g,
            checkpoint.model_d,
            checkpoint.optimizer_g,
            checkpoint.optimizer_d,
            checkpoint.scheduler_g,
            checkpoint.scheduler_d,
        )

        config.last_epoch = checkpoint.epoch
        config.global_step = checkpoint.global_step
        config.best_loss = checkpoint.best_loss

        _LOGGER.info(
            "Loaded checkpoint from %s (global step=%s, best loss=%s)",
            args.checkpoint,
            config.global_step,
            config.best_loss,
        )
    else:
        # Fresh models
        model_g = setup_model(config)
        model_d = setup_discriminator(config)

    if is_distributed:
        model_g = DistributedDataParallel(
            model_g,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        model_d = DistributedDataParallel(
            model_d,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # Train
    _LOGGER.info(
        "Training started (batch size=%s, epochs=%s)", batch_size, config.epochs
    )

    try:
        train(
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            model_dir=args.output,
            model_g=model_g,
            model_d=model_d,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
            checkpoint_epochs=args.checkpoint_epochs,
            rank=(local_rank if is_distributed else 0),
        )
        _LOGGER.info("Training finished")
    except KeyboardInterrupt:
        _LOGGER.info("Training stopped")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
