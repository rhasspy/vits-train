#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .config import TrainingConfig

_LOGGER = logging.getLogger("vits_train.export_onnx")

OPSET_VERSION = 12

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="vits-export-onnx")
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pth)")
    parser.add_argument("output", help="Path to output directory")
    parser.add_argument(
        "--config", action="append", help="Path to JSON configuration file(s)"
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

    # Convert to paths
    if args.config:
        args.config = [Path(p) for p in args.config]

    args.checkpoint = Path(args.checkpoint)
    args.output = Path(args.output)

    # Load configuration
    config = TrainingConfig()
    if args.config:
        _LOGGER.debug("Loading configuration(s) from %s", args.config)
        config = TrainingConfig.load_and_merge(config, args.config)

    # Load checkpoint
    _LOGGER.debug("Loading checkpoint from %s", args.checkpoint)
    model_g = load_checkpoint(
        args.checkpoint,
        config=config,
        load_discrimiator=True,
        load_optimizers=False,
        load_schedulers=False,
        use_cuda=False,
    ).model_g

    _LOGGER.info(
        "Loaded checkpoint from %s", args.checkpoint,
    )

    # Inference only
    model_g.eval()
    model_g.forward = model_g.infer

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Write config
    with open(args.output / "config.json", "w") as config_file:
        config.save(config_file)

    # Create dummy input
    sequences = torch.randint(
        low=0, high=config.model.num_symbols, size=(1, 50), dtype=torch.long
    )
    sequence_lengths = torch.IntTensor([sequences.size(1)]).long()

    # noise, noise_w, length
    scales = torch.FloatTensor([0.667, 0.8, 1.0])

    dummy_input = (sequences, sequence_lengths, scales)

    # Export
    torch.onnx.export(
        model_g,
        dummy_input,
        str(args.output / "generator.onnx"),
        opset_version=OPSET_VERSION,
        do_constant_folding=False,
        enable_onnx_checker=True,
        input_names=["input", "input_lengths", "scales"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time"},
        },
    )

    _LOGGER.info("Exported model to %s", args.output)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
