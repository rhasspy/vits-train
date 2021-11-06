#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

from vits_train.checkpoint import load_checkpoint, save_checkpoint
from vits_train.config import TrainingConfig

_LOGGER = logging.getLogger("vits_train.export_generator")


# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="vits-export-generator")
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pth)")
    parser.add_argument("output", help="Path to generator output file")
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
    checkpoint = load_checkpoint(
        args.checkpoint,
        config=config,
        load_discrimiator=False,
        load_optimizers=False,
        load_schedulers=False,
        use_cuda=False,
    )

    _LOGGER.info(
        "Loaded checkpoint from %s", args.checkpoint,
    )

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write new .pth
    save_checkpoint(checkpoint, args.output)

    _LOGGER.info("Exported generator to %s", args.output)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
