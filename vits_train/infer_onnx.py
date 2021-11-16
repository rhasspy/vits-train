#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime

from vits_train.config import TrainingConfig
from vits_train.utils import audio_float_to_int16
from vits_train.wavfile import write as write_wav

_LOGGER = logging.getLogger("vits_train.infer_onnx")

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="vits-train.infer_onnx")
    parser.add_argument("model", help="Path to onnx model")
    parser.add_argument(
        "--output-dir",
        help="Directory to write WAV file(s) (default: current directory)",
    )
    parser.add_argument(
        "--config", action="append", help="Path to JSON configuration file(s)"
    )
    parser.add_argument(
        "--csv", action="store_true", help="Input format is id|p1 p2 p3..."
    )
    parser.add_argument(
        "--no-optimizations", action="store_true", help="Disable Onnx optimizations"
    )
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # Convert to paths
    if args.config:
        args.config = [Path(p) for p in args.config]

    args.model = Path(args.model)

    if args.output_dir:
        args.output_dir = Path(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.output_dir = Path.cwd()

    # Load configuration
    config = TrainingConfig()
    if args.config:
        _LOGGER.debug("Loading configuration(s) from %s", args.config)
        config = TrainingConfig.load_and_merge(config, args.config)

    # Load model
    sess_options = onnxruntime.SessionOptions()
    if args.no_optimizations:
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )

    _LOGGER.debug("Loading model from %s", args.model)
    model = onnxruntime.InferenceSession(str(args.model), sess_options=sess_options)
    _LOGGER.info("Loaded model from %s", args.model)

    # Process input phonemes
    start_time = time.perf_counter()

    if os.isatty(sys.stdin.fileno()):
        print("Reading whitespace-separated phoneme ids from stdin...", file=sys.stderr)

    # Read phoneme ids from standard input.
    # Phoneme ids are separated by whitespace (<p1> <p2> ...)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            utt_id = "output"
            if args.csv:
                # Input format is id | p1 p2 p3...
                utt_id, line = line.split("|", maxsplit=1)

            # Phoneme ids as p1 p2 p3...
            phoneme_ids = [int(p) for p in line.split()]
            _LOGGER.debug("%s (id=%s)", phoneme_ids, utt_id)

            # Convert to tensors
            # TODO: Allow batches
            text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
            text_lengths = np.array([text.shape[1]], dtype=np.int64)
            scales = np.array(
                [args.noise_scale, args.length_scale, args.noise_scale_w],
                dtype=np.float32,
            )

            # Infer audio
            start_time = time.perf_counter()
            audio = model.run(
                None, {"input": text, "input_lengths": text_lengths, "scales": scales}
            )[0].squeeze()
            audio = audio_float_to_int16(audio)
            end_time = time.perf_counter()

            audio_duration_sec = audio.shape[-1] / config.audio.sample_rate
            infer_sec = end_time - start_time
            real_time_factor = (
                infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0
            )

            _LOGGER.debug(
                "Real-time factor for %s: %0.2f (infer=%0.2f sec, audio=%0.2f sec)",
                utt_id,
                real_time_factor,
                infer_sec,
                audio_duration_sec,
            )

            output_file_name = utt_id
            if not output_file_name.endswith(".wav"):
                output_file_name += ".wav"

            output_path = args.output_dir / output_file_name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            write_wav(str(output_path), config.audio.sample_rate, audio)

            _LOGGER.info("Wrote WAV to %s", output_path)

    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
