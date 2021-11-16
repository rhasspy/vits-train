#!/usr/bin/env python3
import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

import torch

from vits_train.checkpoint import load_checkpoint
from vits_train.commons import intersperse
from vits_train.config import TrainingConfig
from vits_train.utils import audio_float_to_int16
from vits_train.wavfile import write as write_wav

_LOGGER = logging.getLogger("vits_train.infer")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="vits-train.infer")
    parser.add_argument("--model-dir", help="Path to model directory")
    parser.add_argument(
        "--checkpoint", help="Path to model checkpoint (default: best_model.pth)"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to write WAV file(s) (default: current directory)",
    )
    parser.add_argument(
        "--text", action="store_true", help="Input is text instead of phoneme ids"
    )
    parser.add_argument(
        "--csv", action="store_true", help="Input format is id|p1 p2 p3..."
    )
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--cuda", action="store_true", help="Use GPU for inference")
    parser.add_argument(
        "--speaker", type=int, help="Speaker id number (multispeaker model only)"
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
    if args.model_dir:
        args.model_dir = Path(args.model_dir)
    else:
        args.model_dir = Path.cwd()

    if args.output_dir:
        args.output_dir = Path(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.output_dir = Path.cwd()

    # Load config
    config_path = args.model_dir / "config.json"
    _LOGGER.debug("Loading configuration(s) from %s", config_path)
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = TrainingConfig.load(config_file)

    _LOGGER.debug(config)

    phoneme_to_id = {}
    if args.text:
        # Load phonemes
        num_phonemes = 0
        phonemes_path = args.model_dir / "phonemes.txt"
        _LOGGER.debug("Loading configuration(s) from %s", phonemes_path)
        with open(phonemes_path, "r", encoding="utf-8") as phonemes_file:
            for line in phonemes_file:
                line = line.strip("\r\n")
                if (not line) or line.startswith("#"):
                    continue

                phoneme_id, phoneme = re.split(r"[ \t]", line, maxsplit=1)

                # Avoid overwriting duplicates
                if phoneme not in phoneme_to_id:
                    phoneme_id = int(phoneme_id)
                    phoneme_to_id[phoneme] = phoneme_id

                # Need to count separately because phonemes may be duplicated
                num_phonemes += 1

        assert (
            num_phonemes == config.model.num_symbols
        ), f"Model has {config.model.num_symbols}, but phonemes.txt has {num_phonemes}"

    # Load checkpoint
    start_time = time.perf_counter()

    # Checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = args.model_dir / "best_model.pth"

    _LOGGER.debug("Loading checkpoint from %s", checkpoint_path)
    model = load_checkpoint(
        str(checkpoint_path),
        config=config,
        load_discrimiator=False,
        load_optimizers=False,
        load_schedulers=False,
        use_cuda=args.cuda,
    ).model_g

    end_time = time.perf_counter()

    _LOGGER.info(
        "Loaded checkpoint from %s in %s second(s)",
        checkpoint_path,
        end_time - start_time,
    )

    if args.cuda:
        model.cuda()

    model.eval()

    # Multispeaker
    if config.model.n_speakers > 1:
        if args.speaker is None:
            args.speaker = 0

    # -------------------------------------------------------------------------

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
            speaker_num = args.speaker

            if args.csv:
                # Input format is id | p1 p2 p3...
                line_parts = line.split("|", maxsplit=3)
                utt_id = line_parts[0]
                line = line_parts[-1]

                if (len(line_parts) > 2) and (config.model.n_speakers > 1):
                    speaker_num = int(line_parts[1])

            if args.text:
                # Map phonemes to ids
                assert phoneme_to_id, "No phonemes were loaded"
                phoneme_ids = [phoneme_to_id[p] for p in line if p in phoneme_to_id]
                # phoneme_ids = intersperse(phoneme_ids, 0)
            else:
                # Phoneme ids as p1 p2 p3...
                phoneme_ids = [int(p) for p in line.split()]

            _LOGGER.debug("%s (id=%s)", phoneme_ids, utt_id)

            with torch.no_grad():
                text = torch.LongTensor(phoneme_ids).unsqueeze(0)
                text_lengths = torch.LongTensor([text.shape[1]])
                speaker_id = None

                if config.model.n_speakers > 1:
                    speaker_id = torch.LongTensor([speaker_num])

                if args.cuda:
                    text.contiguous().cuda()
                    text_lengths.contiguous().cuda()

                    if speaker_id is not None:
                        speaker_id = speaker_id.cuda()

                _LOGGER.debug(
                    "Inferring audio for %s symbols (speaker=%s)",
                    text.shape[1],
                    speaker_id,
                )

                start_time = time.perf_counter()
                audio = (
                    model.infer(
                        text,
                        text_lengths,
                        sid=speaker_id,
                        noise_scale=args.noise_scale,
                        noise_scale_w=args.noise_scale_w,
                        length_scale=args.length_scale,
                    )[0][0, 0]
                    .data.cpu()
                    .float()
                    .numpy()
                )
                audio = audio_float_to_int16(audio)
                end_time = time.perf_counter()

                audio_duration_sec = audio.shape[-1] / config.audio.sample_rate
                infer_sec = end_time - start_time
                real_time_factor = (
                    infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0
                )

                _LOGGER.debug(
                    "Real-time factor: %0.2f (infer=%0.2f sec, audio=%0.2f sec)",
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
