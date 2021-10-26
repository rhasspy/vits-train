#!/usr/bin/env python3
import re
import csv
import math
import typing
import os
import logging
from pathlib import Path

import librosa
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from vits_train import commons, utils
from vits_train.config import TrainingConfig
from vits_train.dataset import (
    PhonemeIdsAndMelsDataset,
    UtteranceCollate,
    DistributedBucketSampler,
    Batch,
)
from vits_train import setup_model
from vits_train.models import MultiPeriodDiscriminator
from vits_train.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from vits_train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch


torch.backends.cudnn.benchmark = True
global_step = 0
log_interval = 25
eval_interval = 25

_LOGGER = logging.getLogger("vits_train.train")


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "80000"

    logging.basicConfig(level=logging.DEBUG)

    model_dir = Path("local/ljspeech")

    config_path = model_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = TrainingConfig.load(config_file)

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, model_dir, config,))


def run(rank: int, n_gpus: int, model_dir: Path, config: TrainingConfig):
    setattr(config, "model_dir", str(model_dir))
    audio_dir = Path("data") / "ljspeech" / "wavs"
    cache_dir = audio_dir

    global global_step
    if rank == 0:
        _LOGGER.debug(config)

        writer = SummaryWriter(log_dir=str(model_dir))
        writer_eval = SummaryWriter(log_dir=str(model_dir / "eval"))

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(config.seed)
    torch.cuda.set_device(rank)

    train_path = model_dir / "train.csv"
    val_path = model_dir / "val.csv"

    phoneme_to_id = {}
    phonemes_path = model_dir / "phonemes.txt"
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

    utt_phoneme_ids = {}
    utt_speaker_ids = {}
    train_ids = []
    val_ids = []

    for ids, csv_path in [
        (train_ids, train_path),
        (val_ids, val_path),
    ]:
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file, delimiter="|")
            for row_idx, row in enumerate(reader):
                # TODO: speaker
                assert len(row) > 1, f"{row} in {csv_path}:{row_idx+1}"
                utt_id, phonemes = row[0], row[1]
                phoneme_ids = [phoneme_to_id[p] for p in phonemes if p in phoneme_to_id]

                assert phoneme_ids, utt_id
                phoneme_ids = commons.intersperse(phoneme_ids, 0)
                utt_phoneme_ids[utt_id] = phoneme_ids
                ids.append(utt_id)

    # Filter utterances based on min/max settings in config
    drop_utt_ids: typing.Set[str] = set()

    num_phonemes_too_small = 0
    num_phonemes_too_large = 0
    num_audio_missing = 0
    num_spec_too_small = 0
    num_spec_too_large = 0

    spec_lengths: typing.Dict[str, int] = {}

    for utt_id, phoneme_ids in utt_phoneme_ids.items():
        # Check phonemes length
        if (config.min_phonemes_len is not None) and (
            len(phoneme_ids) < config.min_phonemes_len
        ):
            drop_utt_ids.add(utt_id)
            num_phonemes_too_small += 1
            continue

        if (config.max_phonemes_len is not None) and (
            len(phoneme_ids) > config.max_phonemes_len
        ):
            drop_utt_ids.add(utt_id)
            num_phonemes_too_large += 1
            continue

        # Check if audio file is missing
        audio_path = audio_dir / utt_id
        if not audio_path.is_file():
            # Try WAV extension
            audio_path = audio_dir / f"{utt_id}.wav"

        if not audio_path.is_file():
            drop_utt_ids.add(utt_id)
            _LOGGER.warning(
                "Dropped %s because audio file is missing: %s", utt_id, audio_path
            )
            continue

        if (config.min_spec_len is None) and (config.max_spec_len is None):
            # Don't bother checking spectrogram length
            continue

        # Check estimated spectrogram length
        duration_sec = librosa.get_duration(filename=str(audio_path))
        num_samples = int(math.ceil(config.audio.sample_rate * duration_sec))
        spec_length = num_samples // config.audio.hop_length

        if (config.min_spec_len is not None) and (spec_length < config.min_spec_len):
            drop_utt_ids.add(utt_id)
            num_spec_too_small += 1
            continue

        if (config.max_spec_len is not None) and (spec_length > config.max_spec_len):
            drop_utt_ids.add(utt_id)
            num_spec_too_large += 1
            continue

        # Cache for datasets
        spec_lengths[utt_id] = spec_length

    # Filter out dropped utterances
    if drop_utt_ids:
        _LOGGER.info("Dropped %s utterance(s)", len(drop_utt_ids))

        if num_phonemes_too_small > 0:
            _LOGGER.debug(
                "%s utterance(s) dropped whose phoneme length was smaller than %s",
                num_phonemes_too_small,
                config.min_phonemes_len,
            )

        if num_phonemes_too_large > 0:
            _LOGGER.debug(
                "%s utterance(s) dropped whose phoneme length was larger than %s",
                num_phonemes_too_large,
                config.max_phonemes_len,
            )

        if num_audio_missing > 0:
            _LOGGER.debug(
                "%s utterance(s) dropped whose audio file was missing",
                num_audio_missing,
            )

        if num_spec_too_small > 0:
            _LOGGER.debug(
                "%s utterance(s) dropped whose spectrogram length was smaller than %s",
                num_spec_too_small,
                config.min_spec_len,
            )

        if num_spec_too_large > 0:
            _LOGGER.debug(
                "%s utterance(s) dropped whose spectrogram length was larger than %s",
                num_spec_too_large,
                config.max_spec_len,
            )

        utt_phoneme_ids = {
            utt_id: phoneme_ids
            for utt_id, phoneme_ids in utt_phoneme_ids.items()
            if utt_id not in drop_utt_ids
        }
    else:
        _LOGGER.info("Kept all %s utterances", len(utt_phoneme_ids))

    assert utt_phoneme_ids, "No utterances after filtering"

    train_ids = set(train_ids) - drop_utt_ids
    assert train_ids, "No training utterances after filtering"

    val_ids = set(val_ids) - drop_utt_ids
    assert val_ids, "No validation utterances after filtering"

    # DEBUG
    train_ids = list(train_ids)[:20]
    val_ids = list(val_ids)[:20]

    train_dataset = PhonemeIdsAndMelsDataset(
        config=config,
        utt_phoneme_ids={utt_id: utt_phoneme_ids[utt_id] for utt_id in train_ids},
        audio_dir=audio_dir,
        utt_speaker_ids={
            utt_id: utt_speaker_ids[utt_id]
            for utt_id in train_ids
            if utt_id in utt_speaker_ids
        },
        spec_lengths={utt_id: spec_lengths[utt_id] for utt_id in train_ids},
        cache_dir=cache_dir,
    )

    train_sampler = DistributedBucketSampler(
        train_dataset,
        config.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = UtteranceCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )
    if rank == 0:
        eval_dataset = PhonemeIdsAndMelsDataset(
            config=config,
            utt_phoneme_ids={utt_id: utt_phoneme_ids[utt_id] for utt_id in val_ids},
            audio_dir=audio_dir,
            utt_speaker_ids={
                utt_id: utt_speaker_ids[utt_id]
                for utt_id in val_ids
                if utt_id in utt_speaker_ids
            },
            spec_lengths={utt_id: spec_lengths[utt_id] for utt_id in val_ids},
            cache_dir=cache_dir,
        )
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=8,
            shuffle=False,
            batch_size=config.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    net_g = setup_model(config).cuda(rank)
    net_d = MultiPeriodDiscriminator(config.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(), config.learning_rate, betas=config.betas, eps=config.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(), config.learning_rate, betas=config.betas, eps=config.eps,
    )
    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(model_dir, "G_*.pth"), net_g, optim_g
        )
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(model_dir, "D_*.pth"), net_d, optim_d
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=config.fp16_run)

    for epoch in range(epoch_str, config.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                config,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, eval_loader],
                None,
                [writer, writer_eval],
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                config,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank, epoch, config, nets, optims, schedulers, scaler, loaders, logger, writers
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    _scheduler_g, _scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = typing.cast(Batch, batch)
        x, x_lengths, spec, spec_lengths, y, y_lengths = (
            batch.phoneme_ids,
            batch.phoneme_lengths,
            batch.spectrograms,
            batch.spectrogram_lengths,
            batch.audios,
            batch.audio_lengths,
        )
        x, x_lengths = (
            x.cuda(rank, non_blocking=True),
            x_lengths.cuda(rank, non_blocking=True),
        )
        spec, spec_lengths = (
            spec.cuda(rank, non_blocking=True),
            spec_lengths.cuda(rank, non_blocking=True),
        )
        y, y_lengths = (
            y.cuda(rank, non_blocking=True),
            y_lengths.cuda(rank, non_blocking=True),
        )

        with autocast(enabled=config.fp16_run):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                _x_mask,
                z_mask,
                (_z, z_p, m_p, logs_p, _m_q, logs_q),
            ) = net_g(x, x_lengths, spec, spec_lengths)

            mel = spec_to_mel_torch(
                spec,
                config.audio.filter_length,
                config.audio.mel_channels,
                config.audio.sample_rate,
                config.audio.mel_fmin,
                config.audio.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, config.segment_size // config.audio.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                config.audio.filter_length,
                config.audio.mel_channels,
                config.audio.sample_rate,
                config.audio.hop_length,
                config.audio.win_length,
                config.audio.mel_fmin,
                config.audio.mel_fmax,
            )

            y = commons.slice_segments(
                y, ids_slice * config.audio.hop_length, config.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=config.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * config.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                _LOGGER.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                _LOGGER.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/dur": loss_dur,
                        "loss/g/kl": loss_kl,
                    }
                )

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/attn": utils.plot_alignment_to_numpy(
                        attn[0, 0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            if global_step % eval_interval == 0:
                evaluate(config, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    config.learning_rate,
                    epoch,
                    os.path.join(config.model_dir, "G_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    config.learning_rate,
                    epoch,
                    os.path.join(config.model_dir, "D_{}.pth".format(global_step)),
                )
        global_step += 1

    if rank == 0:
        _LOGGER.info("====> Epoch: {}".format(epoch))


def evaluate(config, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
        for _batch_idx, batch in enumerate(eval_loader):
            batch = typing.cast(Batch, batch)
            x, x_lengths, spec, spec_lengths, y, y_lengths = (
                batch.phoneme_ids,
                batch.phoneme_lengths,
                batch.spectrograms,
                batch.spectrogram_lengths,
                batch.audios,
                batch.audio_lengths,
            )

            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)

            # remove else
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            break
        y_hat, _attn, mask, *_ = generator.module.infer(x, x_lengths, max_len=1000)
        y_hat_lengths = mask.sum([1, 2]).long() * config.audio.hop_length

        mel = spec_to_mel_torch(
            spec,
            config.audio.filter_length,
            config.audio.mel_channels,
            config.audio.sample_rate,
            config.audio.mel_fmin,
            config.audio.mel_fmax,
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            config.audio.filter_length,
            config.audio.mel_channels,
            config.audio.sample_rate,
            config.audio.hop_length,
            config.audio.win_length,
            config.audio.mel_fmin,
            config.audio.mel_fmax,
        )
    image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {"gen/audio": y_hat[0, :, : y_hat_lengths[0]]}
    if global_step == 0:
        image_dict.update(
            {"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())}
        )
        audio_dict.update({"gt/audio": y[0, :, : y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=config.audio.sample_rate,
    )
    generator.train()


if __name__ == "__main__":
    main()
