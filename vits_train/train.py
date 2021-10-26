import csv
import math
import logging
import re
import typing
from dataclasses import dataclass
from pathlib import Path

import librosa
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

from vits_train import setup_model
from vits_train.losses import generator_loss, feature_loss, discriminator_loss, kl_loss
from vits_train.config import TrainingConfig
from vits_train.commons import slice_segments, intersperse
from vits_train.models import MultiPeriodDiscriminator
from vits_train.mel_processing import (
    spectrogram_torch,
    spec_to_mel_torch,
    mel_spectrogram_torch,
)

_LOGGER = logging.getLogger("vits_train.train")


@dataclass
class Batch:
    phoneme_ids: torch.LongTensor
    phoneme_lengths: torch.LongTensor
    spectrograms: torch.FloatTensor
    spectrogram_lengths: torch.LongTensor
    audios: torch.FloatTensor
    audio_lengths: torch.LongTensor
    speaker_ids: typing.Optional[torch.LongTensor] = None


class VITSTraining(pl.LightningModule):
    def __init__(
        self,
        config: TrainingConfig,
        utt_phoneme_ids: typing.Mapping[str, torch.LongTensor],
        audio_dir: typing.Union[str, Path],
        train_ids: typing.Iterable[str],
        val_ids: typing.Iterable[str],
        test_ids: typing.Iterable[str],
        utt_speaker_ids: typing.Optional[typing.Mapping[str, int]] = None,
    ):
        super().__init__()

        self.config = config
        self.utt_phoneme_ids = utt_phoneme_ids
        self.audio_dir = Path(audio_dir)
        self.utt_speaker_ids = utt_speaker_ids if utt_speaker_ids is not None else {}

        self.net_g = setup_model(config)
        self.net_d = MultiPeriodDiscriminator(config.model.use_spectral_norm)

        self.collate_fn = UtteranceCollate()

        # Filter utterances based on min/max settings in config
        drop_utt_ids: typing.Set[str] = set()

        num_phonemes_too_small = 0
        num_phonemes_too_large = 0
        num_audio_missing = 0
        num_spec_too_small = 0
        num_spec_too_large = 0

        for utt_id, phoneme_ids in utt_phoneme_ids.items():
            # Check phonemes length
            if (self.config.min_phonemes_len is not None) and (
                len(phoneme_ids) < self.config.min_phonemes_len
            ):
                drop_utt_ids.add(utt_id)
                num_phonemes_too_small += 1
                continue

            if (self.config.max_phonemes_len is not None) and (
                len(phoneme_ids) > self.config.max_phonemes_len
            ):
                drop_utt_ids.add(utt_id)
                num_phonemes_too_large += 1
                continue

            # Check if audio file is missing
            audio_path = self.audio_dir / utt_id
            if not audio_path.is_file():
                # Try WAV extension
                audio_path = self.audio_dir / f"{utt_id}.wav"

            if not audio_path.is_file():
                drop_utt_ids.add(utt_id)
                _LOGGER.warning(
                    "Dropped %s because audio file is missing: %s", utt_id, audio_path
                )
                continue

            if (self.config.min_spec_len is None) and (
                self.config.max_spec_len is None
            ):
                # Don't bother checking spectrogram length
                continue

            # Check estimated spectrogram length
            duration_sec = librosa.get_duration(filename=str(audio_path))
            num_samples = int(math.ceil(self.config.audio.sample_rate * duration_sec))
            spec_length = num_samples // self.config.audio.hop_length

            if (self.config.min_spec_len is not None) and (
                spec_length < self.config.min_spec_len
            ):
                drop_utt_ids.add(utt_id)
                num_spec_too_small += 1
                continue

            if (self.config.max_spec_len is not None) and (
                spec_length > self.config.max_spec_len
            ):
                drop_utt_ids.add(utt_id)
                num_spec_too_large += 1
                continue

        # Filter out dropped utterances
        if drop_utt_ids:
            _LOGGER.info("Dropped %s utterance(s)", len(drop_utt_ids))

            if num_phonemes_too_small > 0:
                _LOGGER.debug(
                    "%s utterance(s) dropped whose phoneme length was smaller than %s",
                    num_phonemes_too_small,
                    self.config.min_phonemes_len,
                )

            if num_phonemes_too_large > 0:
                _LOGGER.debug(
                    "%s utterance(s) dropped whose phoneme length was larger than %s",
                    num_phonemes_too_large,
                    self.config.max_phonemes_len,
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
                    self.config.min_spec_len,
                )

            if num_spec_too_large > 0:
                _LOGGER.debug(
                    "%s utterance(s) dropped whose spectrogram length was larger than %s",
                    num_spec_too_large,
                    self.config.max_spec_len,
                )

            utt_phoneme_ids = {
                utt_id: phoneme_ids
                for utt_id, phoneme_ids in utt_phoneme_ids.items()
                if utt_id not in drop_utt_ids
            }
        else:
            _LOGGER.info("Kept all %s utterances", len(utt_phoneme_ids))

        assert utt_phoneme_ids, "No utterances after filtering"

        self.train_dataset = PhonemeIdsAndMelsDataset(
            config=self.config,
            utt_phoneme_ids={
                utt_id: self.utt_phoneme_ids[utt_id] for utt_id in train_ids
            },
            audio_dir=self.audio_dir,
            utt_speaker_ids={
                utt_id: self.utt_speaker_ids[utt_id]
                for utt_id in train_ids
                if utt_id in self.utt_speaker_ids
            },
        )

        self.val_dataset = PhonemeIdsAndMelsDataset(
            config=self.config,
            utt_phoneme_ids={
                utt_id: self.utt_phoneme_ids[utt_id] for utt_id in val_ids
            },
            audio_dir=self.audio_dir,
            utt_speaker_ids={
                utt_id: self.utt_speaker_ids[utt_id]
                for utt_id in val_ids
                if utt_id in self.utt_speaker_ids
            },
        )

        self.test_dataset = PhonemeIdsAndMelsDataset(
            config=self.config,
            utt_phoneme_ids={
                utt_id: self.utt_phoneme_ids[utt_id] for utt_id in test_ids
            },
            audio_dir=self.audio_dir,
            utt_speaker_ids={
                utt_id: self.utt_speaker_ids[utt_id]
                for utt_id in test_ids
                if utt_id in self.utt_speaker_ids
            },
        )

    def forward(self, *args, **kwargs):
        return self.net_g(*args, **kwargs)

    def configure_optimizers(self):
        optim_g = torch.optim.AdamW(
            self.net_g.parameters(),
            self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps,
        )
        optim_d = torch.optim.AdamW(
            self.net_d.parameters(),
            self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps,
        )

        return optim_g, optim_d

    def training_step(self, train_batch: Batch, batch_idx: int, optimizer_idx: int):

        if optimizer_idx == 0:
            (
                y_hat,
                l_length,
                _attn,
                ids_slice,
                _x_mask,
                z_mask,
                (_z, z_p, m_p, logs_p, _m_q, logs_q),
            ) = self.net_g(
                train_batch.phoneme_ids,
                train_batch.phoneme_lengths,
                train_batch.spectrograms,
                train_batch.spectrogram_lengths,
            )

            self.train_y_hat = y_hat
            mel = spec_to_mel_torch(
                train_batch.spectrograms,
                self.config.audio.filter_length,
                self.config.audio.mel_channels,
                self.config.audio.sample_rate,
                self.config.audio.mel_fmin,
                self.config.audio.mel_fmax,
            )
            y_mel = slice_segments(
                mel, ids_slice, self.config.segment_size // self.config.audio.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                self.config.audio.filter_length,
                self.config.audio.mel_channels,
                self.config.audio.sample_rate,
                self.config.audio.hop_length,
                self.config.audio.win_length,
                self.config.audio.mel_fmin,
                self.config.audio.mel_fmax,
            )

            y = slice_segments(
                train_batch.audios,
                ids_slice * self.config.audio.hop_length,
                self.config.segment_size,
            )  # slice

            self.train_y = y

            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.config.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.config.c_kl

            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

            return loss_gen_all

        # Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = self.net_d(self.train_y, self.train_y_hat.detach())
        loss_disc, _losses_disc_r, _losses_disc_g = discriminator_loss(
            y_d_hat_r, y_d_hat_g
        )
        loss_disc_all = loss_disc

        _LOGGER.info(loss_disc_all.dtype)

        return loss_disc_all

    # def validation_step(self, val_batch: Batch, batch_idx, optimizer_idx):
    #     pass

    # def test_step(self, val_batch: Batch, batch_idx, optimizer_idx):
    #     pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            # pin_memory=True,
            collate_fn=self.collate_fn,
            num_workers=8,
            # TODO: bucket sampler
            # batch_sampler=train_sampler,
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_dataset,
    #         shuffle=False,
    #         batch_size=self.config.batch_size,
    #         # pin_memory=True,
    #         drop_last=False,
    #         collate_fn=self.collate_fn,
    #         num_workers=8,
    #     )

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test_dataset,
    #         shuffle=False,
    #         batch_size=self.config.batch_size,
    #         # pin_memory=True,
    #         drop_last=False,
    #         collate_fn=self.collate_fn,
    #         num_workers=8,
    #     )


# -----------------------------------------------------------------------------


@dataclass
class Utterance:
    id: str
    phoneme_ids: torch.LongTensor
    audio_path: Path
    speaker_id: typing.Optional[int] = None
    spectrogram: typing.Optional[torch.FloatTensor] = None
    audio_norm: typing.Optional[torch.FloatTensor] = None


class PhonemeIdsAndMelsDataset(Dataset):
    def __init__(
        self,
        config: TrainingConfig,
        utt_phoneme_ids: typing.Mapping[str, torch.LongTensor],
        audio_dir: typing.Union[str, Path],
        utt_speaker_ids: typing.Optional[typing.Mapping[str, int]] = None,
    ):
        super().__init__()

        self.config = config
        self.audio_dir = Path(audio_dir)
        self.utterances: typing.List[Utterance] = []

        if utt_speaker_ids is None:
            utt_speaker_ids = {}

        for utt_id, phoneme_ids in utt_phoneme_ids.items():
            audio_path = self.audio_dir / utt_id
            if not audio_path.is_file():
                # Try WAV extension
                audio_path = self.audio_dir / f"{utt_id}.wav"

            if audio_path.is_file():
                self.utterances.append(
                    Utterance(
                        id=utt_id,
                        phoneme_ids=phoneme_ids,
                        audio_path=audio_path,
                        speaker_id=utt_speaker_ids.get(utt_id),
                    )
                )
            else:
                _LOGGER.warning("Missing audio file: %s", audio_path)

    def __getitem__(self, index):
        utterance = self.utterances[index]

        if utterance.audio_norm is None:
            # Load audio and resample
            # _LOGGER.debug("Loading audio file: %s", utterance.audio_path)
            audio, _sample_rate = librosa.load(
                str(utterance.audio_path), sr=self.config.audio.sample_rate
            )

            utterance.audio_norm = torch.tensor(
                audio / self.config.audio.max_wav_value
            ).unsqueeze(0)

        assert utterance.audio_norm is not None

        if utterance.spectrogram is None:
            utterance.spectrogram = spectrogram_torch(
                utterance.audio_norm,
                self.config.audio.filter_length,
                self.config.audio.sample_rate,
                self.config.audio.hop_length,
                self.config.audio.win_length,
                center=False,
            ).squeeze(0)

        assert utterance.spectrogram is not None

        return utterance

    def __len__(self):
        return len(self.utterances)


class UtteranceCollate:
    def __call__(self, utterances: typing.Sequence[Utterance]) -> Batch:
        # _, ids_sorted_decreasing = torch.sort(
        #     torch.LongTensor([x[1].size(1) for x in utterances]), dim=0, descending=True
        # )

        num_utterances = len(utterances)
        assert num_utterances > 0, "No utterances"

        max_phonemes_length = 0
        max_spec_length = 0
        max_audio_length = 0

        num_mels = 0
        multispeaker = False

        # Determine lengths
        for utt_idx, utt in enumerate(utterances):
            assert utt.spectrogram is not None
            assert utt.audio_norm is not None

            phoneme_length = utt.phoneme_ids.size(0)
            spec_length = utt.spectrogram.size(1)
            audio_length = utt.audio_norm.size(1)

            max_phonemes_length = max(max_phonemes_length, phoneme_length)
            max_spec_length = max(max_spec_length, spec_length)
            max_audio_length = max(max_audio_length, audio_length)

            num_mels = utt.spectrogram.size(0)
            if utt.speaker_id is not None:
                multispeaker = True

        # Create padded tensors
        phonemes_padded = torch.LongTensor(num_utterances, max_phonemes_length)
        spec_padded = torch.FloatTensor(num_utterances, num_mels, max_spec_length)
        audio_padded = torch.FloatTensor(num_utterances, 1, max_audio_length)

        phonemes_padded.zero_()
        spec_padded.zero_()
        audio_padded.zero_()

        phoneme_lengths = torch.LongTensor(num_utterances)
        spec_lengths = torch.LongTensor(num_utterances)
        audio_lengths = torch.LongTensor(num_utterances)

        speaker_ids: typing.Optional[torch.LongTensor] = None
        if multispeaker:
            speaker_ids = torch.LongTensor(num_utterances)

        # Sort by decreasing spectrogram length
        sorted_utterances = sorted(
            utterances, key=lambda u: u.spectrogram.size(1), reverse=True
        )
        for utt_idx, utt in enumerate(sorted_utterances):
            phoneme_length = utt.phoneme_ids.size(0)
            spec_length = utt.spectrogram.size(1)
            audio_length = utt.audio_norm.size(1)

            phonemes_padded[utt_idx, :phoneme_length] = utt.phoneme_ids
            phoneme_lengths[utt_idx] = phoneme_length

            spec_padded[utt_idx, :, :spec_length] = utt.spectrogram
            spec_lengths[utt_idx] = spec_length

            audio_padded[utt_idx, :, :audio_length] = utt.audio_norm
            audio_lengths[utt_idx] = audio_length

            if utt.speaker_id is not None:
                assert speaker_ids is not None
                speaker_ids[utt_idx] = utt.speaker_id

        return Batch(
            phoneme_ids=phonemes_padded,
            phoneme_lengths=phoneme_lengths,
            spectrograms=spec_padded,
            spectrogram_lengths=spec_lengths,
            audios=audio_padded,
            audio_lengths=audio_lengths,
            speaker_ids=speaker_ids,
        )


# -----------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.DEBUG)

    train_path = "data/ljspeech/ljs_audio_text_train_filelist.txt.cleaned"
    val_path = "data/ljspeech/ljs_audio_text_val_filelist.txt.cleaned"
    test_path = "data/ljspeech/ljs_audio_text_test_filelist.txt.cleaned"

    config_path = "local/ljspeech/config.json"
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = TrainingConfig.load(config_file)

    phoneme_to_id = {}
    phonemes_path = "local/ljspeech/phonemes.txt"
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
    train_ids = []
    val_ids = []
    test_ids = []

    for ids, csv_path in [
        (train_ids, train_path),
        (val_ids, val_path),
        (test_ids, test_path),
    ]:
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file, delimiter="|")
            for row_idx, row in enumerate(reader):
                # TODO: speaker
                assert len(row) > 1, f"{row} in {csv_path}:{row_idx+1}"
                utt_id, phonemes = row[0], row[1]
                phoneme_ids = [phoneme_to_id[p] for p in phonemes if p in phoneme_to_id]

                assert phoneme_ids, utt_id
                phoneme_ids = intersperse(phoneme_ids, 0)
                utt_phoneme_ids[utt_id] = torch.LongTensor(phoneme_ids)
                ids.append(utt_id)

                if len(ids) > 20:
                    break

        break

    model = VITSTraining(
        config=config,
        utt_phoneme_ids=utt_phoneme_ids,
        audio_dir="data/ljspeech/wavs",
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )
    trainer = pl.Trainer(gpus=1, precision=16)
    trainer.fit(model)


if __name__ == "__main__":
    main()
