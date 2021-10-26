import logging
import tempfile
import typing
from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from vits_train.config import TrainingConfig
from vits_train.mel_processing import spectrogram_torch

_LOGGER = logging.getLogger("vits_train.dataset")


@dataclass
class Utterance:
    id: str
    phoneme_ids: typing.Sequence[int]
    audio_path: Path
    spec_length: int
    speaker_id: typing.Optional[int] = None


@dataclass
class UtteranceTensors:
    id: str
    phoneme_ids: torch.LongTensor
    spectrogram: torch.FloatTensor
    audio_norm: torch.FloatTensor
    spec_length: int
    speaker_id: typing.Optional[torch.LongTensor] = None


@dataclass
class Batch:
    phoneme_ids: torch.LongTensor
    phoneme_lengths: torch.LongTensor
    spectrograms: torch.FloatTensor
    spectrogram_lengths: torch.LongTensor
    audios: torch.FloatTensor
    audio_lengths: torch.LongTensor
    speaker_ids: typing.Optional[torch.LongTensor] = None


# -----------------------------------------------------------------------------


class PhonemeIdsAndMelsDataset(Dataset):
    def __init__(
        self,
        config: TrainingConfig,
        utt_phoneme_ids: typing.Mapping[str, typing.Sequence[int]],
        audio_dir: typing.Union[str, Path],
        spec_lengths: typing.Mapping[str, int],
        utt_speaker_ids: typing.Optional[typing.Mapping[str, int]] = None,
        cache_dir: typing.Optional[typing.Union[str, Path]] = None,
    ):
        super().__init__()

        self.config = config
        self.audio_dir = Path(audio_dir)
        self.utterances: typing.List[Utterance] = []

        self.temp_dir: typing.Optional[tempfile.TemporaryDirectory] = None

        if cache_dir is None:
            self.temp_dir = tempfile.TemporaryDirectory(prefix="vits_train")
            self.cache_dir = Path(self.temp_dir.name)
        else:
            self.cache_dir = Path(cache_dir)

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
                        spec_length=spec_lengths[utt_id],
                        speaker_id=utt_speaker_ids.get(utt_id),
                    )
                )
            else:
                _LOGGER.warning("Missing audio file: %s", audio_path)

    def __getitem__(self, index):
        utterance = self.utterances[index]

        audio_norm_path = (
            self.cache_dir / utterance.audio_path.with_suffix(".norm.pt").name
        )
        if audio_norm_path.is_file():
            audio_norm = torch.load(str(audio_norm_path))
        else:
            # Load audio and resample
            audio, _sample_rate = librosa.load(
                str(utterance.audio_path), sr=self.config.audio.sample_rate
            )

            audio_norm = torch.FloatTensor(
                audio / self.config.audio.max_wav_value
            ).unsqueeze(0)

            torch.save(audio_norm, str(audio_norm_path))

        spectrogram_path = (
            self.cache_dir / utterance.audio_path.with_suffix(".spec.pt").name
        )
        if spectrogram_path.is_file():
            spectrogram = torch.load(str(spectrogram_path))
        else:
            spectrogram = spectrogram_torch(
                y=audio_norm,
                n_fft=self.config.audio.filter_length,
                sampling_rate=self.config.audio.sample_rate,
                hop_size=self.config.audio.hop_length,
                win_size=self.config.audio.win_length,
                center=False,
            ).squeeze(0)

            torch.save(spectrogram, str(spectrogram_path))

        speaker_id = None
        if utterance.speaker_id is not None:
            speaker_id = torch.LongTensor([utterance.speaker_id])

        return UtteranceTensors(
            id=utterance.id,
            phoneme_ids=torch.LongTensor(utterance.phoneme_ids),
            audio_norm=audio_norm,
            spectrogram=spectrogram,
            spec_length=utterance.spec_length,
            speaker_id=speaker_id,
        )

    def __len__(self):
        return len(self.utterances)


class UtteranceCollate:
    def __call__(self, utterances: typing.Sequence[UtteranceTensors]) -> Batch:
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


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = [utt.spec_length for utt in dataset]
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
