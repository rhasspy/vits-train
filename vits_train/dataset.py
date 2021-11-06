import csv
import logging
import math
import re
import shutil
import tempfile
import typing
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
from torch.utils.data import Dataset

from vits_train.config import TrainingConfig
from vits_train.mel_processing import spectrogram_torch

_LOGGER = logging.getLogger("vits_train.dataset")


@dataclass
class Utterance:
    id: str
    phoneme_ids: typing.List[int]
    audio_path: Path
    cache_path: typing.Optional[Path]
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


UTTERANCE_PHONEME_IDS = typing.Dict[str, typing.List[int]]
UTTERANCE_SPEAKER_IDS = typing.Dict[str, int]
UTTERANCE_IDS = typing.Collection[str]


@dataclass
class DatasetInfo:
    name: str
    audio_dir: Path
    utt_phoneme_ids: UTTERANCE_PHONEME_IDS
    utt_speaker_ids: UTTERANCE_SPEAKER_IDS
    split_ids: typing.Mapping[str, UTTERANCE_IDS]


# -----------------------------------------------------------------------------


class PhonemeIdsAndMelsDataset(Dataset):
    def __init__(
        self,
        config: TrainingConfig,
        datasets: typing.Sequence[DatasetInfo],
        split: str,
        cache_dir: typing.Optional[typing.Union[str, Path]] = None,
    ):
        super().__init__()

        self.config = config
        self.utterances = []
        self.split = split

        self.temp_dir: typing.Optional[tempfile.TemporaryDirectory] = None

        if cache_dir is None:
            # pylint: disable=consider-using-with
            self.temp_dir = tempfile.TemporaryDirectory(prefix="vits_train")
            self.cache_dir = Path(self.temp_dir.name)
        else:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Check utterances
        speakers_with_data: typing.Set[int] = set()

        for dataset in datasets:
            for utt_id in dataset.split_ids.get(split, []):
                audio_path = dataset.audio_dir / utt_id

                if not audio_path.is_file():
                    # Try WAV extension
                    audio_path = dataset.audio_dir / f"{utt_id}.wav"

                if audio_path.is_file():
                    cache_path = self.cache_dir / dataset.name / utt_id
                    speaker_id = dataset.utt_speaker_ids.get(utt_id)

                    if config.model.is_multispeaker:
                        assert speaker_id is not None, f"No speaker for {utt_id}"
                        speakers_with_data.add(speaker_id)

                    self.utterances.append(
                        Utterance(
                            id=utt_id,
                            phoneme_ids=dataset.utt_phoneme_ids[utt_id],
                            audio_path=audio_path,
                            cache_path=cache_path,
                            speaker_id=speaker_id,
                        )
                    )
                else:
                    _LOGGER.warning("Missing audio file: %s", audio_path)

        if config.model.is_multispeaker and (
            len(speakers_with_data) < config.model.n_speakers
        ):
            # Possilbly missing data
            _LOGGER.warning(
                "Data was found for only %s/%s speakers",
                len(speakers_with_data),
                config.model.n_speakers,
            )

    def __getitem__(self, index):
        utterance = self.utterances[index]

        # Normalized audio
        audio_norm_path = utterance.cache_path.with_suffix(".audio.pt")
        if audio_norm_path.is_file():
            # Load from cache
            audio_norm = torch.load(str(audio_norm_path))
        else:
            # Load audio and resample
            audio, _sample_rate = librosa.load(
                str(utterance.audio_path), sr=self.config.audio.sample_rate
            )

            # NOTE: audio is already in [-1, 1] coming from librosa
            audio_norm = torch.FloatTensor(audio).unsqueeze(0)

            # Save to cache
            audio_norm_path.parent.mkdir(parents=True, exist_ok=True)

            # Use temporary file to avoid multiple processes writing at the same time.
            with tempfile.NamedTemporaryFile(mode="wb") as audio_norm_file:
                torch.save(audio_norm, audio_norm_file.name)
                shutil.copy(audio_norm_file.name, audio_norm_path)

        # Mel spectrogram
        spectrogram_path = utterance.cache_path.with_suffix(".spec.pt")
        if spectrogram_path.is_file():
            # Load from cache
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

            # Save to cache
            spectrogram_path.parent.mkdir(parents=True, exist_ok=True)

            # Use temporary file to avoid multiple processes writing at the same time.
            with tempfile.NamedTemporaryFile(mode="wb") as spec_file:
                torch.save(spectrogram, spec_file.name)
                shutil.copy(spec_file.name, spectrogram_path)

        speaker_id = None
        if utterance.speaker_id is not None:
            speaker_id = torch.LongTensor([utterance.speaker_id])

        return UtteranceTensors(
            id=utterance.id,
            phoneme_ids=torch.LongTensor(utterance.phoneme_ids),
            audio_norm=audio_norm,
            spectrogram=spectrogram,
            spec_length=spectrogram.size(1),
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


# -----------------------------------------------------------------------------


def load_dataset(
    config: TrainingConfig,
    dataset_name: str,
    metadata_dir: typing.Union[str, Path],
    audio_dir: typing.Union[str, Path],
    splits=("train", "val"),
    speaker_id_map: typing.Optional[typing.Dict[str, int]] = None,
) -> DatasetInfo:
    metadata_dir = Path(metadata_dir)
    audio_dir = Path(audio_dir)

    # Determine data paths
    data_paths = defaultdict(dict)
    for split in splits:
        is_phonemes = False
        csv_path = metadata_dir / f"{split}_ids.csv"
        if not csv_path.is_file():
            csv_path = metadata_dir / f"{split}_phonemes.csv"
            is_phonemes = True

        data_paths[split]["is_phonemes"] = is_phonemes
        data_paths[split]["csv_path"] = csv_path
        data_paths[split]["utt_ids"] = []

    # train/val sets are required
    for split in splits:
        assert data_paths[split][
            "csv_path"
        ].is_file(), (
            f"Missing {split}_ids.csv or {split}_phonemes.csv in {metadata_dir}"
        )

    # Load phonemes
    phoneme_to_id = {}
    phonemes_path = metadata_dir / "phonemes.txt"

    _LOGGER.debug("Loading phonemes from %s", phonemes_path)
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

    id_to_phoneme = {i: p for p, i in phoneme_to_id.items()}

    # Load utterances
    utt_phoneme_ids: typing.Dict[str, str] = {}
    utt_speaker_ids: typing.Dict[str, int] = {}

    for split in splits:
        csv_path = data_paths[split]["csv_path"]
        if not csv_path.is_file():
            _LOGGER.debug("Skipping data for %s", split)
            continue

        is_phonemes = data_paths[split]["is_phonemes"]
        utt_ids = data_paths[split]["utt_ids"]

        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file, delimiter="|")
            for row_idx, row in enumerate(reader):
                assert len(row) > 1, f"{row} in {csv_path}:{row_idx+1}"
                utt_id, phonemes_or_ids = row[0], row[-1]

                if config.model.is_multispeaker:
                    if len(row) > 2:
                        speaker = row[1]
                    else:
                        speaker = dataset_name

                    if speaker not in speaker_id_map:
                        # Add to cross-datatset speaker id map
                        speaker_id_map[speaker] = len(speaker_id_map)

                    utt_speaker_ids[utt_id] = speaker_id_map[speaker]

                if is_phonemes:
                    # TODO: Map phonemes with phonemes2ids
                    raise NotImplementedError(csv_path)
                    # phoneme_ids = [phoneme_to_id[p] for p in phonemes if p in phoneme_to_id]
                    # phoneme_ids = intersperse(phoneme_ids, 0)
                else:
                    phoneme_ids = [int(p_id) for p_id in phonemes_or_ids.split()]
                    phoneme_ids = [
                        p_id for p_id in phoneme_ids if p_id in id_to_phoneme
                    ]

                if phoneme_ids:
                    utt_phoneme_ids[utt_id] = phoneme_ids
                    utt_ids.append(utt_id)
                else:
                    _LOGGER.warning("No phoneme ids for %s (%s)", utt_id, csv_path)

        _LOGGER.debug(
            "Loaded %s utterance(s) for %s from %s", len(utt_ids), split, csv_path
        )

    # Filter utterances based on min/max settings in config
    _LOGGER.debug("Filtering data")
    drop_utt_ids: typing.Set[str] = set()

    num_phonemes_too_small = 0
    num_phonemes_too_large = 0
    num_audio_missing = 0
    num_spec_too_small = 0
    num_spec_too_large = 0

    for utt_id, phoneme_ids in utt_phoneme_ids.items():
        # Check phonemes length
        if (config.min_seq_length is not None) and (
            len(phoneme_ids) < config.min_seq_length
        ):
            drop_utt_ids.add(utt_id)
            num_phonemes_too_small += 1
            continue

        if (config.max_seq_length is not None) and (
            len(phoneme_ids) > config.max_seq_length
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

        # Check estimated spectrogram length
        duration_sec = librosa.get_duration(filename=str(audio_path))
        num_samples = int(math.ceil(config.audio.sample_rate * duration_sec))
        spec_length = num_samples // config.audio.hop_length

        if (config.min_spec_length is not None) and (
            spec_length < config.min_spec_length
        ):
            drop_utt_ids.add(utt_id)
            num_spec_too_small += 1
            continue

        if (config.max_spec_length is not None) and (
            spec_length > config.max_spec_length
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
                config.min_seq_length,
            )

        if num_phonemes_too_large > 0:
            _LOGGER.debug(
                "%s utterance(s) dropped whose phoneme length was larger than %s",
                num_phonemes_too_large,
                config.max_seq_length,
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
                config.min_spec_length,
            )

        if num_spec_too_large > 0:
            _LOGGER.debug(
                "%s utterance(s) dropped whose spectrogram length was larger than %s",
                num_spec_too_large,
                config.max_spec_length,
            )

        utt_phoneme_ids = {
            utt_id: phoneme_ids
            for utt_id, phoneme_ids in utt_phoneme_ids.items()
            if utt_id not in drop_utt_ids
        }
    else:
        _LOGGER.info("Kept all %s utterances", len(utt_phoneme_ids))

    if not utt_phoneme_ids:
        _LOGGER.warning("No utterances after filtering")

    return DatasetInfo(
        name=dataset_name,
        audio_dir=audio_dir,
        utt_phoneme_ids=utt_phoneme_ids,
        utt_speaker_ids=utt_speaker_ids,
        split_ids={
            split: set(data_paths[split]["utt_ids"]) - drop_utt_ids for split in splits
        },
    )
