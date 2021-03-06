"""Configuration classes"""
import collections
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
from dataclasses_json import DataClassJsonMixin
from phonemes2ids import STRESS, BlankBetween


@dataclass
class AudioConfig(DataClassJsonMixin):
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    mel_channels: int = 80
    sample_rate: int = 22050
    sample_bytes: int = 2
    channels: int = 1
    mel_fmin: float = 0.0
    mel_fmax: typing.Optional[float] = None
    ref_level_db: float = 20.0
    spec_gain: float = 1.0

    # Normalization
    signal_norm: bool = True
    min_level_db: float = -100.0
    max_norm: float = 1.0
    clip_norm: bool = True
    symmetric_norm: bool = True
    do_dynamic_range_compression: bool = True
    convert_db_to_amp: bool = True

    do_trim_silence: bool = False
    trim_silence_db: float = 60.0

    def __post_init__(self):
        if self.mel_fmax is not None:
            assert self.mel_fmax <= self.sample_rate // 2

        # Compute mel bases
        self._mel_basis = librosa.filters.mel(
            self.sample_rate,
            self.filter_length,
            n_mels=self.mel_channels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
        )

        self._inv_mel_basis = np.linalg.pinv(self._mel_basis)

    # -------------------------------------------------------------------------
    # Mel Spectrogram
    # -------------------------------------------------------------------------

    def wav2mel(self, wav: np.ndarray) -> np.ndarray:
        if self.do_trim_silence:
            wav = self.trim_silence(wav, trim_db=self.trim_silence_db)

        linear = self.stft(wav)
        mel_amp = self.linear_to_mel(np.abs(linear))
        mel_db = self.amp_to_db(mel_amp)

        if self.signal_norm:
            mel_db = self.normalize(mel_db)

        return mel_db

    def mel2wav(
        self, mel_db: np.ndarray, num_iters: int = 60, power: float = 1.0
    ) -> np.ndarray:
        """Converts melspectrogram to waveform using Griffim-Lim"""
        if self.signal_norm:
            mel_db = self.denormalize(mel_db)

        mel_amp = self.db_to_amp(mel_db)
        linear = self.mel_to_linear(mel_amp) ** power

        return self.griffin_lim(linear, num_iters=num_iters)

    def linear_to_mel(self, linear: np.ndarray) -> np.ndarray:
        """Linear spectrogram to mel amp"""
        return np.dot(self._mel_basis, linear)

    def mel_to_linear(
        self, mel_amp: np.ndarray, threshold: float = 1e-10
    ) -> np.ndarray:
        """Mel amp to linear spectrogram"""
        return np.maximum(threshold, np.dot(self._inv_mel_basis, mel_amp))

    def amp_to_db(self, mel_amp: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
        return self.spec_gain * np.log10(np.maximum(threshold, mel_amp))

    def db_to_amp(self, mel_db: np.ndarray, power: float = 10.0) -> np.ndarray:
        return np.power(power, mel_db / self.spec_gain)

    # -------------------------------------------------------------------------
    # STFT
    # -------------------------------------------------------------------------

    def stft(self, wav: np.ndarray) -> np.ndarray:
        """Waveform to linear spectrogram"""
        return librosa.stft(
            y=wav,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode="reflect",
        )

    def istft(self, linear: np.ndarray) -> np.ndarray:
        """Linear spectrogram to waveform"""
        return librosa.istft(
            linear, hop_length=self.hop_length, win_length=self.win_length
        )

    def griffin_lim(self, linear: np.ndarray, num_iters: int = 60) -> np.ndarray:
        """Linear spectrogram to waveform using Griffin-Lim"""
        angles = np.exp(2j * np.pi * np.random.rand(*linear.shape))
        linear_complex = np.abs(linear).astype(np.complex)
        audio = self.istft(linear_complex * angles)

        for _ in range(num_iters):
            angles = np.exp(1j * np.angle(self.stft(audio)))
            audio = self.istft(linear_complex * angles)

        return audio

    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------

    def normalize(self, mel_db: np.ndarray) -> np.ndarray:
        """Put values in [0, max_norm] or [-max_norm, max_norm]"""
        mel_norm = ((mel_db - self.ref_level_db) - self.min_level_db) / (
            -self.min_level_db
        )
        if self.symmetric_norm:
            # Symmetric norm
            mel_norm = ((2 * self.max_norm) * mel_norm) - self.max_norm
            if self.clip_norm:
                mel_norm = np.clip(mel_norm, -self.max_norm, self.max_norm)
        else:
            # Asymmetric norm
            mel_norm = self.max_norm * mel_norm
            if self.clip_norm:
                mel_norm = np.clip(mel_norm, 0, self.max_norm)

        return mel_norm

    def denormalize(self, mel_db: np.ndarray) -> np.ndarray:
        """Pull values out of [0, max_norm] or [-max_norm, max_norm]"""
        if self.symmetric_norm:
            # Symmetric norm
            if self.clip_norm:
                mel_denorm = np.clip(mel_db, -self.max_norm, self.max_norm)

            mel_denorm = (
                (mel_denorm + self.max_norm) * -self.min_level_db / (2 * self.max_norm)
            ) + self.min_level_db
        else:
            # Asymmetric norm
            if self.clip_norm:
                mel_denorm = np.clip(mel_db, 0, self.max_norm)

            mel_denorm = (
                mel_denorm * -self.min_level_db / self.max_norm
            ) + self.min_level_db

        mel_denorm += self.ref_level_db

        return mel_denorm

    # -------------------------------------------------------------------------
    # Silence Trimming
    # -------------------------------------------------------------------------

    def trim_silence(
        self,
        wav: np.ndarray,
        trim_db: float = 60.0,
        margin_sec: float = 0.01,
        keep_sec: float = 0.1,
    ):
        """
        Trim silent parts with a threshold and margin.
        Keep keep_sec seconds on either side of trimmed audio.
        """
        margin = int(self.sample_rate * margin_sec)
        wav = wav[margin:-margin]
        _, trim_index = librosa.effects.trim(
            wav,
            top_db=trim_db,
            frame_length=self.win_length,
            hop_length=self.hop_length,
        )

        keep_samples = int(self.sample_rate * keep_sec)
        trim_start, trim_end = (
            max(0, trim_index[0] - keep_samples),
            min(len(wav), trim_index[1] + keep_samples),
        )

        return wav[trim_start:trim_end]


@dataclass
class ModelConfig(DataClassJsonMixin):
    num_symbols: int = 0
    n_speakers: int = 1

    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock: str = "1"
    resblock_kernel_sizes: typing.Tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: typing.Tuple[typing.Tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    upsample_rates: typing.Tuple[int, ...] = (8, 8, 2, 2)
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: typing.Tuple[int, ...] = (16, 16, 4, 4)
    n_layers_q: int = 3
    use_spectral_norm: bool = False
    gin_channels: int = 256
    use_sdp: bool = True  # StochasticDurationPredictor

    @property
    def is_multispeaker(self) -> bool:
        return self.n_speakers > 1


@dataclass
class PhonemesConfig(DataClassJsonMixin):
    phoneme_separator: str = " "
    """Separator between individual phonemes in CSV input"""

    word_separator: str = "#"
    """Separator between word phonemes in CSV input (must not match phoneme_separator)"""

    phoneme_to_id: typing.Optional[typing.Mapping[str, int]] = None
    pad: typing.Optional[str] = "_"
    bos: typing.Optional[str] = None
    eos: typing.Optional[str] = None
    blank: typing.Optional[str] = "#"
    blank_word: typing.Optional[str] = None
    blank_between: typing.Union[str, BlankBetween] = BlankBetween.WORDS
    blank_at_start: bool = True
    blank_at_end: bool = True
    simple_punctuation: bool = True
    punctuation_map: typing.Optional[typing.Mapping[str, str]] = None
    separate: typing.Optional[typing.Collection[str]] = STRESS
    separate_graphemes: bool = False
    separate_tones: bool = False
    tone_before: bool = False

    def split_word_phonemes(self, phonemes_str: str) -> typing.List[typing.List[str]]:
        """Split phonemes string into a list of lists (outer is words, inner is individual phonemes in each word)"""
        return [
            word_phonemes_str.split(self.phoneme_separator)
            for word_phonemes_str in phonemes_str.split(self.word_separator)
        ]


@dataclass
class TrainingConfig(DataClassJsonMixin):
    seed: int = 1234
    epochs: int = 10000
    learning_rate: float = 2e-4
    betas: typing.Tuple[float, float] = field(default=(0.8, 0.99))
    eps: float = 1e-9
    batch_size: int = 32
    fp16_run: bool = False
    lr_decay: float = 0.999875
    segment_size: int = 8192
    init_lr_ratio: float = 1.0
    warmup_epochs: int = 0
    c_mel: int = 45
    c_kl: float = 1.0
    grad_clip: typing.Optional[float] = None

    min_seq_length: typing.Optional[int] = None
    max_seq_length: typing.Optional[int] = None

    min_spec_length: typing.Optional[int] = None
    max_spec_length: typing.Optional[int] = None

    last_epoch: int = 1
    global_step: int = 1
    best_loss: typing.Optional[float] = None
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    phonemes: PhonemesConfig = field(default_factory=PhonemesConfig)
    version: int = 1
    git_commit: str = ""

    def save(self, config_file: typing.TextIO):
        """Save config as JSON to a file"""
        json.dump(self.to_dict(), config_file, indent=4)

    @staticmethod
    def load(config_file: typing.TextIO) -> "TrainingConfig":
        """Load config from a JSON file"""
        return TrainingConfig.from_json(config_file.read())

    @staticmethod
    def load_and_merge(
        config: "TrainingConfig",
        config_files: typing.Iterable[typing.Union[str, Path, typing.TextIO]],
    ) -> "TrainingConfig":
        """Loads one or more JSON configuration files and overlays them on top of an existing config"""
        base_dict = config.to_dict()
        for maybe_config_file in config_files:
            if isinstance(maybe_config_file, (str, Path)):
                # File path
                config_file = open(maybe_config_file, "r", encoding="utf-8")
            else:
                # File object
                config_file = maybe_config_file

            with config_file:
                # Load new config and overlay on existing config
                new_dict = json.load(config_file)
                TrainingConfig.recursive_update(base_dict, new_dict)

        return TrainingConfig.from_dict(base_dict)

    @staticmethod
    def recursive_update(
        base_dict: typing.Dict[typing.Any, typing.Any],
        new_dict: typing.Mapping[typing.Any, typing.Any],
    ) -> None:
        """Recursively overwrites values in base dictionary with values from new dictionary"""
        for key, value in new_dict.items():
            if isinstance(value, collections.Mapping) and (
                base_dict.get(key) is not None
            ):
                TrainingConfig.recursive_update(base_dict[key], value)
            else:
                base_dict[key] = value
