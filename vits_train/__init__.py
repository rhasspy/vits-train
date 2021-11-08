#!/usr/bin/env python3

from vits_train.config import TrainingConfig
from vits_train.models import MultiPeriodDiscriminator, SynthesizerTrn


def setup_model(config: TrainingConfig, use_cuda: bool = True) -> SynthesizerTrn:
    """Set up a synthesizer from a training configuration"""

    model = SynthesizerTrn(
        n_vocab=config.model.num_symbols,
        spec_channels=config.audio.filter_length // 2 + 1,
        segment_size=config.segment_size // config.audio.hop_length,
        inter_channels=config.model.inter_channels,
        hidden_channels=config.model.hidden_channels,
        filter_channels=config.model.filter_channels,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        kernel_size=config.model.kernel_size,
        p_dropout=config.model.p_dropout,
        resblock=config.model.resblock,
        resblock_kernel_sizes=config.model.resblock_kernel_sizes,
        resblock_dilation_sizes=config.model.resblock_dilation_sizes,
        upsample_rates=config.model.upsample_rates,
        upsample_initial_channel=config.model.upsample_initial_channel,
        upsample_kernel_sizes=config.model.upsample_kernel_sizes,
        n_speakers=config.model.n_speakers,
        gin_channels=config.model.gin_channels,
        use_sdp=config.model.use_sdp,
    )

    if use_cuda:
        model.cuda()

    return model


def setup_discriminator(
    config: TrainingConfig, use_cuda: bool = True
) -> MultiPeriodDiscriminator:
    model = MultiPeriodDiscriminator(use_spectral_norm=config.model.use_spectral_norm)
    if use_cuda:
        model.cuda()

    return model
