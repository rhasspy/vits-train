import pytorch_lightning as pl
import torch

from vits_train.config import TrainingConfig
from vits_train.models import MultiPeriodDiscriminator, SynthesizerTrn


class VITSModel(pl.LightningModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()

        self.config = config

        self.net_g = SynthesizerTrn(
            n_vocab=config.model.num_symbols,
            spec_channels=config.audio.filter_length // 2 + 1,
            segment_size=config.model.segment_size // config.audio.hop_length,
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
        )
        self.net_d = MultiPeriodDiscriminator(config.model.use_spectral_norm)

    def forward(self):
        pass

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

    def training_step(self, train_batch, batch_idx):
        pass

    def validation_step(self, val_batch, batch_idx):
        pass
