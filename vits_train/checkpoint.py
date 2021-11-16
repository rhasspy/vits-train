"""Methods for saving/loading checkpoints"""
import logging
import typing
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.optim

from vits_train import setup_discriminator, setup_model
from vits_train.config import TrainingConfig
from vits_train.models import MultiPeriodDiscriminator, SynthesizerTrn

_LOGGER = logging.getLogger("vits_train.checkpoint")

# -----------------------------------------------------------------------------


@dataclass
class Checkpoint:
    model_g: SynthesizerTrn
    global_step: int
    epoch: int
    version: int
    best_loss: typing.Optional[float] = None
    model_d: typing.Optional[MultiPeriodDiscriminator] = None
    optimizer_g: typing.Optional[torch.optim.Optimizer] = None
    optimizer_d: typing.Optional[torch.optim.Optimizer] = None
    scheduler_g: typing.Optional[torch.optim.lr_scheduler._LRScheduler] = None
    scheduler_d: typing.Optional[torch.optim.lr_scheduler._LRScheduler] = None


def save_checkpoint(checkpoint: Checkpoint, checkpoint_path: Path):
    """Save model/optimizer/training state to a Torch checkpoint"""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model_g = checkpoint.model_g

    if hasattr(model_g, "module"):
        state_dict_g = model_g.module.state_dict()  # type: ignore
    else:
        state_dict_g = model_g.state_dict()

    checkpoint_dict = {
        "model_g": state_dict_g,
        "global_step": checkpoint.global_step,
        "epoch": checkpoint.epoch,
        "version": checkpoint.version,
        "best_loss": checkpoint.best_loss,
    }
    model_d = checkpoint.model_d

    if model_d is not None:
        if hasattr(model_d, "module"):
            state_dict_d = model_d.module.state_dict()  # type: ignore
        else:
            state_dict_d = model_d.state_dict()

        checkpoint_dict["model_d"] = state_dict_d

    model_d = checkpoint.model_d

    if model_d is not None:
        if hasattr(model_d, "module"):
            state_dict_d = model_d.module.state_dict()  # type: ignore
        else:
            state_dict_d = model_d.state_dict()

        checkpoint_dict["model_d"] = state_dict_d

    optimizer_g = checkpoint.optimizer_g

    if optimizer_g is not None:
        checkpoint_dict["optimizer_g"] = optimizer_g.state_dict()

    optimizer_d = checkpoint.optimizer_d

    if optimizer_d is not None:
        checkpoint_dict["optimizer_d"] = optimizer_d.state_dict()

    scheduler_g = checkpoint.scheduler_g

    if scheduler_g is not None:
        checkpoint_dict["scheduler_g"] = scheduler_g.state_dict()

    scheduler_d = checkpoint.scheduler_d

    if scheduler_d is not None:
        checkpoint_dict["scheduler_d"] = scheduler_d.state_dict()

    torch.save(checkpoint_dict, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    config: TrainingConfig,
    model_g: typing.Optional[SynthesizerTrn] = None,
    model_d: typing.Optional[MultiPeriodDiscriminator] = None,
    load_discrimiator: bool = True,
    optimizer_g: typing.Optional[torch.optim.Optimizer] = None,
    optimizer_d: typing.Optional[torch.optim.Optimizer] = None,
    scheduler_g: typing.Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scheduler_d: typing.Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    load_optimizers: bool = True,
    load_schedulers: bool = True,
    use_cuda: bool = True,
) -> Checkpoint:
    """Load model/optimizer/training state from a Torch checkpoint"""
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    version = int(checkpoint_dict.get("version", 1))
    global_step = int(checkpoint_dict.get("global_step", 1))
    epoch = int(checkpoint_dict.get("epoch", 1))

    best_loss = checkpoint_dict.get("best_loss")
    if best_loss is not None:
        best_loss = float(best_loss)

    # Create generator if necessary
    if model_g is None:
        model_g = setup_model(config, use_cuda=use_cuda)

    _load_state_dict(model_g, checkpoint_dict, "model_g")

    if load_discrimiator:
        if model_d is None:
            model_d = setup_discriminator(config, use_cuda=use_cuda)

        _load_state_dict(model_d, checkpoint_dict, "model_d")

    # Load optimizer states
    if load_optimizers:
        if optimizer_g is not None:
            optimizer_g.load_state_dict(checkpoint_dict["optimizer_g"])

        if optimizer_d is not None:
            optimizer_d.load_state_dict(checkpoint_dict["optimizer_d"])

    # Load scheduler states
    if load_schedulers:
        if scheduler_g is not None:
            scheduler_g.load_state_dict(checkpoint_dict["scheduler_g"])

        if scheduler_d is not None:
            scheduler_d.load_state_dict(checkpoint_dict["scheduler_d"])

    return Checkpoint(
        model_g=model_g,
        model_d=model_d,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
        global_step=global_step,
        epoch=epoch,
        version=version,
        best_loss=best_loss,
    )


def _load_state_dict(model, checkpoint_dict, key):
    saved_state_dict_g = checkpoint_dict[key]
    if hasattr(model, "module"):
        state_dict_g = model.module.state_dict()  # type: ignore
    else:
        state_dict_g = model.state_dict()

    new_state_dict_g = {}

    for k, v in state_dict_g.items():
        if k in saved_state_dict_g:
            # Use saved value
            new_state_dict_g[k] = saved_state_dict_g[k]
        else:
            # Use initialized value
            _LOGGER.warning("%s is not in the checkpoint for %s", k, key)
            new_state_dict_g[k] = v

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict_g)  # type: ignore
    else:
        model.load_state_dict(new_state_dict_g)
