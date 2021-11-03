import torch


def to_gpu(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().cuda(non_blocking=True)
