#!/usr/bin/env python3
import sys

import torch

assert len(sys.argv) == 3, "IN OUT"
c = torch.load(sys.argv[1], map_location="cpu")

model = {}

for key, value in c["state_dict"].items():
    if key.startswith("net_g."):
        model[key[6:]] = value

torch.save({"model": model}, sys.argv[2])
