from cs336_systems.modal_utils import app, build_image
from cs336_systems.fsdp import FSDP








import os
import timeit
import subprocess
import argparse
from pathlib import Path

import modal
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.cuda.nvtx as nvtx
from cs336_basics.nn_utils import *
from cs336_basics.optimizer import *
from cs336_systems.create_model import *
from cs336_systems.modal_utils import app, build_image, user_volume
from cs336_systems.naive_ddp import DDP, NaiveDDP, OverlapDDP


XL_MODEL = ModelConfig(model_type="xl", d_model=2560, d_ff=10240, num_layers=32, num_heads=32, vocab_size=10000, context_length=512)
DEFAULT_NSYS_FLAGS = ["--trace=cuda,nvtx,osrt", "--force-overwrite=true"]


@app.function(
    image=build_image(include_tests=True),
    gpu="B200:2",
    timeout=2700,
    volumes={"/root/data": user_volume}
)
def test_fsdp():
    subprocess.run(["uv", "run", "pytest", "/root/tests/test_fsdp.py", "-x", "-v", "-s"], check=False)
    user_volume.commit()


@app.local_entrypoint()
def run_test():
    test_fsdp.remote()


