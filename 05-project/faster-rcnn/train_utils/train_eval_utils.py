import math
import sys
import time

import torch
import torch.nn as nn

def train_one_epoch(
    model: nn.Module,
    optimizer,
    data_loader,
    device,
    epoch,
    print_freq,
    warmup=False
):
    model.train()