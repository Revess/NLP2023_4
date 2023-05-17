import torch
import torch.nn as nn
import random

class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("Drop-out probability is lower than zero or higher than one")
        self.p = p

    def forward(self, x):
        for i in range(0,len(x)):
            if random.random() < self.p:
                x[0] = 0.0


