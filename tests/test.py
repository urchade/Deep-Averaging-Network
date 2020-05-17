from model import DAN
from torch import nn
import torch

input = torch.randint(100, size=(2, 23)).reshape(2, 23)

model = DAN(n_emb=100, emb_dim=10, n_layers=10, hidden_size=16, n_outputs=3)

model(input)