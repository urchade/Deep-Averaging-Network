import torch
from rnn_classifier import RNNClassifier
from torch import nn

model = RNNClassifier(input_size=1, hidden_units=16, rnn_type='lstm', num_classes=1)

batch_size, seq_len, input_size = 16, 10, 1
loss_func = nn.MSELoss()

x = torch.rand(size=(batch_size, seq_len, input_size))
y = x.max(dim=1)[0]

y_pred = model(x)

loss = loss_func(y, y_pred)
