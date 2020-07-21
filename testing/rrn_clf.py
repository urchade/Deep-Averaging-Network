import torch
from rnn_classifier import RNNClassifier
from torch import nn
import matplotlib.pyplot as plt
from data import data_rnn


model = RNNClassifier(input_size=1, hidden_units=16, rnn_type='lstm', num_classes=1)
data_loader = data_rnn(1, 20, 32)

crit = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.001)


def train(num_epoch=10, data=data_loader,
          optimizer=opt, criterion=crit):
    losses = []
    for i, (x, y) in enumerate(data):
        y_ = model(x)
        loss = criterion(y, y_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss.item())
        losses.append(loss.item())
        if i == num_epoch:
            break
    return losses


losses = train(10)

plt.plot(losses)
plt.show()
