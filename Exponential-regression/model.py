import torch
from torch import nn


class ExpM(nn.Module):
    def __init__(self):
        super().__init__()

        self.a = nn.Parameter(torch.randn(size=(1, 1)))

    def forward(self, x):
        return torch.exp_(x @ self.a)


func = lambda x: torch.exp_(3.7 * x)


def generate():
    while True:
        X = torch.randn(size=(32, 1))
        y = func(X)
        yield X, y

data = generate()
model = ExpM()

opt = torch.optim.SGD(model.parameters(), lr=1e-5)

for i, (x, y) in enumerate(data):
    y_ = model(x)
    loss = (y-y_).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss)
    print((model.a))
    if i==50:
        break

model.b