import torch


def data_rnn(input_size=1, seq_length=20, batch_size=1):
    while True:
        x = torch.randn(size=(input_size, seq_length, batch_size)).reshape(batch_size, seq_length, input_size)
        y = x.max(dim=1)[0]
        yield x, y
