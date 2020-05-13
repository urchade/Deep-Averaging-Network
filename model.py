from torch import nn


class DAN(nn.Module):
    def __init__(self, n_emb, emb_dim, n_layers,
                 hidden_size, n_outputs, pad_idx=0):
        super().__init__()

        self.emb_dim = emb_dim

        self.emb = nn.Embedding(num_embeddings=n_emb, embedding_dim=emb_dim,
                                padding_idx=pad_idx)

        modules = []
        in_features = emb_dim

        for i in range(n_layers):
            modules.append(nn.Linear(in_features, hidden_size))
            modules.append(nn.SELU())
            in_features = hidden_size

        modules.append(nn.Linear(hidden_size, n_outputs))

        if n_outputs == 1:
            modules.append(nn.Sigmoid())
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        x = self.emb(x)
        x = x.mean(dim=0)  # Averaging
        x = self.layers(x.view(-1, self.emb_dim))
        return x
