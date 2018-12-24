import torch

class ResidualRNN(torch.nn.Module):
    def __init__(self, rnn, n_layers=1):
        super().__init__()
        self.rnn = rnn
    def forward(self, x):
        return self.rnn(x)[0] + x
