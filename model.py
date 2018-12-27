from functools import partial

import skorch
import torch

from utils import Identity
from utils import bucketing_dataloader


def bucket(Xi, yi):
    Xi['X'] = Xi['X'][:, :, :max(Xi['lens'])]
    return Xi, yi


class NoSwearModel(torch.nn.Module):
    def __init__(
        self,
        base_model,
        n_hidden=10,
        n_layer=1,
        p_dropout=0.2
    ):
        super().__init__()
        self.base_model = base_model
        self.base_model.rnns = Identity()
        self.base_model.lookahead = Identity()
        self.base_model.fc = Identity()
        self.base_model.inference_softmax = Identity()

        self.rnn = torch.nn.GRU(672, n_hidden, num_layers=n_layer, bias=False, batch_first=True)

        self.clf = torch.nn.Linear(n_hidden, 2, bias=False)
        self.dropout = torch.nn.Dropout(p=p_dropout)

    def forward_step(self, X, lens, h=None):
        # run base model, output is NxTxH with
        # T=Time, N=samples, H=hidden.
        y_pre = self.base_model(X)
        y_pre = self.dropout(y_pre)

        # run RNN over sequence and extract last item
        y, h = self.rnn(y_pre, h)

        # pick frame ith highest activation
        i = y.mean(-1).argmax(1)
        y = y[torch.arange(len(y)), i]

        y = self.dropout(y)

        # run classifier
        y = self.clf(y)
        y = torch.softmax(y, dim=-1)
        return y, h

    def forward(self, X, lens):
        y, _ = self.forward_step(X, lens)
        return y


def get_net(base_model, device='cpu', **kwargs):
    net = skorch.NeuralNetClassifier(
        partial(NoSwearModel, base_model),

        iterator_train=bucketing_dataloader,
        iterator_train__bucket_fn=bucket,
        iterator_valid=bucketing_dataloader,
        iterator_valid__bucket_fn=bucket,

        batch_size=64,
        max_epochs=40,
        device=device,

        module__p_dropout=0.2,
        module__n_hidden=300,

        optimizer=torch.optim.Adam,
        optimizer__lr=0.0004,

        callbacks=[
            skorch.callbacks.Freezer('base_model.*'),
        ],

        **kwargs,
    )
    net.initialize()
    return net


def load_model(base_model, weights):
    net = get_net(base_model)
    net.load_params(weights)
    return net
