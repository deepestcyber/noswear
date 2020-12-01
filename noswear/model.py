from functools import partial

import skorch
import torch

from noswear.utils import Identity
from noswear.utils import bucketing_dataloader



def bucket(Xi, yi):
    Xi['X'] = Xi['X'][:, :, :max(Xi['lens'])]
    return Xi, yi


class NoSwearModel(torch.nn.Module):
    def __init__(
        self,
        base_model,
        n_hidden=10,
        n_layers=1,
        p_dropout=0.2,
        selector='last',
        inital_state_trainable=False,
    ):
        super().__init__()
        self.base_model = base_model
        self.base_model.rnns = Identity()
        self.base_model.lookahead = Identity()
        self.base_model.fc = Identity()
        self.base_model.inference_softmax = Identity()

        self.selector = selector

        self.rnn = torch.nn.LSTM(672, n_hidden, num_layers=n_layers, bias=False, batch_first=True)

        if inital_state_trainable:
            self.h0 = torch.nn.Parameter(torch.zeros(1, 1, n_hidden))
            self.c0 = torch.nn.Parameter(torch.zeros(1, 1, n_hidden))
        else:
            self.h0 = torch.zeros(1, 1, n_hidden)
            self.c0 = torch.zeros(1, 1, n_hidden)

        self.clf = torch.nn.Linear(n_hidden, 1, bias=False)
        self.dropout = torch.nn.Dropout(p=p_dropout)

        self.reset_parameters()

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.clf.reset_parameters()

    def forward(self, X, lens):
        # run base model, output is NxTxH with
        # T=Time, N=samples, H=hidden.
        y_pre = self.base_model(X)
        y_pre = self.dropout(y_pre)

        # run RNN over sequence and extract "last" item
        y, _ = self.rnn(y_pre, (
            self.h0.repeat(1, X.shape[0], 1).to(X),
            self.c0.repeat(1, X.shape[0], 1).to(X),
        ))

        if self.selector == 'designated':
            # instead of taking element -1 we just
            # designate neuron 0 to be our confidence indicator;
            # we take the element where [0] is greatest.
            indicator = y[:, :, 0]
            idcs_time = indicator.argmax(axis=-1)
            idcs_batch = list(range(X.shape[0]))

            # index using (batch, time) tuples
            y = y[idcs_batch, idcs_time, :]
        elif self.selector == 'designated_sum':
            ble = y
            indicator = y[:, :, [0]*100].sum(dim=-1)
            idcs_time = indicator.argmax(axis=-1)
            idcs_batch = list(range(X.shape[0]))

            y = y[idcs_batch, idcs_time, :]
        elif self.selector == 'designated_afew':
            # same as designated but look at at least N frames
            N = 10
            indicator = y[:, N:, 0]
            idcs_time = indicator.argmax(axis=-1)
            idcs_batch = list(range(X.shape[0]))

            y = y[idcs_batch, idcs_time, :]
        else:
            y = y[:, -1]

        y = self.clf(y)
        return y, indicator[idcs_batch, idcs_time], indicator


class BinaryClassifier(skorch.classifier.NeuralNetBinaryClassifier):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        return super().get_loss(y_pred[0], y_true, *args, **kwargs)


def get_net(base_model, device='cpu', **kwargs):
    net = BinaryClassifier(
        partial(NoSwearModel, base_model),

        iterator_train=bucketing_dataloader,
        iterator_train__bucket_fn=bucket,
        iterator_train__shuffle=True,
        iterator_valid=bucketing_dataloader,
        iterator_valid__bucket_fn=bucket,

        batch_size=4,
        max_epochs=80,
        device=device,

        train_split=predefined_split(Dataset(X_valid, y_valid)),

        module__p_dropout=0.0,
        module__n_hidden=32,
        module__n_layers=1,
        module__selector='designated_afew',

        optimizer=torch.optim.Adam,
        optimizer__lr=0.0004,

        callbacks=[
            skorch.callbacks.Freezer('base_model.*'),
            skorch.callbacks.Checkpoint(monitor='valid_acc_best'),
        ],
        **kwargs,
    )
    net.initialize()


def load_model(base_model, weights):
    net = get_net(base_model)
    net.load_params(weights)
    return net
