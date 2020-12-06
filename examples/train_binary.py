#!/usr/bin/env python
# coding: utf-8
from functools import partial

import numpy as np
from matplotlib import pyplot as plt
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import skorch
from skorch.helper import predefined_split
from skorch.dataset import Dataset

from deepspeech.model import DeepSpeech
from deepspeech.model import SequenceWise
from deepspeech.data.data_loader import SpectrogramParser
from deepspeech.data.data_loader import BucketingSampler

from noswear.data import dataset
from noswear.utils import Identity
from noswear.utils import bucketing_dataloader
from noswear.utils import filter_low_count_words
from noswear.model import bucket
from noswear.model import BinaryClassifier
from noswear.model import NoSwearModel


# We are building upon DeepSpeech CNN layers.
base_model = DeepSpeech.load_model(
    '../models/librispeech_pretrained.pth'
)
audio_conf = DeepSpeech.get_audio_conf(base_model)
parser = SpectrogramParser(audio_conf, normalize=True)
sampler = dataset.SOXSampler(sample_rate=audio_conf['sample_rate'])

# We have our own swear/non-swear data, load it.
ds_swear = dataset.SwearDataset(base_path='../', providers=dataset.DEFAULT_PROVIDERS)
X_swear, y_swear = ds_swear.load()
ds = dataset.SwearBinaryAudioDataset(X_swear, y_swear, parser, sampler)
X, y = ds.load()


seq_lens = np.array([x.shape[1] for x in X])
max_seq_len = max(seq_lens)
max_seq_len, np.mean(seq_lens), np.median(seq_lens)


X_pad = np.zeros(
    (len(X), X[0].shape[0], max_seq_len),
    dtype='float32'
)
for i, _ in enumerate(X):
    X_pad[i, :, :seq_lens[i]] = X[i]

y = np.array(y)

# Filter low count words
idcs = filter_low_count_words(X_swear, min_count=4)
X_pad = X_pad[idcs]
y = y[idcs]

# Splitting into train/valid/test
split = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# use word as class for stratified split to make sure that train/test
# set both contain examples of the all words.
y_word = np.array([n[0] for n in X_swear])[idcs]
train_idcs_proto, test_idcs = next(split.split(y, y=y_word))
train_idcs, valid_idcs = next(split.split(y[train_idcs_proto], y=y_word[train_idcs_proto]))

train_idcs = train_idcs_proto[train_idcs]
valid_idcs = train_idcs_proto[valid_idcs]

X_train = {'lens': seq_lens[train_idcs], 'X': X_pad[train_idcs]}
y_train = np.array(y)[train_idcs].astype('float32')

X_valid = {'lens': seq_lens[valid_idcs], 'X': X_pad[valid_idcs]}
y_valid = np.array(y)[valid_idcs].astype('float32')

X_test = {'lens': seq_lens[test_idcs], 'X': X_pad[test_idcs]}
y_test = np.array(y)[test_idcs].astype('float32')


print(len(X_train['X']), len(X_valid['X']), len(X_test['X']))

# Check if train/valid/test are balanced ($y \approx 0.5$)
print(y_train.mean(), y_valid.mean(), y_test.mean())


# Training loop and stuff
torch.manual_seed(42)

net = BinaryClassifier(
    partial(NoSwearModel, base_model),

    iterator_train=bucketing_dataloader,
    iterator_train__bucket_fn=bucket,
    iterator_train__shuffle=True,
    iterator_valid=bucketing_dataloader,
    iterator_valid__bucket_fn=bucket,

    batch_size=4,
    max_epochs=30,
    device='cuda',

    train_split=predefined_split(Dataset(X_valid, y_valid)),

    module__p_dropout=0.0,
    module__n_hidden=48,
    module__n_layers=1,
    module__selector='designated_afew',

    optimizer=torch.optim.Adam,
    optimizer__lr=0.0002,

    callbacks=[
        skorch.callbacks.Freezer('base_model.*'),
        skorch.callbacks.Checkpoint(
            monitor='valid_acc_best',
            f_pickle='model_valid_acc_best.pkl'),
        #skorch.callbacks.TrainEndCheckpoint(),
    ]
)

net.fit(X_train, y_train)

# Training done, let's look at the accuracy of the best net (via valid loss)
checkpoint_cb = dict(net.callbacks_)['Checkpoint']
net.load_params(checkpoint=checkpoint_cb)

print(accuracy_score(y_valid, net.predict(X_valid)))
print(accuracy_score(y_test, net.predict(X_test)))

print(f"Model training done, result is in {checkpoint_cb.f_pickle}")
