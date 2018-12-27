
from deepspeech.model import DeepSpeech
from deepspeech.data.data_loader import SpectrogramParser

from model import load_model


base_model = DeepSpeech.load_model(
    'models/librispeech_pretrained.pth'
)
audio_conf = DeepSpeech.get_audio_conf(base_model)
parser = SpectrogramParser(audio_conf)


net = load_model(base_model, 'binary_clf.pt')

print(net)

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from skorch.helper import SliceDict

from utils import filter_low_count_words
from data import dataset

ds_swear = dataset.SwearDataset(dataset.DEFAULT_PROVIDERS)
X_swear, y_swear = ds_swear.load()
ds = dataset.SwearBinaryAudioDataset(X_swear, y_swear, parser)
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

idcs = filter_low_count_words(X_swear, min_count=4)
X_pad = X_pad[idcs]
y = y[idcs]

split = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# use word as class for stratified split to make sure that train/test
# set both contain examples of the all words.
y_word = np.array([n[0] for n in X_swear])[idcs]
train_idcs_proto, test_idcs = next(split.split(y, y=y_word))

train_idcs, valid_idcs = next(split.split(y[train_idcs_proto], y=y_word[train_idcs_proto]))


train_idcs = train_idcs_proto[train_idcs]
valid_idcs = train_idcs_proto[valid_idcs]

X_train = {'lens': seq_lens[train_idcs], 'X': X_pad[train_idcs]}
y_train = np.array(y)[train_idcs]

X_valid = {'lens': seq_lens[valid_idcs], 'X': X_pad[valid_idcs]}
y_valid = np.array(y)[valid_idcs]

X_test = {'lens': seq_lens[test_idcs], 'X': X_pad[test_idcs]}
y_test = np.array(y)[test_idcs]




print(accuracy_score(y_valid[:3], net.predict(SliceDict(**X_valid)[:3])))
