from collections import Counter

import torch


class RNNValueExtractor(torch.nn.Module):
    def forward(self, x):
        assert type(x) == tuple
        return x[0]

class RNNHiddenExtractor(torch.nn.Module):
    def forward(self, x):
        assert type(x) == tuple
        return x[1]

class Identity(torch.nn.Module):
    def forward(self, x):
        return x

def bucketing_dataloader(ds, bucket_fn, **kwargs):
    """Calls ``bucket_fn`` on every batch and expects it to
    bucket the tensors by length. The 'X' tensor is formatted
    to fit the convolution of the DeepSpeech model.
    """
    dl = torch.utils.data.DataLoader(ds, **kwargs)
    for Xi, yi in dl:
        Xi, yi = bucket_fn(Xi, yi)

        # conv layer expects 2d frames
        Xi['X'] = Xi['X'][:, None, :, :]

        yield Xi, yi

def low_count_words(X, min_count=-1):
    for k, v in Counter([n[0] for n in X]).items():
        if v < min_count:
            yield k

def indices_by_word(X, words):
    for i, w in enumerate([n[0] for n in X]):
        if w in words:
            yield i

def filter_low_count_words(X, min_count=-1):
    """Return indices that satisfy given min. word count.
    """
    words = list(low_count_words(X, min_count))
    idcs = set(indices_by_word(X, words))
    mask = list(set(range(len(X))) - idcs)

    return mask
