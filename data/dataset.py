import os
from itertools import cycle


DEFAULT_PROVIDERS = {
    'dictcc': {
        'path': 'data/distcc/download/',
        'format': 'mp3',
    },
    'forvo': {
        'path': 'data/forvo/download/',
        'format': 'mp3',
    },
    'meariamwebster': {
        'path': 'data/meriamwebster/download/',
        'format': 'wav',
    },
}


class SwearDataset:
    """Low-level dataset. Provides path to sound files (X) and
    corresponding label (y).
    """
    def __init__(self, providers, good_word_path='data/good_words.txt',
            bad_word_path='data/bad_words.txt'):
        self.good_word_path = good_word_path
        self.bad_word_path = bad_word_path
        self.providers = providers

    def words_from_file(self, path):
        with open(path) as f:
            for line in f:
                if not line.strip() or line.startswith('#'):
                    continue
                yield line.strip()

    def retrieve_words(self, word, provider):
        path = provider['path']
        fmt = provider['format']

        for i in range(999):
            fpath = os.path.join(path, '{word}_{n}.{fmt}'.format(
			word=word, n=i+1, fmt=fmt))
            if not os.path.exists(fpath):
                break
            yield fpath

    def table(self):
        good_words = self.words_from_file(self.good_word_path)
        good_words = zip(good_words, cycle([False]))

        bad_words = self.words_from_file(self.bad_word_path)
        bad_words = zip(bad_words, cycle([True]))

        for word, goodness in [*good_words, *bad_words]:
            for provider in self.providers.values():
                for file in self.retrieve_words(word, provider):
                    yield goodness, word, file

    def load(self):
        table = list(self.table())

        X = [n[1:] for n in table]
        y = [n[0] for n in table]

        return X, y



class SwearBinaryAudioDataset:
    """Provides audio frames and a binary label (swear/noswear).
    """
    def __init__(self, X, y, parser):
        self.X = X
        self.y = y
        assert len(X) == len(y)
        self.parser = parser

    def table(self):
        for i, (word, fpath) in enumerate(self.X):
            frames = self.parser.parse_audio(fpath)
            yield frames, int(self.y[i])

    def load(self):
        Xy = list(self.table())
        return (
            [n[0] for n in Xy],
            [n[1] for n in Xy],
        )
