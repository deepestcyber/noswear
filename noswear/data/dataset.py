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

    Expects all given paths to be relative to a certain base path to
    enforce basically the following directory structure:

    base/
     |
     `- data/
         |
         `- good_words.txt
         |
         `- bad_words.txt
         |
         `- providerA
               |
               `- sample1.wav

    The base_path would be `base` in this case.
    """
    def __init__(self, base_path, providers, good_word_path='data/good_words.txt',
            bad_word_path='data/bad_words.txt'):
        self.good_word_path = os.path.join(base_path, good_word_path)
        self.bad_word_path = os.path.join(base_path, bad_word_path)
        self.providers = providers
        for provider in self.providers:
            self.providers[provider]['path'] = os.path.join(
                base_path,
                self.providers[provider]['path'],
            )

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


def remove_common_prefix(a, b):
    """remove the common prefix from b.
    Useful when removing a common prefix between strings a and b
    such as common directory names (/tmp/foo/a, /tmp/foo/b).
    """
    i = 0
    for ca, cb in zip(a, b):
        i += 1
        if ca != cb:
            i -= 1
            break
    return b[i:]


class SOXSampler:
    """Audio down-sampler using file-system as cache."""
    def __init__(self, sample_rate, directory='./sampler_cache'):
        self.sample_rate = int(sample_rate)
        self.directory = directory

    def insert_suffix(self, fname, suffix):
        exts = os.path.splitext(fname)
        return ''.join(exts[:-1] + (suffix,) + exts[-1:])

    def new_path(self, fpath):
        cache_dir = os.path.abspath(self.directory)
        source_dir = os.path.dirname(fpath)
        source_dir = remove_common_prefix(cache_dir, source_dir)
        source_dir = source_dir.lstrip(os.path.sep)

        return os.path.join(
            os.path.abspath(self.directory),
            source_dir,
            self.insert_suffix(os.path.basename(fpath), f".{self.sample_rate}"),
        )

    def downsample_using_sox(self, fpath, sample_rate):
        """Change sample to the given sample rate and return the new
        (temporary) file.
        """
        fpath = os.path.abspath(fpath)
        fpath_ds = self.new_path(fpath)
        dpath_ds = os.path.dirname(fpath_ds)

        if not os.path.exists(dpath_ds):
            os.makedirs(dpath_ds)

        if os.path.exists(fpath_ds):
            return fpath_ds

        sox_params = "sox \"{f_in}\" -r {sr} -c 1 -b 16 -e si {f_out} >/dev/null 2>&1".format(
            f_in=fpath,
            sr=sample_rate,
            f_out=fpath_ds,
        )
        os.system(sox_params)
        return fpath_ds

    def downsample(self, fpath):
        return self.downsample_using_sox(fpath, self.sample_rate)


class SwearBinaryAudioDataset:
    """Provides audio frames and a binary label (swear/noswear).
    """
    def __init__(self, X, y, parser, sampler):
        self.X = X
        self.y = y
        assert len(X) == len(y)
        self.parser = parser
        self.sampler = sampler

    def table(self):
        for i, (word, fpath) in enumerate(self.X):
            if self.sampler:
                fpath = self.sampler.downsample(fpath)
            frames = self.parser.parse_audio(fpath)
            yield frames, int(self.y[i])

    def load(self):
        Xy = list(self.table())
        return (
            [n[0] for n in Xy],
            [n[1] for n in Xy],
        )
