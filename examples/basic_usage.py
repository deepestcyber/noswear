import warnings

warnings.filterwarnings("ignore", message='"sox" backend is being deprecated. The default backend will be changed to "sox_io" backend in 0.8.0 and "sox" backend will be removed in 0.9.0. Please migrate to "sox_io" backend. Please refer to https://github.com/pytorch/audio/issues/903 for the detail.')


import argparse

import numpy as np

from deepspeech.model import DeepSpeech
from deepspeech.data.data_loader import SpectrogramParser

from noswear.model import load_model




parser = argparse.ArgumentParser()
parser.add_argument('audio_file', type=argparse.FileType('r'),
    help='File to classify')

args = parser.parse_args()


base_model = DeepSpeech.load_model(
    'models/librispeech_pretrained.pth'
)
audio_conf = DeepSpeech.get_audio_conf(base_model)
parser = SpectrogramParser(audio_conf, normalize=True)


net = load_model(base_model, 'models/binary_clf.pt')
#print(net)

fpath = args.audio_file.name
audio = parser.parse_audio(fpath)

X = {'lens': np.array([audio.shape[0]]), 'X': np.array(audio)[None]}
y_pred = net.predict(X)

print(y_pred[0] and 'swear! :(' or 'noswear :)')
