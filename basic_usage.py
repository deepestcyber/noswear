import numpy as np

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

from sys import argv

fpath = argv[1]
audio = parser.parse_audio(fpath)

X = {'lens': np.array([audio.shape[1]]), 'X': np.array(audio)[None]}
y_pred = net.predict(X)

print(y_pred[0] and 'swear! :(' or 'noswear :)')
