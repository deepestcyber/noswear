
- capture.py: audio capture process
- worker.py: classifies speech captured via capture.capture

- models: contains pre-trained models, such as the librispeech
  	      DeepSpeech.pytorch model

## Installation

Expected: debian based OS (preferably Ubuntu, tested on 18.04).

	sudo apt install libsox3 libsox-fmt-all libsox-dev

	git clone https://github.com/pytorch/audio.git torchaudio
	cd torchaudio && python setup.py install

	git clone https://github.com/deepestcyber/DeepSpeech.pytorch -b 35c3
	cd DeepSpeech.pytorch && python setup.py install
