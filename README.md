# Contents

- models: contains pre-trained models, such as the librispeech
  	      DeepSpeech.pytorch model

- examples: basic usage examples

- noswear: python software package where all the magic happens


## Installation

Expected: debian based OS (preferably Ubuntu, tested on 18.04).

	sudo apt install libsox3 libsox-fmt-all libsox-dev
	sudo apt install libasound2-dev

	virtualenv -p python3 ~/envs/noswear
	. ~/envs/noswear/bin/activate

	git clone https://github.com/deepestcyber/DeepSpeech.pytorch -b 35c3
	cd DeepSpeech.pytorch && python setup.py install

	pip install -r requirements.txt
	python setup.py install


## Training

The current model is still experimental so the most detail is
in `notebooks/binary_recognition_binclass.ipynb` but there is
also a training script in `examples/train_binary.py` which
will produce a trained model.
