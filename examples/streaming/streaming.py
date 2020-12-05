import numpy as np
import librosa
import alsaaudio
import wave


def compute_spect(
    y,
    audio_conf,
    eps=1e-6,
    normalize=True,
):
    sample_rate = audio_conf['sample_rate']
    window_stride = audio_conf['window_stride']
    window_size = audio_conf['window_size']
    window = audio_conf['window']

    n_fft = int(sample_rate * window_size)
    win_length = n_fft
    hop_length = int(sample_rate * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, _phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)
    if normalize:
        mean = spect.mean()
        std = spect.std()
        spect -= mean
        spect /= std + eps
    return spect


def wav_file_handle(path: str, sample_rate: float):
    f = wave.open(path, mode='wb')
    f.setparams((1, 2, sample_rate, 0, 'NONE', 'not compressed'))
    return f


def capture(audio_conf, queue, verbose, debug=False):
    sample_rate = audio_conf['sample_rate']
    window_size_abs = int(audio_conf['window_size'] * sample_rate)
    write_to_file = False
    period_size = 20 * window_size_abs

    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)

    # TODO: if we have more, take the average
    inp.setchannels(1)
    inp.setrate(audio_conf['sample_rate'])
    inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)

    inp.setperiodsize(period_size)

    if write_to_file:
        wav_file = wav_file_handle(
            'debug_input.wav',
            sample_rate,
        )

    step = 0
    while inp is not None:
        l, data = inp.read()
        y = np.fromstring(data, dtype='int16')

        if write_to_file:
            wav_file.writeframes(y.tostring())

        spect = compute_spect(
            y.astype('float16'),
            audio_conf,
            normalize=True,
        )

        if debug:
            import pdb
            pdb.set_trace()

        if verbose:
            print("queueing step", step)
        queue.put((step, spect))

        step += 1
        img_i = 0
    print("Finished capture")


# TODO:
# - maybe normalize before detection (after all possible concatenations)

def detect(base_model_path, capture_queue, verbose=False):
    if verbose:
        print("Starting detector, loading models")

    base_model = DeepSpeech.load_model(base_model_path)
    net = load_model(base_model, 'models/binary_clf.pt')

    if verbose:
        print('Model loaded.')

    h0 = None
    c0 = None
    spect_tm1 = None

    while True:
        step, spect = capture_queue.get()

        # prepend the last half of the last seen spectrum
        # to widen the detection horizon a bit.
        if spect_tm1 is not None:
            spect = np.concatenate([spect_tm1, spect], axis=-1)
        spect_tm1 = spect[:, -spect.shape[1]//2:]

        if verbose:
            print(f"Processing {step}...")

        # exptects X.shape = (batch, feature, time)
        # we have spect.shape = [feature, time]
        X = {
            'lens': np.array([spect.shape[1]]),
            'X': np.array(spect)[None],
        }

        if h0 is not None and c0 is not None:
            X['h0'] = h0
            X['c0'] = c0

        #print(X['lens'], X['X'].shape)
        y, indicator, _indicator_seq, h0, c0 = list(net.forward_iter(X))[0]

        print(y, indicator)



if __name__ == "__main__":
    import argparse
    #from multiprocessing import Queue, Process
    import torch
    from torch.multiprocessing import Queue, Process

    from deepspeech.model import DeepSpeech
    from noswear.model import load_model

    parser = argparse.ArgumentParser(description='Audio capture')
    parser.add_argument('--base_model_path',
                        default='./models/librispeech_pretrained.pth',
                        help='Path to pre-trained DS model')
    parser.add_argument('--model_path',
                        default='./models/binary_clf.pt',
                        help='Path to model file created by training')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    # normally we would just load everything here but due to
    # multiprocessing weirdness we cannot. once we load the
    # deepspeech model to the main process, we cannot use it
    # in subprocesses (i don't know why) but simply loading
    # the audio-conf via torch.load does not induce this,
    # probably because the state_dict is not loaded.
    #
    #base_model = DeepSpeech.load_model(args.base_model_path)
    #audio_conf = DeepSpeech.get_audio_conf(base_model)
    #net = load_model(base_model, 'models/binary_clf.pt')
    audio_conf = torch.load(
        args.base_model_path,
        map_location=lambda storage, loc: storage
    )['audio_conf']

    q = Queue()
    p_capture = Process(
            target=capture,
            args=(audio_conf, q, args.verbose, args.debug))

    p_detect = Process(
            target=detect,
            args=(args.base_model_path, q, args.verbose))

    try:
        p_capture.start()
        p_detect.start()

        p_capture.join()
        p_detect.join()

    except KeyboardInterrupt:
        p_capture.terminate()
        p_detect.terminate()
