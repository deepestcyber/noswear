import argparse
import time

import numpy as np
import torch


def transcribe(net, q, verbose):
    hidden = None
    print("start transcribing...")

    while True:
        step, spect = q.get()

        tick = time.time()

        spect = torch.tensor(spect)
        spect_in = spect.contiguous().view(1, 1, spect.size(0), spect.size(1))
        out, hidden = net.module_.forward_step(spect_in, hidden)

        if verbose:
            print('od',out.data.shape)
            print('out', out)

        tock = time.time()
        print("modeling step", step, "model time:", tock - tick)


if __name__ == '__main__':
    import argparse
    from multiprocessing import Queue, Process

    from deepspeech.model import DeepSpeech
    from capture import capture
    from model import load_model

    parser = argparse.ArgumentParser(description='DeepSpeech transcription')
    parser.add_argument('--base_model_path',
                        default='./models/librispeech_pretrained.pth',
                        help='Path to pre-trained DS model')
    parser.add_argument('--model_path',
                        default='./models/binary_clf.pt',
                        help='Path to model file created by training')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use_file', action='store_true')

    args = parser.parse_args()

    base_model = DeepSpeech.load_model(args.base_model_path)
    audio_conf = DeepSpeech.get_audio_conf(base_model)
    net = load_model(base_model, args.model_path)
    net.module_.eval()

    q = Queue()

    p_capture = Process(
            target=capture,
            args=(audio_conf, args.use_file, q, args.verbose))

    p_transcribe = Process(
            target=transcribe,
            args=(net, q, args.verbose))

    try:
        p_capture.start()
        p_transcribe.start()

        p_capture.join()
        p_transcribe.join()
    except KeyboardInterrupt:
        p_capture.terminate()
        p_transcribe.terminate()
