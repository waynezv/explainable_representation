# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from scipy.io import wavfile as siowav
from scipy.signal import get_window, spectrogram
from scipy.fftpack import dct
from base import fbank, mfcc, delta
import pdb


def pad_signal(sig, pad_length=8000, seed=1234):
    '''
    Pad signal to fixed length in random position.

    Parameters
    ----------
    sig: numpy.array[float]
    pad_length: int

    Returns
    -------
    sig_pad: numpy.array[float], shape (pad_length,)
    '''
    np.random.seed(seed)

    sig_len = len(sig)

    if sig_len == pad_length:
        return sig

    if sig_len > pad_length:  # extract middle
        mid = sig_len // 2
        sp = int(mid - pad_length / 2)
        ep = int(mid + pad_length / 2)
        return sig[sp:ep]

    if sig_len < pad_length:  # random pad
        len_diff = pad_length - sig_len
        split = np.random.randint(0, len_diff + 1)  # random split
        pad_s = int(split)
        pad_e = int(len_diff - split)
        sig_pad = np.pad(sig, (pad_s, pad_e), 'constant', constant_values=0)
        return sig_pad


# Data dirs
data_root = "../../data/FEMH_Data"
data_list = './FEMH_data.lst'
save_dir = "../../data/FEMH_Data/processed/mfcc26"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
files = [(l.rstrip('\n').split()) for l in open(data_list)]

lens = []
for f, label in files:
    # Read signal
    sample_rate, samples = siowav.read(os.path.join(data_root, f))
    lens.append(len(samples))
    assert sample_rate == 44100, "{}: incompatible sample rate"\
        " need 8000 but got {}".format(f, sample_rate)

    # Pad
    samples = samples.astype('float32')
    samples = pad_signal(samples, 44100 * 10)
    assert len(samples) == 44100 * 10, "{}: incorrect signal length"\
        " need 44100 * 10 but got {}".format(f, len(samples))

    # Compute cepstra
    feat, energy = fbank(
        samples,
        sample_rate,
        0.016,  # winlen
        0.008,  # winstep
        40,  # nfilt
        1024,  # nfft
        0,  # low freq
        sample_rate / 2.,  # high freq
        0.97,  # preemph
        np.hamming  # winfunc
    )

    # Log Mel 40
    feat = np.log(feat)

    # DCT Cepstra 40
    feat = dct(feat, type=2, axis=1, norm='ortho')
    # Replace first cepstral coefficient with log of frame energy
    feat[:, 0] = np.log(energy)

    # MFCC 13
    feat = feat[:, :13]

    # MFCC 26
    delta_feat = delta(feat, 1)
    feat = np.concatenate((feat, delta_feat), axis=1)

    # from matplotlib import pyplot as plt
    # plt.plot(feat[600, :])
    # plt.show()

    # Save
    save_f = os.path.join(save_dir, os.path.splitext(f)[0])

    basedir = os.path.dirname(save_f)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    np.save(save_f, feat)
    print("saved {}.npy".format(save_f))
