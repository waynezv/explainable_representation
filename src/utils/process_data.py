# -*- coding: utf-8 -*-
'''
Data processing utilities for spkoken digit dataset.

https://github.com/Jakobovski/free-spoken-digit-dataset
'''
import os
import sys
import numpy as np
from scipy.io import wavfile as siowav
from scipy.signal import get_window, spectrogram
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


def normalize_signal(sig, percentile=95.):
    '''
    Normalize signal against the pth percentile.
    '''
    return sig / np.percentile(sig, percentile)


data_root = "../../data/free-spoken-digit-dataset/recordings/"
data_list = './recording.list'
save_dir = "../../data/free-spoken-digit-dataset/processed/spectrograms"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
files = [l.rstrip('\n') for l in open(data_list)]

win = get_window('hann', 454)

for f in files:
    # Read signal
    sample_rate, samples = siowav.read(os.path.join(data_root, f))
    assert sample_rate == 8000, "{}: incompatible sample rate"\
        " need 8000 but got {}".format(f, sample_rate)

    # Pad
    samples.astype('float32')
    samples = pad_signal(samples)
    assert len(samples) == 8000, "{}: incorrect signal length"\
        " need 8000 but got {}".format(f, len(samples))

    # Normalize
    # samples = normalize_signal(samples)

    # Compute spectrogram
    _, _, spec = spectrogram(
        samples,
        fs=sample_rate,
        window=win,
        nperseg=454,
        noverlap=421,
        nfft=454,
        detrend='constant',
        return_onesided=True,
        scaling='density',
        mode='magnitude')

    # Flip and drop
    spec = np.flipud(spec)
    spec = spec[1:, :-2]
    assert spec.shape == (227, 227), "{}: incorrect spectrogram shape"\
        " need (227, 227) but got {}".format(f, spec.shape)

    # Save
    save_f = os.path.join(save_dir, os.path.splitext(f)[0])
    np.save(save_f, spec)
    print("saved {}.npy".format(save_f))
