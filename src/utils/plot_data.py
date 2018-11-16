# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.io import wavfile as siowav
import pdb


data_root = "../../data/free-spoken-digit-dataset/recordings/"

test_file = "0_jackson_5.wav"

sample_rate, samples = siowav.read(os.path.join(data_root, test_file))
print('sample_rate: ', sample_rate)
print('signal length: ', len(samples), ' ', len(samples) / sample_rate, 's')

fig = plt.figure(figsize=(8, 8))

ax1 = fig.add_subplot(211)
ax1.plot(range(len(samples)), samples, 'r', alpha=0.5)
ax1.axis('tight')
ax1.grid(True)
ax1.set_xlabel('t')
ax1.set_ylabel('amp')
ax1.set_title('jackson 5, Fs={}'.format(sample_rate))

# cmap = 'viridis'
cmap = 'jet'

ax2 = fig.add_subplot(223)
spec, freqs, ts, im = ax2.specgram(
    samples,
    Fs=sample_rate,
    NFFT=256,
    noverlap=128,
    mode="magnitude",
    scale="dB",
    cmap=cmap
    )
ax2.axis('off')
ax2.set_title('NFFT={} noverlap={}'.format(256, 128))
print('spectrogram shape: ', spec.shape)

ax3 = fig.add_subplot(224)
spec, freqs, ts, im = ax3.specgram(
    samples,
    Fs=sample_rate,
    NFFT=454,
    noverlap=421,
    mode="magnitude",
    scale="dB",
    cmap=cmap
    )
ax3.axis('off')
ax3.set_title('NFFT={} noverlap={}'.format(454, 421))
print('spectrogram shape: ', spec.shape)

fig.tight_layout()
# fig.show()
# fig.savefig('sample_plot_jackson_5.jpg')

# Test im.get_array
fig = plt.figure(figsize=(8, 8))
plt.imshow(im.get_array(), cmap=cmap)
# plt.show()
# plt.imsave('sample_spec_jackson_5.jpg', im.get_array(), cmap=cmap)

# Test gray scale
fig = plt.figure(figsize=(8, 8))
img = spec
img = 255 * (img - img.min()) / (img.max() - img.min())
img = img.astype('uint8')
plt.imshow(np.flipud(img), cmap=cmap)
#  plt.show()
#  fig.savefig('gray_scale.jpg')
