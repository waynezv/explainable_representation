# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.io import wavfile as siowav
import pdb


data_root = "/media/sdd/wzhao/ProJEX/explainable_representation/model_outputs/lrp/outputs/test"

img_files = [l.rstrip('\n') for l in open('./lrp_img_test.lst')]

cmap = 'jet'
for i in range(len(img_files) // 2):
    fig = plt.figure(figsize=(8, 6))

    f = img_files[i * 2]
    img = plt.imread(os.path.join(data_root, f))

    ax1 = fig.add_subplot(121)
    plt.imshow(img, cmap=cmap)
    ax1.axis('off')
    ax1.set_title(f)

    f = img_files[i * 2 + 1]
    img = plt.imread(os.path.join(data_root, f))

    ax2 = fig.add_subplot(122)
    plt.imshow(img, cmap=cmap)
    ax2.axis('off')
    ax2.set_title(f)

    plt.show()
