# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import pdb

file = './neoplasm_001.8k.pitch'
pitch = [v.rstrip('\n') for v in open(file)]
pitch = np.asarray(pitch, dtype=float)

plt.figure()
plt.plot(pitch)
# plt.show()
plt.savefig(file + '.png')
