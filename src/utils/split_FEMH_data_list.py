# -*- coding: utf-8 -*-

import math
import numpy as np
import pdb

np.random.seed(1234)

data_lst = [l.rstrip('\n') for l in open('./FEMH_data.lst')]

num_samples = len(data_lst)
num_train = int(math.floor(num_samples * 0.8))
num_test = num_samples - num_train

inds = np.random.permutation(num_samples)
train_inds = inds[:num_train]
test_inds = inds[num_train:]

train_lst = np.take(data_lst, train_inds)
test_lst = np.take(data_lst, test_inds)

with open('FEMH_data_train.lst', 'w') as f:
    for l in train_lst:
        f.write(l + '\n')

with open('FEMH_data_test.lst', 'w') as f:
    for l in test_lst:
        f.write(l + '\n')
