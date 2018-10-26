# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Dataloader


def prepare_data(data_list, num_folds=5, seed=1234):
    '''
    Prepare train and test data with N folds.

    Parameters
    ----------
    data_list: List[string]
        List of files.
    num_folds: Int
    seed: Int

    Returns
    -------
    yield_train_test: Function
        A closure function to yield train and test lists.
    '''
    np.random.seed(seed)

    def split_folds(N):
        '''
        Split data_list to N folds.
        '''
        files = [l.rstrip('\n') for l in data_list]
        num_files = len(files)
        num_per_fold = num_files // N

        file_idx = np.random.permutation(range(num_files))
        folds = [file_idx[i:i + num_per_fold]
                 for i in range(0, num_files, num_per_fold)]

        return folds

    folds = split_folds(num_folds)
    tot_idx = (0, 1, 2, 3, 4)
    train_idx = [(0, 1, 2),
                 (0, 1, 3),
                 (0, 1, 4),
                 (0, 2, 3),
                 (0, 2, 4),
                 (0, 3, 4),
                 (1, 2, 3),
                 (1, 2, 4),
                 (1, 3, 4),
                 (2, 3, 4)]

    def yield_train_test(i):
        '''
        Yield train and test lists.

        Parameters
        ----------
        i: Int
            The i-th run in total C(num_folds, 3) runs.

        Returns
        -------
        train_set, test_set: List[string]
        '''
        train_i = train_idx[i]
        test_i = list(set(tot_idx) - set(train_i))

        train_set = folds[train_i[0]] + folds[train_i[1]] + folds[train_i[2]]
        test_set = folds[test_i[0]] + folds[test_i[1]] + folds[test_i[2]]

        return train_set, test_set

    return yield_train_test


class SpokenDigits(Dataset):
    '''
    Wrapper for SpokenDigits dataset.
    '''
    def __init__(self, file_list, root='.'):
        self.root = root
        self.files = [l.rstrip('\n') for l in open(file_list)]
        self.num_files = len(self.files)

    def __len__(self):
        return self.num_files

    def __getitem__(self, i):
        X = np.load(os.path.join(self.root, self.files[i]))
        attr = self.files[i].split('_')  # e.g. "0_theo_7.npy"
        Y = float(attr[0])
        return (X, Y)
