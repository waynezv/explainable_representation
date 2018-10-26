# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import logging
import json
import torch
import pdb

from models.AlexNet import AlexNet
from models.trainer import trainer
from utils.datasets import prepare_data, SpokenDigits

if len(sys.argv) < 2:
    print('python {} configure.json'.format(sys.argv[0]))
    sys.exit(-1)

args = json.load(sys.argv[1])

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter(
    fmt='%(asctime)s - [%(name)s - %(funcName)s:%(lineno)d]'
    ' - [%(levelname)-5.5s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M'
)

if os.path.exists(args['log']):
    # TODO: remove
    pass
file_handler = logging.FileHandler(args['log'])
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

std_handler = logging.StreamHandler(sys.stdout)
std_handler.setFormatter(log_formatter)
std_handler.setLevel(logging.DEBUG)
logger.addHandler(std_handler)

yield_train_test = prepare_data(data_list, num_folds=5)

for i in range(args['num_folds']):
    logger.info('Loading & preparing data')

    train_list, test_list = yield_train_test(i)
    train_data = SpokenDigits(train_list)
    test_data = SpokenDigits(test_list)

    trainer(train_data, model, criterion, args, logger)



logger.info('Parameters & settings')

logger.info('Parameters & settings')

if args['verbose']:
    logger.debug('[{}/{}][{}/{}] Train: loss = {:.4f}'.format())

logger.info('[{}/{}] Train: loss = {:.4f} err = {:.4f}'.format())
logger.info('[{}/{}] Test: loss = {:.4f} err = {:.4f}'.format())
