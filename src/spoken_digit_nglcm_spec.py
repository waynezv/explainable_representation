# -*- coding: utf-8 -*-
'''
Main program for spoken digit classification with
Neural Gray Level Co-occurrence Matrix projection.
'''
import os
import sys
import numpy as np
import logging
import logging.config
import json
import torch
import pdb

from nglcm.nglcm import AlexNet, GLCM
from nglcm.trainer import trainer
from utils.datasets import prepare_data, SpokenDigits

# Parse arguments
if len(sys.argv) < 2:
    print('python {} configure.json'.format(sys.argv[0]))
    sys.exit(-1)

with open(sys.argv[1], 'r') as f:
    args = json.load(f)

# CUDA, randomness
if torch.cuda.is_available():
    print('CUDA available: True')

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.fastest = True
else:
    print('CUDA not available!')
    sys.exit(-1)

torch.manual_seed(args['random_seed'])
torch.cuda.manual_seed_all(args['random_seed'])

# Log
if not os.path.exists(args['checkpoint_dir']):
    os.makedirs(args['checkpoint_dir'])
if not os.path.exists(args['log_dir']):
    os.makedirs(args['log_dir'])

log_file = os.path.join(args['log_dir'], args['log_file'])
if os.path.isfile(log_file):
    os.remove(log_file)

logging.config.dictConfig(args['log'])
logger = logging.getLogger('main')

# Data func
yield_train_test = prepare_data(args['data']['list'],
                                num_folds=args['num_folds'],
                                seed=args['random_seed'])

# Main
loss_folds_train = []  # collect loss per validation fold
loss_folds_test = []
err_folds_train = []
err_folds_test = []

for i in range(args['num_runs']):  # num_runs = C(num_folds, 3)
    # Data
    logger.info('Loading & preparing data')

    train_list, test_list = yield_train_test(i)
    train_data = SpokenDigits(train_list, root=args['data']['path'])
    test_data = SpokenDigits(test_list, root=args['data']['path'])

    train_loader = torch.utils.data.DataLoader(
        train_data,
        args['batch_size'],
        shuffle=True,
        num_workers=args['num_workers']
        )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        args['batch_size'],
        shuffle=True,
        num_workers=args['num_workers']
        )

    # Model
    logger.info('Building model')
    model = AlexNet(num_classes=args['AlexNet']['num_classes']) \
        .to(args['device'])

    # load pretrained model state
    checkpoint = torch.load(os.path.join(args['saved_model_dir'],
                                         args['model_to_eval']))
    pretrained_model_state = checkpoint['model']
    pretrained_model_state['classifier.0.weight'] = torch.cat(
        (pretrained_model_state['classifier.0.weight'],
         torch.zeros_like(
             pretrained_model_state['classifier.0.weight']).normal_()
         ), dim=1)
    model.load_state_dict(pretrained_model_state)

    # freeze model parameters except last layer
    num_parameters = len(list(model.parameters()))
    for i, param in enumerate(model.parameters()):
        if i >= num_parameters - 2:
            break
        param.requires_grad = False

    glcm = GLCM(in_dim=args['GLCM']['input_dim'] * args['GLCM']['input_dim'],
                num_pixel_vals=args['GLCM']['num_pixel_vals']) \
        .to(args['device'])

    # Train
    logger.info('Training')
    train_losses, test_losses, train_errors, test_errors = \
        trainer(train_loader, test_loader, model, glcm, args, logger, run_id=i)

    # Log
    logger.info('*' * 50)
    logger.info('Fold = {}'.format(i))

    loss_best = np.min(train_losses)
    err_best = np.min(train_errors)
    loss_folds_train.append(loss_best)
    err_folds_train.append(err_best)
    logger.info('Train loss = {:.4f}  error = {:.4f}'.
                format(loss_best, err_best))

    loss_best = np.min(test_losses)
    err_best = np.min(test_errors)
    loss_folds_test.append(loss_best)
    err_folds_test.append(err_best)
    logger.info('Test loss = {:.4f}  error = {:.4f}'.
                format(loss_best, err_best))
    logger.info('*' * 50)

logger.info('Average over {} runs'.format(args['num_runs']))

loss_avg = np.mean(loss_folds_train)
err_avg = np.mean(err_folds_train)
logger.info('Train loss = {:.4f}  error = {:.4f}'.
            format(loss_avg, err_avg))

loss_avg = np.mean(loss_folds_test)
err_avg = np.mean(err_folds_test)
logger.info('Test loss = {:.4f}  error = {:.4f}'.
            format(loss_avg, err_avg))

logger.info('=' * 50)
logger.info('Parameters & settings')
logger.info(args)
