# -*- coding: utf-8 -*-

import torch
from torch.nn import functional as F
import torch.optim as optim
import os
import sys
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.realpath(__file__))),
        'utils'))
from utils.utils import save_checkpoint
import pdb


def _convert2gray(x, scale=256.):
    '''
    Convert data to gray-scale [0, scale].

    Parameters
    ----------
    x: torch.FloatTensor, shape (B, C, M, M)
        Input tensor of batch B, channel C, size N-by-N.
    scale: Float

    Returns
    -------
    x_gray: torch.FloatTensor, shape (B, C, M, M)
        Tensor in gray-scale [0., scale].
    '''
    B, C, _, _ = x.size()

    x_flat = x.view(B, -1)

    xmin, _ = x_flat.min(dim=1)
    xmax, _ = x_flat.max(dim=1)
    xmin = xmin.reshape((B, 1, 1, 1))
    xmax = xmax.reshape((B, 1, 1, 1))

    x = scale * (x - xmin) / (xmax - xmin)
    x = x.type(dtype=torch.uint8).type(dtype=torch.float32)
    return x


def _resize(x, size=16):
    '''
    Resize original input for GLCM.

    Parameters
    ----------
    x: torch.FloatTensor, shape (B, C, M, M)

    Returns
    -------
    x_resize: torch.FloatTensor, shape (B, C, R, R)
        R is new size.
    '''
    return F.interpolate(x, (size, size), mode='bilinear', align_corners=False)


def train_epoch(dataloader, model, glcm, optimizer, epoch, args, logger):
    model.train()

    loss_avg = 0.
    err_avg = 0.
    tot_samples = 0

    for i, (X, Y) in enumerate(dataloader):
        X = X.to(args['device'])
        Y = Y.to(args['device'])

        # Preprocess data
        X = _convert2gray(X, scale=args['GLCM']['num_pixel_vals'])
        X = _resize(X, size=args['GLCM']['input_dim'])

        optimizer.zero_grad()

        G = glcm(X)
        Yh = model(X, G)

        loss = torch.nn.functional.cross_entropy(
            Yh, Y, reduction='elementwise_mean')

        loss.backward()

        # Clip gradient
        torch.nn.utils.clip_grad_value_(
            model.parameters(), args['optimizer']['grad_clip'])

        optimizer.step()

        loss_avg += loss.item() * Y.size(0)
        tot_samples += Y.size(0)

        if args['verbose']:
            logger.debug('[{}/{}][{}/{}] Train: loss = {:.4f}'.
                         format(epoch, args['num_epochs'],
                                i, len(dataloader),
                                loss.item()))

        Y_pred = model.predict(X)
        mis_classified = Y_pred.ne(Y).sum().item()
        err_avg += mis_classified

    loss_avg /= float(tot_samples)
    err_avg /= float(tot_samples)

    return loss_avg, err_avg


def test_epoch(dataloader, model, glcm, args, logger):
    model.eval()

    loss_avg = 0.
    err_avg = 0.
    tot_samples = 0

    for i, (X, Y) in enumerate(dataloader):
        X = X.to(args['device'])
        Y = Y.to(args['device'])

        X = _convert2gray(X, scale=args['GLCM']['num_pixel_vals'])
        X = _resize(X, size=args['GLCM']['input_dim'])

        G = glcm(X)
        Yh = model(X, G)

        loss = torch.nn.functional.cross_entropy(
            Yh, Y, reduction='elementwise_mean')

        loss_avg += loss.item() * Y.size(0)
        tot_samples += Y.size(0)

        Y_pred = model.predict(X)
        mis_classified = Y_pred.ne(Y).sum().item()
        err_avg += mis_classified

    loss_avg /= float(tot_samples)
    err_avg /= float(tot_samples)

    return loss_avg, err_avg


def trainer(train_loader, test_loader, model, glcm, args, logger, run_id=0):

    # Optimizer
    optimizer = optim.SGD(list(model.parameters()) + list(glcm.parameters()),
                          lr=args['optimizer']['lr'],
                          momentum=args['optimizer']['momentum']
                          )

    # optimizer = optim.SGD(model.parameters(),
                          # lr=args['optimizer']['lr'],
                          # momentum=args['optimizer']['momentum']
                          # )

    # optimizer = optim.SGD(glcm.parameters(),
                          # lr=args['optimizer']['lr'],
                          # momentum=args['optimizer']['momentum']
                          # )

    # Lr scheduler
    lambda_func = lambda epoch: args['optimizer']['lr_decay_mul'] **\
        (epoch // args['optimizer']['lr_decay_epochs'])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)

    # Train
    num_epochs = args['num_epochs']

    train_losses = []  # losses over epochs
    test_losses = []
    train_errors = []
    test_errors = []

    for epoch in range(num_epochs):

        scheduler.step()

        # Train an epoch
        _, _ = train_epoch(train_loader, model, glcm, optimizer, epoch,
                           args, logger)

        # Test
        loss, err = test_epoch(train_loader, model, glcm, args, logger)
        train_losses.append(loss)
        train_errors.append(err)
        logger.info('[{}/{}] Train loss = {:.4f} err = {:.4f} lr={}'.
                    format(epoch, num_epochs, loss, err,
                           scheduler.get_lr()[0]))

        loss, err = test_epoch(test_loader, model, glcm, args, logger)
        test_losses.append(loss)
        test_errors.append(err)
        logger.info('[{}/{}] Test loss = {:.4f} err = {:.4f} lr={}'.
                    format(epoch, num_epochs, loss, err,
                           scheduler.get_lr()[0]))

        # Save model state
        state = {
            'args': args,
            'epoch': epoch,
            'model': model.state_dict(),
            'glcm': glcm.state_dict()
        }
        save_checkpoint(state, args['checkpoint_dir'],
                        'checkpoint_fold_{:d}_epoch_{:d}.pth.tar'.
                        format(run_id, epoch))
        logger.info('Saved checkpoint_fold_{:d}_epoch_{}.pth.tar'
                    .format(run_id, epoch))

    return train_losses, test_losses, train_errors, test_errors
