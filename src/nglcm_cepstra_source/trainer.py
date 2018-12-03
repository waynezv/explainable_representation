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


def train_epoch(dataloader, model, filnet, optimizer, epoch, args, logger):
    model.train()

    loss_avg = 0.
    err_avg = 0.
    tot_samples = 0

    for i, (X1, X2, Y) in enumerate(dataloader):
        X1 = X1.to(args['device'])  # spec
        X2 = X2.to(args['device'])  # ceps
        Y = Y.to(args['device'])

        optimizer.zero_grad()

        G = filnet(X2)
        Yh = model(X1, G)

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

        Y_pred = model.predict(X1)
        mis_classified = Y_pred.ne(Y).sum().item()
        err_avg += mis_classified

    loss_avg /= float(tot_samples)
    err_avg /= float(tot_samples)

    return loss_avg, err_avg


def test_epoch(dataloader, model, filnet, args, logger):
    model.eval()

    loss_avg = 0.
    err_avg = 0.
    tot_samples = 0

    for i, (X1, X2, Y) in enumerate(dataloader):
        X1 = X1.to(args['device'])  # spec
        X2 = X2.to(args['device'])  # ceps
        Y = Y.to(args['device'])

        G = filnet(X2)
        Yh = model(X1, G)

        loss = torch.nn.functional.cross_entropy(
            Yh, Y, reduction='elementwise_mean')

        loss_avg += loss.item() * Y.size(0)
        tot_samples += Y.size(0)

        Y_pred = model.predict(X1)
        mis_classified = Y_pred.ne(Y).sum().item()
        err_avg += mis_classified

    loss_avg /= float(tot_samples)
    err_avg /= float(tot_samples)

    return loss_avg, err_avg


def trainer(train_loader, test_loader, model, filnet,
            args, logger, run_id=0):

    # Optimizer
    optimizer = optim.SGD(list(model.parameters()) +
                          list(filnet.parameters()),
                          lr=args['optimizer']['lr'],
                          momentum=args['optimizer']['momentum'],
                          nesterov=args['optimizer']['nesterov'],
                          dampening=args['optimizer']['dampening'],
                          weight_decay=args['optimizer']['weight_decay']
                          )
    # optimizer = optim.Adam(list(model.parameters()) +
                           # list(filnet.parameters()),
                           # lr=args['optimizer']['lr']
                           # )

    # optimizer = optim.SGD(model.parameters(),
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
        _, _ = train_epoch(train_loader, model, filnet, optimizer,
                           epoch, args, logger)

        # Test
        loss, err = test_epoch(train_loader, model, filnet, args, logger)
        train_losses.append(loss)
        train_errors.append(err)
        logger.info('[{}/{}] Train loss = {:.4f} err = {:.4f} lr={}'.
                    format(epoch, num_epochs, loss, err,
                           scheduler.get_lr()[0]))

        loss, err = test_epoch(test_loader, model, filnet, args, logger)
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
            'filter_net': filnet.state_dict()
        }
        save_checkpoint(state, args['checkpoint_dir'],
                        'checkpoint_fold_{:d}_epoch_{:d}.pth.tar'.
                        format(run_id, epoch))
        logger.info('Saved checkpoint_fold_{:d}_epoch_{}.pth.tar'
                    .format(run_id, epoch))

    return train_losses, test_losses, train_errors, test_errors
