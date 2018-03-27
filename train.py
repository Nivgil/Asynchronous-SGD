import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from copy import deepcopy

from wideresnet import WideResNet
from alexnet import alexnet
from simple_model import SimpleModel

from data import load_data
from parameter_server import ParameterServer
from statistics import Statistics


def main(args):
    if torch.cuda.is_available() is True:
        print('Utilizing GPU')
        # torch.cuda.set_device(args.gpu_num)
    train_loader, val_loader = load_data(args)

    # create model
    if args.dataset == 'image_net':
        model = alexnet()
    else:
        model = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                           args.widen_factor, dropRate=args.droprate)
        # model = SimpleModel()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    server = ParameterServer.get_server('asynchronous', model, args)
    val_statistics = Statistics.get_statistics('image_classification', args)
    train_statistics = Statistics.get_statistics('image_classification', args)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    print('Training neural network for {} epochs with {} workers'.format(args.epochs, args.workers_num))
    train_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, server, epoch, args.workers_num, args.grad_clip)
        train_time = time.time() - train_time

        # evaluate on validation set
        val_time = time.time()
        val_loss, val_error = validate(val_loader, model, criterion, server, val_statistics)
        train_loss, train_error = validate(train_loader, model, criterion, server, train_statistics)
        val_time = time.time() - val_time
        print('Epoch [{0:1d}]: Train: Time [{1:.2f}], Loss [{2:.3f}], Error[{3:.3f}] |'
              ' Test: Time [{4:.2f}], Loss [{5:.3f}], Error[{6:.3f}]'
              .format(epoch, train_time, train_loss, train_error, val_time, val_loss, val_error))
        train_time = time.time()

    return train_statistics, val_statistics


def train(train_loader, model, criterion, server, epoch, workers_number, grad_clip):
    """Train for one epoch on the training set"""

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        current_worker = i % workers_number
        set_model_weights(server.pull(current_worker), model)
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), grad_clip)
        gradients = get_model_gradients(model)
        server.push(current_worker, gradients, epoch)


def validate(val_loader, model, criterion, server, statistics):
    """Perform validation on the validation set"""

    server_weights = server.get_server_weights()
    set_model_weights(server_weights, model)
    # switch to evaluate mode
    model.eval()

    total_loss, error = 0, 0
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        total_loss = total_loss + loss.data[0]
        _, class_pred = torch.max(output, 1)
        error = error + 1 - torch.sum(class_pred.data == target_var.data) / len(target_var)

    statistics.save_weight_norm(server_weights)
    statistics.save_gradient_norm(server.get_server_gradients())
    statistics.save_loss(total_loss / i)
    statistics.save_error(error / i)
    loss = total_loss / i
    error = 100 * error / i
    return loss, error


def set_model_weights(weights, model):
    for name, weight in model.named_parameters():
        if torch.cuda.is_available() is True:
            weight.data = weights[name].cuda()
        else:
            weight.data = weights[name]


def get_model_gradients(model):
    gradients = {}
    for name, weight in model.named_parameters():
        gradients[name] = deepcopy(weight.grad)
    return gradients

# if __name__ == '__main__':
#     main(args)
