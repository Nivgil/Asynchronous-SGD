import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision
from progress.bar import IncrementalBar

from densenet import densenet
from wideresnet import WideResNet
from alexnet import alexnet

from data import load_data
from parameter_server import ParameterServer
from statistics import Statistics


def main(args):
    time_stamp = '{:.0f}'.format(time.time() % 100000)
    if torch.cuda.is_available() is True:
        print('Utilizing GPU')
        # torch.cuda.set_device(args.gpu_num)
    train_loader, val_loader = load_data(args)
    # create model
    if args.dataset == 'imagenet':
        model = alexnet()
        # model = torchvision.models.resnet50()
        args.iterations_per_epoch = len(train_loader.dataset.imgs) // args.batch_size
        val_len = len(val_loader.dataset.imgs) // 512
    else:
        model = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                           args.widen_factor, dropRate=args.droprate)
        # model = densenet(depth=100)
        # args.wd = 1e-4
        args.iterations_per_epoch = len(train_loader.dataset.train_labels) // args.batch_size
        val_len = len(val_loader.dataset.test_labels) // 1280

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    # for training on multiple GPUs.
    model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    server = ParameterServer.get_server(args.optimizer, model, args)
    val_statistics = Statistics.get_statistics('image_classification', args)
    train_statistics = Statistics.get_statistics('image_classification', args)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume + '/checkpoint.pth.tar'):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume + '/checkpoint.pth.tar')
            args.start_epoch = checkpoint['epoch']
            server = checkpoint['server']
            val_statistics = checkpoint['val_stats']
            train_statistics = checkpoint['train_stats']
            model.load_state_dict(checkpoint['state_dict'])
            print('=> loaded checkpoint {} (epoch {})'.format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # ghost batch normalization (128 as baseline)
    repeat = args.batch_size // 128 if args.gbn == 1 else 1
    if args.bar is True:
        train_bar = IncrementalBar('Training  ', max=args.iterations_per_epoch, suffix='%(percent)d%%')
        val_bar = IncrementalBar('Evaluating', max=val_len, suffix='%(percent)d%%')
    else:
        train_bar = None
        val_bar = None

    print(
        '{}: Training neural network for {} epochs with {} workers'.format(args.sim_num, args.epochs, args.workers_num))
    train_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_loss, train_error = train(train_loader, model, criterion, server, epoch, args.workers_num, args.grad_clip,
                                        repeat, train_bar)

        train_time = time.time() - train_time
        if args.bar is True:
            train_bar.finish()
            train_bar.index = 0

        # evaluate on validation set
        val_time = time.time()
        val_loss, val_error = validate(val_loader, model, criterion, server, val_statistics, val_bar)
        train_statistics.save_loss(train_loss)
        train_statistics.save_error(train_error)
        train_statistics.save_weight_mean_dist(server.get_workers_mean_statistics())
        train_statistics.save_weight_master_dist(server.get_workers_master_statistics())
        train_statistics.save_mean_master_dist(server.get_mean_master_dist())
        train_statistics.save_weight_norm(server.get_server_weights())
        train_statistics.save_gradient_norm(server.get_server_gradients())
        val_time = time.time() - val_time
        if args.bar is True:
            val_bar.finish()
            val_bar.index = 0
        print('Epoch [{0:1d}]: Train: Time [{1:.2f}], Loss [{2:.3f}], Error[{3:.3f}] |'
              ' Test: Time [{4:.2f}], Loss [{5:.3f}], Error[{6:.3f}]'
              .format(epoch, train_time, train_loss, train_error, val_time, val_loss, val_error))
        if epoch % 1 == 0:
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'val_stats': val_statistics,
                             'train_stats': train_statistics,
                             'server': server}, sim_name=(args.name + time_stamp + '_' + str(epoch)))
        train_time = time.time()

    return train_statistics, val_statistics


def train(train_loader, model, criterion, server, epoch, workers_number, grad_clip, repeat, bar):
    """Train for one epoch on the training set"""
    train_error = AverageMeter()
    train_loss = AverageMeter()
    backward_pass, forward_pass, param_push, grad_pull, full_pass, param_pull, batch_loading, variable_tran = 0, 0, 0, 0, 0, 0, 0, 0
    # switch to train mode
    model.train()
    time_1 = time.time()
    for i, (input, target) in enumerate(train_loader):
        batch_loading += time.time() - time_1
        current_worker = i % workers_number
        time_3 = time.time()
        set_model_weights(server.pull(current_worker), model)
        param_pull += time.time() - time_3
        time_2 = time.time()
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        variable_tran += time.time() - time_2
        # compute output
        model.zero_grad()
        time_4 = time.time()
        for t in range(0, repeat):
            time_7 = time.time()
            output = model(input_var)
            forward_pass += time.time() - time_7
            loss = criterion(output, target_var)
            time_8 = time.time()
            loss.backward()
            backward_pass += time.time() - time_8
        if repeat > 1:
            normalize_gradients(model, repeat)
        if grad_clip < 1000:
            torch.nn.utils.clip_grad_norm(model.parameters(), grad_clip)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        train_loss.update(loss.data[0], input.size(0))
        train_error.update(100 - prec1[0], input.size(0))
        full_pass += time.time() - time_4
        time_5 = time.time()
        gradients = get_model_gradients(model)
        grad_pull += time.time() - time_5
        tau = (i - current_worker) / workers_number + 1
        time_6 = time.time()
        server.push(current_worker, gradients, epoch, tau=tau, iteration=i)
        param_push += time.time() - time_6
        if bar is not None:
            bar.next()
        time_1 = time.time()
    # print('\nCompleted {} Iterations'.format(i))
    # print('Batch Loading - [{:.3f}]sec'.format(batch_loading))
    # print('param pulling - [{:.3f}]sec'.format(param_pull))
    # print('variable tran - [{:.3f}]sec'.format(variable_tran))
    # print('full pass     - [{:.3f}]sec'.format(full_pass))
    # print('grad pulling  - [{:.3f}]sec'.format(grad_pull))
    # print('param pushing - [{:.3f}]sec'.format(param_push))
    # print('forward pass  - [{:.3f}]sec'.format(forward_pass))
    # print('backward pass - [{:.3f}]sec'.format(backward_pass))
    return train_loss.avg, train_error.avg


def validate(data_loader, model, criterion, server, statistics, bar):
    """Perform validation on the validation set"""

    server_weights = server.get_server_weights()
    set_model_weights(server_weights, model)
    # switch to evaluate mode
    model.eval()

    error = AverageMeter()
    error_5 = AverageMeter()
    total_loss = AverageMeter()
    for i, (input, target) in enumerate(data_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        total_loss.update(loss.data[0], input.size(0))
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        error.update(100 - prec1[0], input.size(0))
        error_5.update(100 - prec5[0], input.size(0))
        if bar is not None:
            bar.next()

    statistics.save_loss(total_loss.avg)
    statistics.save_error(error.avg)
    statistics.save_error_top5(error_5.avg)
    return total_loss.avg, error.avg


def set_model_weights(weights, model):
    for name, weight in model.named_parameters():
        if torch.cuda.is_available() is True:
            weight.data = weights[name].cuda()
        else:
            weight.data = weights[name]


def get_model_gradients(model):
    gradients = {}
    for name, weight in model.named_parameters():
        gradients[name] = weight.grad.clone()
    return gradients


def get_model_weights(model):
    weights = {}
    for name, weight in model.named_parameters():
        weights[name] = weight.clone()
    return weights


def normalize_gradients(model, norm_factor):
    for weight in model.parameters():
        weight.grad.div_(norm_factor)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, filename='checkpoint.pth.tar', sim_name=''):
    """Saves checkpoint to disk"""
    name = sim_name
    directory = '/media/niv/backups/%s/' % name
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    # def get_average(self):
    #     return self.avg.cpu().numpy()[0]
# if __name__ == '__main__':
#     main(args)
