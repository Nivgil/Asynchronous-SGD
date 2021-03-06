import torch
import numpy as np
from copy import deepcopy
from math import sqrt
import logging


class ParameterServer(object):

    @staticmethod
    def get_server(mode, *args, **kwargs):
        return {'synchronous': ASGD, 'asynchronous': ASGD}[mode](*args, **kwargs)

    @staticmethod
    def get_lr_reduce_epochs(model):
        return \
            {'resnet': [30, 60, 80], 'alexnet': [10, 15, 20, 25], 'wideresnet': [60, 120, 160], 'densenet': [150, 225]}[
                model]

    def __init__(self, model, args, **kwargs):
        self._model = deepcopy(model)
        self._workers_num = args.workers_num

        # learning rate initialization
        batch_baseline = args.baseline
        self._lr = args.lr * np.sqrt((args.workers_num * args.batch_size) // batch_baseline) / (args.workers_num)
        self._fast_regime = args.fast_regime
        self._current_lr = args.lr
        self._m_off = args.m_off
        self._current_momentum = args.momentum
        self._lr_points = self.get_lr_reduce_epochs(args.model)
        self._lr_factor = 0.2 if args.model == 'wideresnet' else 0.1
        self._iterations_per_epoch = args.iterations_per_epoch
        self._momentum = args.momentum
        self._client = args.client

        if args.full_regime is True:
            alpha = args.workers_num * args.batch_size // batch_baseline
            self._lr_points = [x * alpha for x in self._lr_points]
            print('Regime Adaptation - LR Reduce at {}'.format(self._lr_points))
            logging.info('Regime Adaptation - LR Reduce at {}'.format(self._lr_points), extra=self._client)
        if args.fast_regime is True:
            end_lr = args.lr * ((args.workers_num * args.batch_size) // batch_baseline) / np.sqrt(args.workers_num)
            start_lr = args.lr / np.sqrt(args.workers_num)
            self._lr = end_lr
            self._start_lr = start_lr
            self._lr_increment_const = (end_lr - start_lr) / (args.iterations_per_epoch * 5)
            print('Fast Regime - Learning Rate Warm Up [{:.5f}]->[{:.5f}] In 5 Epochs'.format(start_lr, end_lr))
            logging.info('Fast Regime - Learning Rate Warm Up [{:.5f}]->[{:.5f}] In 5 Epochs'.format(start_lr, end_lr),
                         extra=self._client)
        else:
            self._start_lr = 0
            self._lr_increment_const = 0

        momentum = args.momentum if args.momentum >= 0 else 0
        self._optimizer = torch.optim.SGD(self._model.parameters(), args.lr,
                                          momentum=momentum,
                                          dampening=args.dampening,
                                          nesterov=args.nesterov,
                                          weight_decay=args.weight_decay)  # already done in train --- args.weight_decay)

        if momentum != args.momentum:  # pytorch 0.4 workaround negative momentum is not allowed
            for param_group in self._optimizer.param_groups:
                param_group['momentum'] = args.momentum

        self._shards_weights = list()
        weights = self._get_model_weights()
        for i in range(0, args.workers_num):
            self._shards_weights.append(deepcopy(weights))

    def _get_model_weights(self):
        parameters = {}
        for name, weight in self._model.named_parameters():
            parameters[name] = weight.data.clone()
        for name, val in self._model.named_modules():
            if 'bn' in name:
                parameters[name + '.running_mean'] = val.running_mean.clone()
                parameters[name + '.running_var'] = val.running_var.clone()
        return parameters

    def _get_model_gradients(self):
        gradients = {}
        for name, weight in self._model.named_parameters():
            gradients[name] = weight.grad.data.clone()
        return gradients

    def _set_model_gradients(self, gradients):
        for name, weight in self._model.named_parameters():
            if torch.cuda.is_available() is True:
                weight.grad = gradients[name].cuda()
            else:
                weight.grad = gradients[name]
        for name, val in self._model.named_modules():
            if 'bn' in name:
                if torch.cuda.is_available() is True:
                    val.running_mean = gradients[name + '.running_mean'].cuda()
                    val.running_var = gradients[name + '.running_var'].cuda()
                else:
                    val.running_mean = gradients[name + '.running_mean']
                    val.running_var = gradients[name + '.running_var']

    def _adjust_learning_rate(self, epoch, iteration):
        lr = self._start_lr + self._lr_increment_const * (iteration + self._iterations_per_epoch * epoch)
        if lr < self._lr and self._fast_regime is True:  # learning rate warm up phase
            logging.info('Warm up Learning Rate - [{:.5f}]'.format(lr), extra=self._client)
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = self._lr
            for r in self._lr_points:
                lr *= self._lr_factor ** int(epoch >= r)
            if self._current_lr != lr:
                print('Adjusting Learning Rate to [{0:.5f}]'.format(lr))
                logging.info('Adjusting Learning Rate to [{0:.5f}]'.format(lr), extra=self._client)
                self._current_lr = lr
                for param_group in self._optimizer.param_groups:
                    param_group['lr'] = lr
        return lr

    def _adjust_momentum(self, epoch, iteration):
        if self._m_off is False:
            return
        if epoch < 5:
            momentum = self._momentum
        else:
            momentum = 0
        if momentum != self._current_momentum:
            self._current_momentum = momentum
            self._lr = 1
            print('Adjusting Momentum to [{0:.3f}]'.format(momentum))
            logging.info('Adjusting Momentum to [{0:.3f}]'.format(momentum), extra=self._client)
            for param_group in self._optimizer.param_groups:
                param_group['momentum'] = momentum
            for param_group in self._optimizer.param_groups:
                param_group['dampening'] = momentum
        return momentum

    def _calc_workers_mean(self):
        mu_mean = {}
        mu_mean_norm = torch.zeros(1)
        keys = self._shards_weights[0].keys()
        for name in keys:
            mu_mean[name] = torch.zeros_like(self._shards_weights[0][name])
            for worker_id in range(0, self._workers_num):
                mu_mean[name].add_(self._shards_weights[worker_id][name])
            mu_mean[name].mul_(1 / self._workers_num)
            mu_mean_norm = mu_mean_norm + mu_mean[name].norm().cpu() ** 2
        return mu_mean, torch.sqrt(mu_mean_norm)

    def _step_norm(self, parameters):
        norm = 0
        for name, weight in self._model.named_parameters():
            norm += norm + (weight.data.add(parameters[name].data.mul(-1))).norm() ** 2
        return sqrt(norm)

    def push(self, worker_id, parameters, epoch, **kwargs):
        raise NotImplementedError

    def pull(self, worker_id):
        raise NotImplementedError

    def get_server_weights(self):
        return self._get_model_weights()

    def get_server_gradients(self):
        return self._get_model_gradients()

    def get_workers_mean_statistics(self):
        mu_mean, mu_mean_norm = self._calc_workers_mean()
        workers_norm_distances_mu = list()
        keys = self._shards_weights[0].keys()
        for worker_id in range(0, self._workers_num):
            norm = torch.zeros(1)
            for name in keys:
                norm.add_(mu_mean[name].add(self._shards_weights[worker_id][name].mul(-1)).norm().cpu() ** 2)
            workers_norm_distances_mu.append(torch.sqrt(norm) / mu_mean_norm)
        workers_norm_distances_mu = torch.cat(workers_norm_distances_mu)
        mean_distance = torch.mean(workers_norm_distances_mu).item()
        min_distance = torch.min(workers_norm_distances_mu).item()
        max_distance = torch.max(workers_norm_distances_mu).item()
        std_distance = torch.std(workers_norm_distances_mu, unbiased=False).item()
        return mean_distance, min_distance, max_distance, std_distance

    def get_mean_master_dist(self):
        mu_mean, mu_mean_norm = self._calc_workers_mean()
        mu_master = self._get_model_weights()
        norm = torch.zeros(1)
        keys = mu_master.keys()
        for name in keys:
            norm.add_(mu_mean[name].add(mu_master[name].mul(-1)).norm().cpu() ** 2)
        return norm.item()

    def get_workers_master_statistics(self):
        mu_master = self._get_model_weights()
        workers_norm_distances_mu = list()
        keys = self._shards_weights[0].keys()
        for worker_id in range(0, self._workers_num):
            norm = torch.zeros(1)
            for name in keys:
                norm.add_(mu_master[name].add(self._shards_weights[worker_id][name].mul(-1)).norm().cpu() ** 2)
            workers_norm_distances_mu.append(torch.sqrt(norm))
        workers_norm_distances_mu = torch.cat(workers_norm_distances_mu)
        mean_distance = torch.mean(workers_norm_distances_mu).item()
        min_distance = torch.min(workers_norm_distances_mu).item()
        max_distance = torch.max(workers_norm_distances_mu).item()
        std_distance = torch.std(workers_norm_distances_mu, unbiased=False).item()
        return mean_distance, min_distance, max_distance, std_distance


class ASGD(ParameterServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def push(self, worker_id, parameters, epoch, **kwargs):
        step_norm = self._step_norm(parameters)
        self._adjust_momentum(epoch, kwargs['iteration'])
        self._adjust_learning_rate(epoch, kwargs['iteration'])
        self._optimizer.zero_grad()
        self._set_model_gradients(deepcopy(parameters))
        self._optimizer.step()
        self._shards_weights[worker_id] = self._get_model_weights()
        return step_norm

    def pull(self, worker_id):
        return self._shards_weights[worker_id]


def _get_norm(parameters):
    return parameters['module.fc.weight'].norm().data.cpu().numpy()[0]
