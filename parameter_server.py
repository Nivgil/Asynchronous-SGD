import torch
import numpy as np
from copy import deepcopy
from math import sqrt
import time


class ParameterServer(object):

    @staticmethod
    def get_server(mode, *args, **kwargs):
        return {'synchronous': ASGD, 'asynchronous': ASGD, 'elastic': EAMSGD}[mode](*args,
                                                                                    **kwargs)

    def __init__(self, model, args, **kwargs):
        self._model = deepcopy(model)
        self._workers_num = args.workers_num

        # learning rate initialization
        batch_baseline = 128
        self._lr = args.lr * np.sqrt(args.batch_size // batch_baseline) / np.sqrt(args.workers_num)
        self._fast_im = args.fast_im
        self._current_lr = args.lr
        self._lr_points = list()
        self._iterations_per_epoch = args.iterations_per_epoch
        if args.regime is True:
            print('No Regime Adaptation')
            alpha = 1
        else:
            print('Regime Adaptation')
            alpha = args.workers_num * args.batch_size // batch_baseline
        if args.fast_im is True:
            print('Fast ImageNet Mode')
            self._lr = self._lr * np.sqrt(args.batch_size // batch_baseline) * np.sqrt(args.workers_num)  # linear scaling
            start_lr = args.lr / np.sqrt(args.workers_num)
            self._start_lr = start_lr
            end_lr = args.lr * (args.batch_size // batch_baseline)
            self._lr_increment_const = (end_lr - start_lr) / (args.iterations_per_epoch * 5)
            alpha = 1
        else:
            self._lr_increment_const = 0
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            self._lr_factor = 0.2
            self._lr_points.append(60 * alpha)
            self._lr_points.append(120 * alpha)
            self._lr_points.append(160 * alpha)
        else:
            self._lr_factor = 0.2
            self._lr_points.append(10 * alpha)
            self._lr_points.append(15 * alpha)
            self._lr_points.append(20 * alpha)
            self._lr_points.append(25 * alpha)

        self._optimizer = torch.optim.SGD(self._model.parameters(), args.lr,
                                          momentum=args.momentum,
                                          nesterov=args.nesterov,
                                          weight_decay=args.weight_decay)
        self._shards_weights = list()
        weights = self._get_model_weights()
        for i in range(0, args.workers_num):
            self._shards_weights.append(deepcopy(weights))

    def _get_model_weights(self):
        parameters = {}
        for name, weight in self._model.named_parameters():
            parameters[name] = weight.data.clone()
        return parameters

    def _set_model_weights(self, weights):
        for name, weight in self._model.named_parameters():
            if torch.cuda.is_available() is True:
                weight = weights[name].cuda()
            else:
                weight = weights[name]

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

    def _adjust_learning_rate(self, epoch, iteration):
        if epoch < 5 and self._fast_im is True:  # learning rate warm up phase
            lr = self._start_lr + self._lr_increment_const * (iteration + self._iterations_per_epoch*epoch)
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = self._lr
            for r in self._lr_points:
                lr *= self._lr_factor ** int(epoch >= r)
            if self._current_lr != lr:
                print('Adjusting Learning Rate to [{0:.5f}]'.format(lr))
                self._current_lr = lr
                for param_group in self._optimizer.param_groups:
                    param_group['lr'] = lr
        return lr

    def _calc_workers_mean(self):
        mu_mean = {}
        keys = self._shards_weights[0].keys()
        for name in keys:
            mu_mean[name] = torch.zeros_like(self._shards_weights[0][name])
            for worker_id in range(0, self._workers_num):
                mu_mean[name].add_(self._shards_weights[worker_id][name])
            mu_mean[name].mul_(1 / self._workers_num)
        return mu_mean

    def push(self, worker_id, parameters, epoch, **kwargs):
        raise NotImplementedError

    def pull(self, worker_id):
        raise NotImplementedError

    def get_server_weights(self):
        return self._get_model_weights()

    def get_server_gradients(self):
        return self._get_model_gradients()

    def get_workers_mean_statistics(self):
        mu_mean = self._calc_workers_mean()
        workers_norm_distances_mu = list()
        keys = self._shards_weights[0].keys()
        for worker_id in range(0, self._workers_num):
            norm = torch.zeros(1)
            for name in keys:
                norm.add_(mu_mean[name].add(self._shards_weights[worker_id][name].mul(-1)).norm() ** 2)
            workers_norm_distances_mu.append(torch.sqrt(norm))
        workers_norm_distances_mu = torch.cat(workers_norm_distances_mu)
        mean_distance = torch.mean(workers_norm_distances_mu)
        min_distance = torch.min(workers_norm_distances_mu)
        max_distance = torch.max(workers_norm_distances_mu)
        std_distance = torch.std(workers_norm_distances_mu, unbiased=False)
        return mean_distance, min_distance, max_distance, std_distance

    def get_mean_master_dist(self):
        mu_mean = self._calc_workers_mean()
        mu_master = self._get_model_weights()
        norm = torch.zeros(1)
        keys = mu_master.keys()
        for name in keys:
            norm.add_(mu_mean[name].add(mu_master[name].mul(-1)).norm() ** 2)
        return norm

    def get_workers_master_statistics(self):

        mu_master = self._get_model_weights()
        workers_norm_distances_mu = list()
        keys = self._shards_weights[0].keys()
        for worker_id in range(0, self._workers_num):
            norm = torch.zeros(1)
            for name in keys:
                norm.add_(mu_master[name].add(self._shards_weights[worker_id][name].mul(-1)).norm() ** 2)
            workers_norm_distances_mu.append(torch.sqrt(norm))
        workers_norm_distances_mu = torch.cat(workers_norm_distances_mu)
        mean_distance = torch.mean(workers_norm_distances_mu)
        min_distance = torch.min(workers_norm_distances_mu)
        max_distance = torch.max(workers_norm_distances_mu)
        std_distance = torch.std(workers_norm_distances_mu, unbiased=False)

        return mean_distance, min_distance, max_distance, std_distance

    def debugger(self, worker_id_1, worker_id_2):
        for name, weight in self._model.named_parameters():
            if torch.eq(self._shards_weights[worker_id_1][name],
                        self._shards_weights[worker_id_2][name]).byte().all() is False:
                return False

        return True


class ASGD(ParameterServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def push(self, worker_id, parameters, epoch, **kwargs):
        self._adjust_learning_rate(epoch, kwargs['iteration'])
        self._optimizer.zero_grad()
        self._set_model_gradients(parameters)
        self._optimizer.step()
        self._shards_weights[worker_id] = self._get_model_weights()

    def pull(self, worker_id):
        return self._shards_weights[worker_id]


class EAMSGD(ParameterServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rho = args[1].rho
        self._tau = args[1].tau
        self._master_weights = self._get_model_weights()

        self._shards_velocity = list()
        init_velocity = dict()

        for name, val in self._master_weights.items():
            init_velocity[name] = torch.zeros_like(val)
        for i in range(0, self._workers_num):
            self._shards_velocity.append(deepcopy(init_velocity))

    def push(self, worker_id, parameters, epoch, **kwargs):
        worker_gradients = parameters
        lr = self._adjust_learning_rate(epoch)
        alpha = lr * self._rho
        if kwargs['tau'] % self._tau == 0:
            for name, val in self._master_weights.items():
                temp = torch.add(self._shards_weights[worker_id][name], -1, val)
                self._shards_weights[worker_id][name].add_(temp.mul_(-alpha))
                val.add_(temp.mul_(-1))
        self._set_model_weights(self._shards_weights[worker_id])
        self._optimizer.zero_grad()
        self._set_model_velocity(worker_id)
        self._set_model_gradients(worker_gradients)
        self._optimizer.step()

    def pull(self, worker_id):
        return self._shards_weights[worker_id]

    def get_server_weights(self):
        return self._master_weights

    def _set_model_velocity(self, worker_id):
        for name, val in self._model.named_parameters():
            self._optimizer.state[val]['momentum_buffer'] = self._shards_velocity[worker_id][name]

#
# def training_regime(regime, lr, batch_size, workers_num, dataset):
#     batch_baseline = 128
#     if regime == 'standard' or regime == 'regime_adaptation':
#         lr = lr * np.sqrt(batch_size // batch_baseline) / np.sqrt(workers_num)
#         lr_points = list()
#         if regime == 'standard':
#             print('No Regime Adaptation')
#             alpha = 1
#         else:
#             print('Regime Adaptation')
#             alpha = workers_num * batch_size // batch_baseline
#         if dataset == 'cifar10' or dataset == 'cifar100':
#             lr_factor = 0.2
#             lr_points.append(60 * alpha)
#             lr_points.append(120 * alpha)
#             lr_points.append(160 * alpha)
#         else:  # imagenet
#             lr_factor = 0.2
#             lr_points.append(10 * alpha)
#             lr_points.append(15 * alpha)
#             lr_points.append(20 * alpha)
#             lr_points.append(25 * alpha)
#     if regime == '1_hour_imagenet':
#         start_lr = lr / np.sqrt(workers_num)
#         end_lr = lr * (batch_size // batch_baseline) / np.sqrt(workers_num)
#         lr_increment_const = (end_lr - start_lr) / (args.iterations_per_epoch * 5)
#         lr_points = list()
#         print('Regime Adaptation')
#         alpha = 1  # workers_num * batch_size // batch_baseline
#         lr_factor = 0.2
#         lr_points.append(10 * alpha)
#         lr_points.append(15 * alpha)
#         lr_points.append(20 * alpha)
#         lr_points.append(25 * alpha)
