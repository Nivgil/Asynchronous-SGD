import torch
import numpy as np
from copy import deepcopy
from math import sqrt
import logging


class ParameterServer(object):

    @staticmethod
    def get_server(mode, *args, **kwargs):
        return {'synchronous': ASGD, 'asynchronous': ASGD, 'elastic': EAMSGD}[mode](*args,
                                                                                    **kwargs)

    @staticmethod
    def get_lr_reduce_epochs(model):
        return {'resnet': [30, 60, 80], 'alexnet': [10, 15, 20, 25], 'wideresnet': [60, 120, 160]}[model]

    def __init__(self, model, args, **kwargs):
        self._model = deepcopy(model)
        self._workers_num = args.workers_num

        # learning rate initialization
        batch_baseline = args.baseline
        self._lr = args.lr * np.sqrt((args.workers_num * args.batch_size) // batch_baseline) / (args.workers_num)
        self._fast_im = args.fast_im
        self._lr_warm_up = args.lr_warm_up
        self._current_lr = args.lr
        self._m_off = args.m_off
        self._current_momentum = args.momentum
        self._lr_points = self.get_lr_reduce_epochs(args.model)
        self._lr_factor = 0.2 if args.model == 'wideresnet' else 0.1
        self._iterations_per_epoch = args.iterations_per_epoch
        self._momentum = args.momentum
        self._client = args.client

        if args.regime is True:
            alpha = args.workers_num * args.batch_size // batch_baseline
            self._lr_points = [x * alpha for x in self._lr_points]
            print('Regime Adaptation - LR Reduce at {}'.format(self._lr_points))
        if args.fast_im is True:
            end_lr = args.lr * ((args.workers_num * args.batch_size) // batch_baseline) / np.sqrt(args.workers_num)
            start_lr = args.lr / (args.workers_num)
            self._lr = end_lr
            self._start_lr = start_lr
            self._lr_increment_const = (end_lr - start_lr) / (args.iterations_per_epoch * 5)
            print('Fast ImageNet Mode - Warm Up [{:.5f}]->[{:.5f}] In 5 Epochs'.format(start_lr, end_lr))
        else:
            self._start_lr = 0
            self._lr_increment_const = 0
        if args.lr_warm_up is True:
            end_lr = args.lr
            start_lr = 0
            self._lr = end_lr
            self._start_lr = start_lr
            print('Learning Rate Warm Up [{:.5f}]->[{:.5f}]'.format(start_lr, end_lr))

        self._optimizer = torch.optim.SGD(self._model.parameters(), args.lr,
                                          momentum=args.momentum,
                                          dampening=args.dampening,
                                          nesterov=args.nesterov,
                                          weight_decay=0)#already done in train --- args.weight_decay)
        # # debug dampening
        # self._model_dampening = deepcopy(model)
        # self._optimizer_dampening = torch.optim.SGD(self._model_dampening.parameters(), 1,
        #                                             momentum=0.9,
        #                                             dampening=0.9,
        #                                             nesterov=args.nesterov,
        #                                             weight_decay=args.weight_decay)
        # # end debug dampening

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
        lr = self._start_lr + self._lr_increment_const * (iteration + self._iterations_per_epoch * epoch)
        if lr < self._lr and self._fast_im is True:  # learning rate warm up phase
            print('\nWarm up Learning Rate Fast ImageNet - [{:.5f}]'.format(lr))
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr
        else:
            if self._lr_warm_up is True:
                self._start_lr = self._start_lr * 0.9 + 1
                lr = 0.1 * self._start_lr
                print('Learning Rate Warm Up [{:.5f}]'.format(lr))
                for param_group in self._optimizer.param_groups:
                    param_group['lr'] = lr
                if np.abs(lr - self._lr) < 1e-3:
                    self._lr_warm_up = False
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
                norm.add_(mu_mean[name].add(self._shards_weights[worker_id][name].mul(-1)).norm() ** 2)
            workers_norm_distances_mu.append(torch.sqrt(norm) / mu_mean_norm)
        workers_norm_distances_mu = torch.cat(workers_norm_distances_mu)
        mean_distance = torch.mean(workers_norm_distances_mu)
        min_distance = torch.min(workers_norm_distances_mu)
        max_distance = torch.max(workers_norm_distances_mu)
        std_distance = torch.std(workers_norm_distances_mu, unbiased=False)
        return mean_distance, min_distance, max_distance, std_distance

    def get_mean_master_dist(self):
        mu_mean, mu_mean_norm = self._calc_workers_mean()
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
        step_norm = self._step_norm(parameters)
        self._adjust_momentum(epoch, kwargs['iteration'])
        self._adjust_learning_rate(epoch, kwargs['iteration'])
        self._optimizer.zero_grad()
        # # debug dampening
        # self._optimizer_dampening.zero_grad()
        # for name, weight in self._model_dampening.named_parameters():
        #     if torch.cuda.is_available() is True:
        #         weight.grad = parameters[name].cuda()
        #     else:
        #         weight.grad = parameters[name]
        # self._optimizer_dampening.step()
        # # debug end dampening

        self._set_model_gradients(deepcopy(parameters))
        self._optimizer.step()
        self._shards_weights[worker_id] = self._get_model_weights()
        return step_norm

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


def _get_norm(parameters):
    return parameters['module.fc.weight'].norm().data.cpu().numpy()[0]
