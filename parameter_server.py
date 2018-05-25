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
        self._current_lr = args.lr
        self._lr_points = list()
        alpha = args.workers_num * args.batch_size // batch_baseline
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            self._lr_points.append(60 * alpha)
            self._lr_points.append(120 * alpha)
            self._lr_points.append(160 * alpha)
        else:
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

    def _adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
        lr = self._lr
        for r in self._lr_points:
            lr *= 0.2 ** int(epoch >= r)
        # lr = self._lr * ((0.2 ** int(epoch >= self._lr_points[0])) *
        #                  (0.2 ** int(epoch >= self._lr_points[1])) *
        #                  (0.2 ** int(epoch >= self._lr_points[2])))
        if self._current_lr != lr:
            print('Adjusting Learning Rate to [{0:.5f}]'.format(lr))
            self._current_lr = lr
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr
        return lr

    def push(self, worker_id, parameters, epoch, **kwargs):
        raise NotImplementedError

    def pull(self, worker_id):
        raise NotImplementedError

    def get_server_weights(self):
        return self._get_model_weights()

    def get_server_gradients(self):
        return self._get_model_gradients()

    def debugger(self, worker_id_1, worker_id_2):
        for name, weight in self._model.named_parameters():
            if torch.eq(self._shards_weights[worker_id_1][name],
                        self._shards_weights[worker_id_2][name]).byte().all() is False:
                return False

        return True

    def calc_norm(self, weights=None):
        norm = 0
        if weights is None:
            for name, val in self._model.named_parameters():
                norm += val.norm()
        else:
            for name, val in weights.items():
                norm += val.norm()
        print(norm)

class ASGD(ParameterServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def push(self, worker_id, parameters, epoch, **kwargs):
        self._adjust_learning_rate(epoch)
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
