import numpy as np
import torch


class Statistics(object):

    @staticmethod
    def get_statistics(mode, *args, **kwargs):
        return {'image_classification': StatImage}[mode](*args, **kwargs)


class StatImage(object):
    def __init__(self, args):

        self._loss = []
        self._error = []
        self._weight_norm = []
        self._gradient_norm = []
        self._epochs = args.epochs
        self._log = args
        self._sim_num = args.sim_num
        # self._folder = params.folder_name
        self._iterations_per_epoch = args.iterations_per_epoch

    def save_loss(self, loss):
        self._loss.append(loss)

    def save_error(self, error):
        self._error.append(error)

    def save_weight_norm(self, weights_dict):
        norm = torch.zeros(1)
        for name in weights_dict:
            norm = norm + torch.sum((weights_dict[name]).view(-1, 1) ** 2)
        self._weight_norm.append(torch.sqrt(norm).numpy()[0])

    def save_gradient_norm(self, weights_dict):
        norm = torch.zeros(1)
        for name in weights_dict:
            norm = norm + torch.sum((weights_dict[name]).view(-1, 1) ** 2)
        self._gradient_norm.append(torch.sqrt(norm).numpy()[0])

    def _visualize_weight_norm(self, handle=None, legend=None, color=None, resolution=None):
        if handle is None:
            return
        norm = self._weight_norm
        if resolution == 'epoch':
            t = np.arange(1, self._epochs + 1)
        else:
            t = np.arange(self._iterations_per_epoch, (self._epochs + 1) * self._iterations_per_epoch,
                          self._iterations_per_epoch)
        handle.line(t, norm, line_width=3, line_dash='solid', legend=legend, line_color=color)

    def _visualize_gradient_norm(self, handle=None, legend=None, color=None, resolution=None):
        if handle is None:
            return
        norm = self._gradient_norm
        if resolution == 'epoch':
            t = np.arange(1, self._epochs + 1)
        else:
            t = np.arange(self._iterations_per_epoch, (self._epochs + 1) * self._iterations_per_epoch,
                          self._iterations_per_epoch)
        handle.line(t, norm, line_width=3, line_dash='solid', legend=legend, line_color=color)

    def _visualize_loss(self, handle=None, legend=None, color=None, line_dash=None, resolution=None):
        if handle is None:
            return
        loss = self._loss
        if resolution == 'epoch':
            t = np.arange(1, self._epochs + 1)
        else:
            t = np.arange(self._iterations_per_epoch, (self._epochs + 1) * self._iterations_per_epoch,
                          self._iterations_per_epoch)
        handle.line(t, loss, line_width=3, line_dash=line_dash, legend=legend, line_color=color)

    def _visualize_error(self, handle=None, legend=None, color=None, line_dash=None, resolution=None):
        if handle is None:
            return
        error = self._error
        if resolution == 'epoch':
            t = np.arange(1, self._epochs + 1)
        else:
            t = np.arange(self._iterations_per_epoch, (self._epochs + 1) * self._iterations_per_epoch,
                          self._iterations_per_epoch)
        handle.line(t, error, line_width=3, line_dash=line_dash, legend=legend, line_color=color)

    def export_data(self, handle_loss=None, handle_error=None, handle_gradient_norm=None, handle_weight_norm=None,
                    legend=None, color=None, line_dash=None, resolution=None):
        self._visualize_loss(handle_loss, legend, color, line_dash, resolution)
        self._visualize_error(handle_error, legend, color, line_dash, resolution)
        self._visualize_weight_norm(handle_weight_norm, legend, color, resolution)
        self._visualize_gradient_norm(handle_gradient_norm, legend, color, resolution)
