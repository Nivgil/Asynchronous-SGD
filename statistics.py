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
        self._error_top5 = []
        self._weight_norm = []
        self._gradient_norm = []
        self._epochs = args.epochs
        self._log = args
        self._sim_num = args.sim_num
        self._iterations_per_epoch = args.iterations_per_epoch
        self._dataset = args.dataset

    def save_loss(self, loss):
        self._loss.append(loss)

    def save_error(self, error):
        self._error.append(error/100)

    def save_error_top5(self, error):
        self._error_top5.append(error/100)

    def save_weight_norm(self, weights_dict):
        norm = torch.zeros(1)
        if self._dataset == 'imagenet':
            norm = norm + weights_dict['module.classifier.0.weight'].norm() ** 2
        else:
            norm = norm + weights_dict['module.fc.weight'].norm() ** 2
        self._weight_norm.append(torch.sqrt(norm).numpy()[0])

    def save_gradient_norm(self, weights_dict):
        norm = torch.zeros(1)
        if self._dataset == 'imagenet':
            norm = norm + weights_dict['module.classifier.0.weight'].norm() ** 2
        else:
            norm = norm + weights_dict['module.fc.weight'].norm() ** 2
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
        # error = [e.cpu().numpy() for e in self._error]
        error = self._error
        # error = self._error
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

    def get_scores(self):
        error = self._error
        # use 15% of last epochs
        start_epoch = int(self._epochs * 0.85)
        error = error[start_epoch:]
        min_score = np.min(error)*100
        mean_score = np.mean(error)*100
        if self._dataset == 'imagenet':
            error_top5 = self._error_top5
            error_top5 = error_top5[start_epoch:]
            min_score_top5 = min(error_top5)*100
            mean_score_top5 = np.mean(error_top5)*100
            # return min_score, mean_score, min_score_top5, mean_score_top5
        return min_score, mean_score

    def generic_future(self):
        pass
