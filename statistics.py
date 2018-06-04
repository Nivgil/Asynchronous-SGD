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
        self._step_norm = []
        self._snr_1_hist = []
        self._snr_2_hist = []
        self._weights_mean_distance_stats = []
        self._weights_master_distance_stats = []
        self._mean_master_dist = []
        self._epochs = args.epochs
        self._log = args
        self._sim_num = args.sim_num
        self._iterations_per_epoch = args.iterations_per_epoch
        self._dataset = args.dataset
        self._model = args.model

    def save_loss(self, loss):
        self._loss.append(loss)

    def save_error(self, error):
        self._error.append(error / 100)

    def save_error_top5(self, error):
        self._error_top5.append(error / 100)

    def save_weight_norm(self, weights_dict):
        if self._model == 'alexnet':
            norm = weights_dict['module.classifier.0.weight'].norm()  #TODO
        else:
            norm = weights_dict['module.fc.weight'].norm()
        self._weight_norm.append(norm)

    def save_gradient_norm(self, weights_dict):
        if self._model == 'alexnet':
            norm = weights_dict['module.classifier.0.weight'].norm()
        else:
            norm = weights_dict['module.fc.weight'].norm()
        import ipdb; ipdb.set_trace()
        self._gradient_norm.append(norm.data.cpu().numpy()[0])

    def save_step_norm(self, step_norm):
        self._step_norm.append(step_norm)

    def save_weight_mean_dist(self, distance):
        self._weights_mean_distance_stats.append(distance)

    def save_weight_master_dist(self, distance):
        self._weights_master_distance_stats.append(distance)

    def save_mean_master_dist(self, dist):
        self._mean_master_dist.append(dist.numpy()[0])

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

    def _visualize_mean_master_dist(self, handle=None, legend=None, color=None, line_dash=None, resolution=None):
        if handle is None:
            return
        distance = self._mean_master_dist
        if resolution == 'epoch':
            t = np.arange(1, self._epochs + 1)
        else:
            t = np.arange(self._iterations_per_epoch, (self._epochs + 1) * self._iterations_per_epoch,
                          self._iterations_per_epoch)
        handle.line(t, distance, line_width=3, line_dash=line_dash, legend=legend, line_color=color)

    def _visualize_weights_mean_distances(self, handle=None, resolution=None):
        if handle is None:
            return
        weights_stats = self._weights_mean_distance_stats
        if resolution == 'epoch':
            t = np.arange(1, self._epochs + 1)
        else:
            t = np.arange(self._iterations_per_epoch, (self._epochs + 1) * self._iterations_per_epoch,
                          self._iterations_per_epoch)
        max_points = [stats[2] for stats in weights_stats]
        min_points = [stats[1] for stats in weights_stats]
        mean_points = [stats[0] for stats in weights_stats]
        std_points_positive = [stats[3] + stats[0] for stats in weights_stats]
        std_points_negative = [-stats[3] + stats[0] for stats in weights_stats]
        # stems
        handle.segment(t, max_points, t, mean_points, line_color='black')
        handle.segment(t, min_points, t, mean_points, line_color='black')
        # boxes
        handle.vbar(t, 0.25, mean_points, std_points_positive, fill_color="#084594", line_color="black")
        handle.vbar(t, 0.25, std_points_negative, mean_points, fill_color="#f03b20", line_color="black")
        # whiskers
        handle.rect(t, max_points, 0.2, 0.00001, line_color='black')
        handle.rect(t, min_points, 0.2, 0.00001, line_color='black')

    def _visualize_weights_master_distances(self, handle=None, resolution=None):
        if handle is None:
            return
        weights_stats = self._weights_master_distance_stats
        if resolution == 'epoch':
            t = np.arange(1, self._epochs + 1)
        else:
            t = np.arange(self._iterations_per_epoch, (self._epochs + 1) * self._iterations_per_epoch,
                          self._iterations_per_epoch)
        max_points = [stats[2] for stats in weights_stats]
        min_points = [stats[1] for stats in weights_stats]
        mean_points = [stats[0] for stats in weights_stats]
        std_points_positive = [stats[3] + stats[0] for stats in weights_stats]
        std_points_negative = [-stats[3] + stats[0] for stats in weights_stats]
        # stems
        handle.segment(t, max_points, t, mean_points, line_color='black')
        handle.segment(t, min_points, t, mean_points, line_color='black')
        # boxes
        handle.vbar(t, 0.25, mean_points, std_points_positive, fill_color="#084594", line_color="black")
        handle.vbar(t, 0.25, std_points_negative, mean_points, fill_color="#f03b20", line_color="black")
        # whiskers
        handle.rect(t, max_points, 0.2, 0.00001, line_color='black')
        handle.rect(t, min_points, 0.2, 0.00001, line_color='black')

    def export_data(self, handle_loss=None, handle_error=None, handle_gradient_norm=None, handle_weight_norm=None,
                    handle_mean_distance=None, handle_master_distance=None, legend=None, color=None, line_dash=None,
                    handle_mean_master_dist=None, resolution=None):
        self._visualize_loss(handle_loss, legend, color, line_dash, resolution)
        self._visualize_error(handle_error, legend, color, line_dash, resolution)
        self._visualize_weight_norm(handle_weight_norm, legend, color, resolution)
        self._visualize_gradient_norm(handle_gradient_norm, legend, color, resolution)
        self._visualize_weights_mean_distances(handle_mean_distance, resolution)
        self._visualize_weights_master_distances(handle_master_distance, resolution)
        self._visualize_mean_master_dist(handle_mean_master_dist, legend, color, line_dash, resolution)

    def get_scores(self, handle=None):
        error = self._error
        min_score = np.min(error) * 100
        if handle is not None:
            min_index = np.argmin(error) + 1
            handle.circle(min_index, min_score/100, color='red', size=5, alpha=0.5)
        # use 15% of last epochs
        start_epoch = int(self._epochs * 0.85)
        error = error[start_epoch:]
        mean_score = np.mean(error) * 100
        last_score = error[-1] * 100
        if self._dataset == 'imagenet':
            error_top5 = self._error_top5
            error_top5 = error_top5[start_epoch:]
            min_score_top5 = min(error_top5) * 100
            mean_score_top5 = np.mean(error_top5) * 100
            # return min_score, mean_score, min_score_top5, mean_score_top5
        return min_score, mean_score, last_score

    def generic_future(self):
        pass
