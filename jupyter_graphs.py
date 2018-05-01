import os
import json
import pickle
import pandas
from bokeh.plotting import figure
from bokeh.palettes import Category10 as palette
from bokeh.models import NumeralTickFormatter
from bokeh.io import output_file, show

import argparse


def create_graphs(sim_num=None, resolution='epoch', linear=False):
    CONFIGURATIONS_DIR = os.path.join(os.path.dirname(__file__))
    if sim_num is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--sim_num', action='store')
        parser.add_argument('--resolution', action='store', default='iteration')
        parser.add_argument('--linear', dest='scale', action='store_true')
        args = vars(parser.parse_args())
        sim_num = args['sim_num']
        resolution = args['resolution']
        linear = args['scale']
    folder_name = 'simulation_{}'.format(sim_num)
    colors = palette[10]
    if resolution == 'epoch':
        x_axis_label = 't [Epochs]'
    else:
        x_axis_label = 'Iterations'

    if linear is True:
        x_scale = 'linear'
    else:
        x_scale = 'log'

    p_loss = figure(plot_width=600, plot_height=600, min_border=10, min_border_left=50,
                    x_axis_label=x_axis_label, y_axis_label='L(w(t))',
                    title="Training & Test Loss", y_axis_type='log', x_axis_type=x_scale)
    p_loss.background_fill_color = "#fafafa"

    p_weight_norm = figure(plot_width=600, plot_height=600, min_border=10, min_border_left=50,
                           x_axis_label=x_axis_label, y_axis_label='||w(t)||',
                           title="The Norm of w(t)", x_axis_type=x_scale, y_axis_type=x_scale)
    p_weight_norm.background_fill_color = "#fafafa"

    p_gradient_norm = figure(plot_width=600, plot_height=600, min_border=10, min_border_left=50,
                             x_axis_label=x_axis_label, y_axis_label='||g||',
                             title="The Norm of Gradients", x_axis_type=x_scale, y_axis_type='log')
    p_gradient_norm.background_fill_color = "#fafafa"

    p_mean_master_dist = figure(plot_width=600, plot_height=600, min_border=10, min_border_left=50,
                                x_axis_label=x_axis_label, y_axis_label='Distance Norm',
                                title="Distance Norm Between Mean & Master Weights", x_axis_type=x_scale,
                                y_axis_type='log')
    p_mean_master_dist.background_fill_color = "#fafafa"

    p_weights_mean_distances = figure(plot_width=600, plot_height=600, min_border=10, min_border_left=50,
                                      x_axis_label=x_axis_label, y_axis_label='Distance',
                                      title="Distance of Workers From Mean")
    p_weights_mean_distances.background_fill_color = "#fafafa"

    p_weights_master_distances = figure(plot_width=600, plot_height=600, min_border=10, min_border_left=50,
                                        x_axis_label=x_axis_label, y_axis_label='Distance',
                                        title="Distance of Workers From Master")
    p_weights_master_distances.background_fill_color = "#fafafa"

    p_error = figure(plot_width=600, plot_height=600, min_border=10, min_border_left=50,
                     x_axis_label=x_axis_label, y_axis_label='Error Rate',
                     title="Training & Test Error", x_axis_type=x_scale)
    p_error.background_fill_color = "#fafafa"
    p_error.yaxis[0].formatter = NumeralTickFormatter(format="0.0%")

    idx = -1
    for file in os.listdir(os.path.join(CONFIGURATIONS_DIR, folder_name)):
        if file.endswith('.log'):
            continue
        with open(os.path.join(CONFIGURATIONS_DIR, folder_name, file), 'rb') as pickle_in:
            stats_test, stats_train = pickle.load(pickle_in)
        with open(os.path.join(CONFIGURATIONS_DIR, folder_name, file + '.log'), 'rb') as log_file:
            params_dict = json.load(log_file)
        log_table = Table(params_dict)
        for key in params_dict.keys():
            log_table.table[key].append(params_dict[key])
        idx += 1
        legend = params_dict['optimizer'] + '_' + str(params_dict['batch_size']) + '_' + str(params_dict['workers_num'])
        stats_train.export_data(handle_loss=p_loss,
                                handle_error=p_error,
                                handle_weight_norm=p_weight_norm,
                                handle_gradient_norm=p_gradient_norm,
                                handle_mean_distance=p_weights_mean_distances,
                                handle_master_distance=p_weights_master_distances,
                                handle_mean_master_dist=p_mean_master_dist,
                                legend=legend,
                                color=colors[idx % 10],
                                line_dash='solid',
                                resolution=resolution)
        stats_test.export_data(handle_loss=p_loss,
                               handle_error=p_error,
                               legend=legend,
                               color=colors[idx % 10],
                               line_dash='dashed',
                               resolution=resolution)
        min_score_test, mean_score_test = stats_test.get_scores()
    p_loss.legend.click_policy = "hide"
    p_loss.legend.location = 'bottom_left'
    p_error.legend.click_policy = "hide"
    p_weight_norm.legend.click_policy = "hide"
    p_weight_norm.legend.location = "top_left"
    p_gradient_norm.legend.click_policy = "hide"
    p_mean_master_dist.legend.click_policy = "hide"
    p_gradient_norm.legend.location = "bottom_left"
    df = pandas.DataFrame(log_table.table).transpose()
    df.rename_axis("Parameters", axis="columns")
    styles = [
        hover(),
        dict(selector="th", props=[("font-size", "110%"),
                                   ("text-align", "center")]),
        dict(selector="caption", props=[("caption-side", "bottom")])
    ]
    table_html = (df.style.set_table_styles(styles)).render()

    return [p_loss, p_error, p_weight_norm, p_gradient_norm, p_weights_mean_distances, p_weights_master_distances,
            p_mean_master_dist], min_score_test, mean_score_test, table_html


if __name__ == '__main__':
    create_graphs()


def hover(hover_color="#ffff99"):
    return dict(selector="tr:hover",
                props=[("background-color", "%s" % hover_color)])


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Table(metaclass=Singleton):
    def __init__(self, params_dict=None):
        keys = params_dict.keys()
        self.table = dict()
        for key in keys:
            self.table[key] = list()
