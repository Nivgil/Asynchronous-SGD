import argparse
import os
import numpy as np
import json
import pickle
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.layouts import column, row
from bokeh.plotting import figure, curdoc, output_file, show, save
from bokeh.palettes import Category10 as palette
from bokeh.models import NumeralTickFormatter
from bokeh.io import export_png, export_svgs


def create_graphs(sim_num=None, resolution='epoch', linear=False):
    CONFIGURATIONS_DIR = os.path.join(os.path.dirname(__file__))
    if sim_num is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--sim_num', action='store')
        parser.add_argument('--resolution', action='store', default='epoch')
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
                             title="The Norm of Gradients", x_axis_type=x_scale, y_axis_type='linear')
    p_gradient_norm.background_fill_color = "#fafafa"

    p_mean_master_dist = figure(plot_width=600, plot_height=600, min_border=10, min_border_left=50,
                                x_axis_label=x_axis_label, y_axis_label='Distance Norm',
                                title="Distance Norm Between Mean & Master Weights", x_axis_type=x_scale,
                                y_axis_type='linear')
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
        with open(os.path.join(CONFIGURATIONS_DIR, folder_name, file + '_param.log'), 'rb') as log_file:
            params_dict = json.load(log_file)
        idx += 1
        legend = str(params_dict['batch_size']) + '_' + str(params_dict['workers_num'])
        stats_train.export_data(handle_loss=p_loss,
                                handle_error=p_error,
                                handle_weight_norm=p_weight_norm,
                                handle_gradient_norm=p_gradient_norm,
                                handle_mean_distance=p_weights_mean_distances,
                                handle_master_distance=p_weights_master_distances,
                                handle_mean_master_dist=p_mean_master_dist,
                                legend=legend,
                                color=colors[idx % 10],
                                line_dash='dashed',
                                resolution=resolution)
        stats_test.export_data(handle_loss=p_loss,
                               handle_error=p_error,
                               legend=legend,
                               color=colors[idx % 10],
                               line_dash='solid',
                               resolution=resolution)
        min_score_test, mean_score_test, last_score_test = stats_test.get_scores(p_error)
        with open(folder_name + '/' + file + '_' + 'results.txt', 'w') as results_out:
            results_out.write('minimum error: {0:.3f}%'.format(min_score_test) + '\n')
            results_out.write('mean error: {0:.3f}%'.format(mean_score_test) + '\n')
            results_out.write('last error: {0:.3f}%'.format(last_score_test) + '\n')
    p_loss.legend.click_policy = "hide"
    p_loss.legend.location = 'bottom_left'
    p_error.legend.click_policy = "hide"
    p_weight_norm.legend.click_policy = "hide"
    p_weight_norm.legend.location = "top_left"
    p_gradient_norm.legend.click_policy = "hide"
    p_gradient_norm.legend.location = "bottom_left"
    grid = column(row(p_loss, p_error), row(p_weight_norm, p_gradient_norm),
                  row(p_weights_mean_distances, p_weights_master_distances), p_mean_master_dist)
    html_norm = file_html(grid, CDN, folder_name)
    with open(folder_name + '/' + folder_name + '.html', 'a') as html_file:
        html_file.write(html_norm)
    graph_path = folder_name + '/error.png'
    export_png(p_error, filename=graph_path)
    return graph_path


def create_averaged_graph(sim_nums=None, resolution='epoch', linear=True, legend=None):
    assert sim_nums is not None
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
                             title="The Norm of Gradients", x_axis_type=x_scale, y_axis_type='linear')
    p_gradient_norm.background_fill_color = "#fafafa"

    p_error = figure(plot_width=600, plot_height=600, min_border=10, min_border_left=50,
                     x_axis_label=x_axis_label, y_axis_label='Error Rate',
                     title="Training & Test Error", x_axis_type=x_scale)
    p_error.background_fill_color = "#fafafa"
    p_error.yaxis[0].formatter = NumeralTickFormatter(format="0.0%")

    for idx, sim_set in enumerate(sim_nums):
        train_loss, train_error, weight_norm, gradient_norm, test_loss, test_error = get_average_graph_values(sim_set)

        t = np.arange(1, len(train_loss) + 1)
        p_loss.line(t, train_loss, line_width=3, line_dash='dashed', legend=legend[idx] + ' - train',
                    line_color=colors[idx])
        p_loss.line(t, test_loss, line_width=3, line_dash='solid', legend=legend[idx] + ' - test',
                    line_color=colors[idx])

        p_error.line(t, train_error, line_width=3, line_dash='dashed', legend=legend[idx] + ' - train',
                     line_color=colors[idx])
        p_error.line(t, test_error, line_width=3, line_dash='solid', legend=legend[idx] + ' - test',
                     line_color=colors[idx])
        p_weight_norm.line(t, weight_norm, line_width=3, line_dash='solid', legend=legend[idx], line_color=colors[idx])
        p_gradient_norm.line(t, gradient_norm, line_width=3, line_dash='solid', legend=legend[idx],
                             line_color=colors[idx])

    p_loss.legend.click_policy = "hide"
    p_loss.legend.location = 'bottom_left'
    p_error.legend.click_policy = "hide"
    p_weight_norm.legend.click_policy = "hide"
    p_weight_norm.legend.location = "top_left"
    p_gradient_norm.legend.click_policy = "hide"
    p_gradient_norm.legend.location = "bottom_left"

    p_loss.output_backend = "svg"
    export_svgs(p_loss, filename="loss.svg")

    p_loss.output_backend = "svg"
    export_svgs(p_loss, filename="loss.svg")
    p_error.output_backend = "svg"
    export_svgs(p_error, filename="error.svg")
    p_weight_norm.output_backend = "svg"
    export_svgs(p_weight_norm, filename="weights_norm.svg")
    p_gradient_norm.output_backend = "svg"
    export_svgs(p_gradient_norm, filename="gradient_norm.svg")


def get_average_graph_values(sim_nums=None):
    assert sim_nums is not None
    CONFIGURATIONS_DIR = os.path.join(os.path.dirname(__file__))
    loss_train = list()
    error_train = list()
    weight_norm = list()
    gradient_norm = list()
    loss_test = list()
    error_test = list()
    for sim_num in sim_nums:
        folder_name = 'outputs/simulation_{}'.format(sim_num)
        for file in os.listdir(os.path.join(CONFIGURATIONS_DIR, folder_name)):
            if file.endswith('.log'):
                continue
            with open(os.path.join(CONFIGURATIONS_DIR, folder_name, file), 'rb') as pickle_in:
                stats_test, stats_train = pickle.load(pickle_in)
            with open(os.path.join(CONFIGURATIONS_DIR, folder_name, file + '_param.log'), 'rb') as log_file:
                params_dict = json.load(log_file)
            loss_train.append(stats_train._loss)
            error_train.append(stats_train._error)
            weight_norm.append(stats_train._weight_norm)
            gradient_norm.append(stats_train._gradient_norm)
            loss_test.append(stats_test._loss)
            error_test.append(stats_test._error)
            break
    average_train_loss = np.zeros(len(loss_train[0]))
    average_train_error = np.zeros(len(error_train[0]))
    average_weight_norm = np.zeros(len(weight_norm[0]))
    average_gradient_norm = np.zeros(len(gradient_norm[0]))
    average_test_loss = np.zeros(len(loss_test[0]))
    average_test_error = np.zeros(len(error_test[0]))

    for sim in range(len(sim_nums)):
        average_train_loss = average_train_loss + loss_train[sim]
        average_train_error = average_train_error + error_train[sim]
        average_weight_norm = average_weight_norm + weight_norm[sim]
        average_gradient_norm = average_gradient_norm + gradient_norm[sim]
        average_test_loss = average_test_loss + loss_test[sim]
        average_test_error = average_test_error + error_test[sim]
    average_train_loss /= len(sim_nums)
    average_train_error /= len(sim_nums)
    average_weight_norm /= len(sim_nums)
    average_gradient_norm /= len(sim_nums)
    average_test_loss /= len(sim_nums)
    average_test_error /= len(sim_nums)

    return average_train_loss, average_train_error, average_weight_norm, average_gradient_norm, average_test_loss, average_test_error


if __name__ == '__main__':
    create_averaged_graph(sim_nums=[[710, 711, 712, 713, 714], [715, 716, 717, 718, 719]],
                          legend=['no momentum', 'momentum'])
