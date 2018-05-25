import pickle
import json
import os
import numpy as np
import torch
import random
import train
from configuration import configuration
from email_notification import send_notification
from graphs import create_graphs
import warnings
import logging
import socket


def main():
    args = configuration()
    base_name = args.name

    for idx in range(1, 2):
        args.sim_num = idx * 10
        args.name = base_name + '_{}'.format(args.id)
        min_error, mean_error = exec_unit(args)
        graph_path = create_graphs(sim_num=args.id, linear=True)

        message = 'Simulation Number {0} Completed\nMin Error - {1:.3f}\nMean Error - {2:.3f}'.format(args.id,
                                                                                                      min_error,
                                                                                                      mean_error)
        send_notification(message, vars(args), graph_path, args)


def exec_unit(args=None):
    if args is None:
        args = configuration()

    CONFIGURATIONS_DIR = os.path.join(os.path.dirname(__file__))
    folder_name = 'simulation_{}'.format(args.id)
    path_name = os.path.join(CONFIGURATIONS_DIR, folder_name)
    if not os.path.exists(path_name):
        os.mkdir(path_name)
    param_log_name = os.path.join(path_name, args.name + '_param.log')
    if os.path.exists(param_log_name):
        raise Exception('Log file already exists')
    args.client = {'clientname': socket.gethostname()}
    FORMAT = '%(asctime)-15s %(clientname)s %(message)s'
    log_name = os.path.join(path_name, args.name) + '.log'
    logging.basicConfig(filename=log_name, level=logging.DEBUG, format=FORMAT, datefmt='%m/%d/%Y %I:%M:%S %p')
    for param in args:
        print(param)
    stats_train, stats_test = train.main(args)
    # stats_train, stats_test = evaluate.main(args)
    args.client = args.client['clientname']
    if logging.root:
        del logging.root.handlers[:]
    with open(folder_name + '/' + args.name, 'wb') as pickle_out:
        pickle.dump((stats_test, stats_train), pickle_out)
    with open(param_log_name, 'w') as log_file:
        log_file.write(json.dumps(vars(args)))
    return stats_test.get_scores()


if __name__ == '__main__':
    torch.manual_seed(214)
    torch.cuda.manual_seed_all(214)
    random.seed(214)
    np.random.seed(214)
    warnings.filterwarnings('ignore')
    # torch.backends.cudnn.enabled = False
    main()
