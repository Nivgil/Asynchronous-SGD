import pickle
import json
import os
import numpy as np
import torch
import random
import train
from configuration import configuration
from email_notification import send_notification
import warnings


def main():
    args = configuration()
    base_name = args.name

    for idx in range(1, 4):
        # args.id = 1000
        args.sim_num = idx * 10
        # args.batch_size = 128
        # args.workers_num = 1
        # args.epochs = 200
        args.name = base_name + '_{}'.format(args.sim_num)
        # args.gbn = 0
        # args.notes = 'baseline'
        min_error, mean_error = exec_unit(args)
        message = 'Simulation Number {0} Completed\nMin Error - {1:.3f}\nMean Error - {2:.3f}'.format(args.sim_num,
                                                                                                      min_error,
                                                                                                      mean_error)
        send_notification(message, vars(args))


def exec_unit(args=None):
    if args is None:
        args = configuration()

    CONFIGURATIONS_DIR = os.path.join(os.path.dirname(__file__))
    folder_name = 'simulation_{}'.format(args.id)
    path_name = os.path.join(CONFIGURATIONS_DIR, folder_name)
    if not os.path.exists(path_name):
        os.mkdir(path_name)
    log_name = os.path.join(path_name, args.name + '.log')
    if os.path.exists(log_name):
        raise Exception('Log file already exists')

    stats_train, stats_test = train.main(args)

    with open(folder_name + '/' + args.name, 'wb') as pickle_out:
        pickle.dump((stats_test, stats_train), pickle_out)
    with open(log_name, 'w') as log_file:
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
