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
from datetime import datetime


def main():
    torch.distributed.init_process_group(backend='gloo', init_method='tcp://132.68.43.137:29500')
    args = configuration()
    base_name = args.name
    base_id = args.id
    seed_vals = [8678576, 4527389, 2113183, 518078, 7370063]
    for idx in range(1, 2):
        args.id = base_id + idx - 1
        # seed_val = random.randrange(10000000)
        seed_val = seed_vals[idx - 1]
        seed_system(seed_val)
        args.seed = seed_val
        args.name = base_name + '_{}'.format(args.id)
        time = str(datetime.now())
        scores = exec_unit(args)
        graph_path = create_graphs(sim_num=args.id, linear=True)

        message = '{0}\nSimulation Number {1} Completed\n' \
                  'Min Error - {2:.3f}\n' \
                  'Mean Error - {3:.3f}\n' \
                  'Val Error - {4:.3f}'.format(time, args.id, scores[0], scores[1], scores[2])
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
    logging.basicConfig(filename=log_name, level=logging.INFO, format=FORMAT, datefmt='%m/%d/%Y %I:%M:%S %p')
    configuration_str = ''
    for arg in vars(args):
        configuration_str = configuration_str + arg + ' ' + str(getattr(args, arg)) + '\n'
    print(configuration_str)
    logging.info(configuration_str, extra=args.client)
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


def seed_system(seed_val=None):
    if seed_val is None:  # generate seed
        seed_val = random.randrange(10000000)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)


if __name__ == '__main__':
    seed_system(214)
    warnings.filterwarnings('ignore')
    # torch.backends.cudnn.enabled = False
    main()
