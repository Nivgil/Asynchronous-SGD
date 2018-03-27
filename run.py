import logging
from multiprocessing import Process
import pickle
import json
import os
import numpy as np
import torch
import random
import train
from configuration import configuration


def main():
    args = configuration()
    # workers = (3, )
    # for idx, worker_num in enumerate(workers):
    #     args.workers_num = worker_num
    #     args.sim_num = idx
    exec_unit(args)
        # p = Process(target=exec_unit, args=(args, simulation_number,))
        # p.start()
        # p.join()


def exec_unit(args=None):
    if args is None:
        args = configuration()

    CONFIGURATIONS_DIR = os.path.join(os.path.dirname(__file__))
    folder_name = 'simulation_{}'.format(args.id)
    path_name = os.path.join(CONFIGURATIONS_DIR, folder_name)
    if not os.path.exists(path_name):
        os.mkdir(path_name)
    log_name = os.path.join(path_name, args.name + '_' + str(args.sim_num) + '.log')
    if os.path.exists(os.path.join(path_name, log_name)):
        raise Exception('Log file already exists')

    stats_train, stats_test = train.main(args)

    with open(folder_name + '/' + args.name + '_' + str(args.sim_num), 'wb') as pickle_out:
        pickle.dump((stats_test, stats_train), pickle_out)
    logging.basicConfig(format='%(message)s', filename=log_name, level=logging.DEBUG)
    logging.info(json.dumps(vars(args)))


if __name__ == '__main__':
    torch.manual_seed(214)
    torch.cuda.manual_seed_all(214)
    random.seed(214)
    np.random.seed(214)
    # torch.backends.cudnn.enabled = False
    main()
