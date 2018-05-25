import argparse


def configuration():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='dataset (cifar10 [default] or cifar100)')
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--layers', default=28, type=int,
                        help='total number of layers (default: 28)')
    parser.add_argument('--widen-factor', default=2, type=int,
                        help='widen factor (default: 2)')
    parser.add_argument('--droprate', default=0, type=float,
                        help='dropout probability (default: 0.0)')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='whether to use standard augmentation (default: True)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--name', default='WideResNet-28-2', type=str,
                        help='name of experiment')
    parser.add_argument('--sim_num', default='1', type=int,
                        help='simulation id')
    parser.add_argument('--workers_num', default=1, type=int,
                        help='number of workers')
    parser.add_argument('--grad_clip', default=1000, type=float,
                        help='gradient clipping threshold')
    parser.add_argument('--no_pbar', dest='bar', action='store_false',
                        help='show progress bar (default: False)')
    parser.add_argument('--time_p', dest='p_time', action='store_true',
                        help='show iteration time analysis')
    parser.add_argument('--no_regime', dest='regime', action='store_false',
                        help='train without regime adaptation')
    parser.add_argument('--fast_im', dest='fast_im', action='store_true',
                        help='1 hour image net training regime')
    parser.add_argument('--gbn', dest='gbn', action='store_true',
                        help='ghost batch normalization')
    parser.add_argument('--id', default=2000, type=int,
                        help='simulation number')
    parser.add_argument('--save', default=100, type=int,
                        help='save simulation state every X epochs')
    parser.add_argument('--optimizer', default='asynchronous', type=str,
                        help='type of optimizer (synchronous/asynchronous/elastic)')
    parser.add_argument('--tau', default=1, type=int,
                        help='tau communication for elastic')
    parser.add_argument('--rho', default=2.5, type=int,
                        help='rho value for elastic')
    parser.add_argument('--baseline', default=128, type=int,
                        help='batch size baseline')
    parser.add_argument('--model', default='resnet', type=str,
                        help='chosen architecture')
    parser.add_argument('--notes', default='', type=str,
                        help='notes for simulation')
    parser.set_defaults(augment=True)
    args = parser.parse_args()
    if args.model == 'wideresnet':
        args.wd = 5e-4
    if args.dataset == 'imagenet':
        if args.model == 'alexnet':
            args.lr = 0.01
            args.save = 5
        args.save = 1
        args.baseline = 256
    if args.fast_im is True:
        args.regime = False
        args.nesterov = True
    args.name = args.model
    for arg in vars(args):
        print(arg, getattr(args, arg))
    return args
