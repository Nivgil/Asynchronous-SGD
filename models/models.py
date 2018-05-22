from models.resnet import resnet
from models.densenet import densenet
from models.alexnet import alexnet
from models.wideresnet import WideResNet
from models.simplenet import SimpleNet


def get_model(model, args):
    if model == 'alexnet':
        return alexnet()
    if model == 'resnet':
        return resnet(dataset=args.dataset)
    if model == 'wideresnet':
        return WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                          args.widen_factor, dropRate=args.droprate, gbn=args.gbn)
    if model == 'simplenet':
        return SimpleNet()
    if model == 'densenet':
        return densenet()
