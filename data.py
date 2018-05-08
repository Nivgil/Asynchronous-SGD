import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from PIL import ImageFile


def load_data(args):
    if args.dataset == 'imagenet':
        print('Loading ImageNet - ', end='')
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

        root = '/home/ehoffer/Datasets/imagenet/'

        '''if batch size is too big to fit in gpu (resnet 50 case) split the batch 256 chunks'''
        if args.batch_size > 256 and args.model == 'resnet50':
            batch_size = 256
        else:
            batch_size = args.batch_size

        trainset = datasets.ImageFolder(root=root + 'train', transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True, num_workers=16, pin_memory=True)

        testset = datasets.ImageFolder(root=root + 'val', transform=transform_test)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=1024,
                                                 shuffle=False, num_workers=16, pin_memory=True)
    else:
        print('Loading CIFAR' + str(args.dataset == 'cifar10' and 10 or 100) + ' - ', end='')
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        if args.augment:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False, volatile=True),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        kwargs = {'num_workers': 8, 'pin_memory': True}
        assert (args.dataset == 'cifar10' or args.dataset == 'cifar100')
        train_loader = torch.utils.data.DataLoader(
            datasets.__dict__[args.dataset.upper()]('../data', train=True, download=True,
                                                    transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.__dict__[args.dataset.upper()]('../data', train=False, transform=transform_test),
            batch_size=1024, shuffle=False, **kwargs)

    return train_loader, val_loader
