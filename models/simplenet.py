import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn0 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = conv3x3(16, 32, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv2 = conv3x3(32, 32)
        self.conv4 = conv3x3(32, 32)

        self.conv3 = conv3x3(32, 64, 2)
        self.bn5 = nn.BatchNorm2d(64)

        self.feats1 = nn.Sequential(self.conv1,
                                    self.bn1,
                                    self.relu,
                                    self.conv2,
                                    self.bn2)

        self.feats2 = nn.Sequential(self.conv4,
                                    self.bn3,
                                    self.relu,
                                    self.conv4,
                                    self.bn4)

        self.feats3 = nn.Sequential(self.conv3,
                                    self.bn5,
                                    self.relu)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.feats1(x)
        x = self.feats2(x)
        x = self.feats3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
