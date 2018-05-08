import torch
from torch.nn import Module, BatchNorm2d


class GhostBatchNorm(Module):

    def __init__(self, num_features, chunk_size=128):
        super(GhostBatchNorm, self).__init__()
        self.bn = BatchNorm2d(num_features)
        self.num_features = num_features
        self.chunk_size = chunk_size
        self.weight = self.bn.weight
        self.bias = self.bn.bias

    def forward(self, input):
        input_bn = list()
        input_chunks = torch.split(input, self.chunk_size)
        for x in input_chunks:
            input_bn.append(self.bn(x))
        x = torch.cat(input_bn)
        return x