import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init


# Referred this implementation
# https://github.com/NVlabs/MUNIT/blob/ec345132dfc7291f60a924ddd4259b97efabf2c9/utils.py#L229-L249


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)

    return init_fun


# This lenet is used in various previous baselines
# Domain-Adversarial Training of Neural Networks [JMLR2016]
# Domain separation networks [NIPS2016]
# Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks [CVPR2017]
class LenetClassifier(nn.Module):
    def __init__(self, n_class, n_ch, res):
        super(LenetClassifier, self).__init__()
        self.use_source_extractor = False
        self.conv1 = nn.Conv2d(n_ch, 32, 5)
        self.conv2 = nn.Conv2d(32, 48, 5)
        self.fc_input_len = (((res - 4) // 2 - 4) // 2) ** 2 * 48
        self.fc1 = nn.Linear(self.fc_input_len, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, n_class)

    def __call__(self, x):
        h = F.max_pool2d(F.relu(self.conv1(x)), 2, stride=2)
        h = F.max_pool2d(F.relu(self.conv2(h)), 2, stride=2)
        h = F.relu(self.fc1(h.view(-1, self.fc_input_len)))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class GenResBlock(nn.Module):
    def __init__(self, n_out_ch):
        super(GenResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_out_ch, n_out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(n_out_ch, n_out_ch, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(n_out_ch)
        self.bn2 = nn.BatchNorm2d(n_out_ch)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        return x + self.bn2(self.conv2(h))


class Generator(nn.Module):
    def __init__(self, n_hidden, n_resblock, n_ch, res, n_c_in, n_c_out):
        super(Generator, self).__init__()
        self.n_resblock = n_resblock
        self.n_hidden = n_hidden
        self.res = res
        self.fc = nn.Linear(n_hidden, self.res * self.res)
        self.conv1 = nn.Conv2d(1 + n_c_in, n_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_ch)

        for i in range(1, self.n_resblock + 1):
            setattr(self, 'block{:d}'.format(i), GenResBlock(n_ch))
        self.conv2 = nn.Conv2d(n_ch, n_c_out, 3, padding=1)

    def gen_noise(self, batchsize):
        return torch.randn(batchsize, self.n_hidden)  # z_{i} ~ N(0, 1)

    def __call__(self, x):
        z = self.gen_noise(x.size(0)).to(x.device)
        h = torch.cat(
            (x, F.relu(self.fc(z)).view(-1, 1, self.res, self.res)),
            dim=1)
        h = F.relu(self.bn1(self.conv1(h)))
        for i in range(1, self.n_resblock + 1):
            h = getattr(self, 'block{:d}'.format(i))(h)
        return torch.tanh(self.conv2(h))


class DisBlock(nn.Module):
    def __init__(self, n_in_ch, n_out_ch, slope=0.2, stride=2, padding=1):
        super(DisBlock, self).__init__()
        self.slope = slope
        self.conv = nn.Conv2d(n_in_ch, n_out_ch, 3,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(n_out_ch)

    def __call__(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), self.slope)


class Discriminator(nn.Module):
    def __init__(self, n_ch, res, n_c_in):
        super(Discriminator, self).__init__()
        self.slope = 0.2
        self.res = res
        self.len_block = int(np.log2(res)) - 3  # 32 -> 2, 28 -> 1

        self.conv1 = nn.Conv2d(n_c_in, n_ch, 3, stride=2, padding=1)
        for i in range(self.len_block):
            setattr(self, 'block{:d}'.format(i + 1),
                    DisBlock(n_ch * (2 ** i), n_ch * (2 ** (i + 1))))
        self.conv2 = nn.Conv2d(n_ch * (2 ** self.len_block), 1, 3, stride=1,
                               padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def __call__(self, x):
        # Don't normalize first conv-relu result!
        h = F.leaky_relu(self.conv1(x), self.slope)
        for i in range(self.len_block):
            h = getattr(self, 'block{:d}'.format(i + 1))(h)
        h = self.avg_pool(self.conv2(h))
        return h.view(-1, 1)
