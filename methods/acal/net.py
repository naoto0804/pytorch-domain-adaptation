import torch.nn as nn
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self, n_class, n_ch):
        super(Classifier, self).__init__()
        self.res = 32
        self.conv1 = nn.Conv2d(n_ch, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc_input_len = (((self.res - 4) // 2 - 4) // 2) ** 2 * 50
        self.fc1 = nn.Linear(self.fc_input_len, 50)
        self.fc2 = nn.Linear(50, n_class)
        self.drop = nn.Dropout()

    def __call__(self, x):
        h = F.max_pool2d(F.relu(self.conv1(x)), 2, stride=2)
        h = F.max_pool2d(F.relu(self.conv2(h)), 2, stride=2)
        h = self.drop(F.relu(self.fc1(h.view(x.size(0), -1))))
        return self.fc2(h)
