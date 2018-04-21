import os

import click
import torch
import torch.cuda
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import DADataset
from datasets import load_source_target_datasets
from net import Classifier
from net import weights_init
from opt import exp_list
from opt import params
from preprocess import get_composed_transforms
from util.io import save_model

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('--exp', type=click.Choice(exp_list), required=True)
def experiment(exp):
    num_epochs = 500
    log_dir = 'log/{:s}/sup'.format(exp)
    os.makedirs(log_dir, exist_ok=True)

    src, tgt = load_source_target_datasets(exp)

    n_ch_t = tgt.train_X.shape[1]  # number of color channels
    res = src.train_X.shape[-1]  # size of image
    n_classes = src.n_classes

    cls = Classifier(n_classes, n_ch_t, res).cuda()
    cls.apply(weights_init('kaiming'))

    config = {'lr': params['base_lr'], 'weight_decay': params['weight_decay']}
    optimizer = Adam(list(cls.parameters()), **config)

    train_tfs = get_composed_transforms(train=True, hflip=False)
    test_tfs = get_composed_transforms(train=False, hflip=False)
    tgt_train = DADataset(tgt.train_X, tgt.train_y, train_tfs, True)
    tgt_test = DADataset(tgt.test_X, tgt.test_y, test_tfs, False)

    tgt_train_loader = DataLoader(tgt_train, batch_size=params['batch_size'],
                                  num_workers=4, shuffle=True)
    tgt_test_loader = DataLoader(tgt_test, batch_size=params['batch_size'],
                                 num_workers=4)

    print('Training...')
    for epoch in range(1, num_epochs + 1):

        cls.train()
        for tgt_X, tgt_y in tgt_train_loader:
            tgt_X = Variable(tgt_X.cuda())
            tgt_y = Variable(tgt_y.cuda())
            loss = F.cross_entropy(cls(tgt_X), tgt_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0 and epoch > 0:
            cls.eval()
            n_err = 0
            for tgt_X, tgt_y in tgt_test_loader:
                tgt_X = Variable(tgt_X.cuda(), requires_grad=False)
                prob_y = F.softmax(cls(tgt_X), dim=1).data.cpu()
                pred_y = torch.max(prob_y, dim=1)[1]
                n_err += (pred_y != tgt_y).sum()
            print('Epoch {:d}, Err {:f}'.format(epoch, n_err / len(tgt_test)))

        if epoch % 100 == 0 and epoch > 0:
            save_model(cls, '{:s}/epoch{:d}.tar'.format(log_dir, epoch))


if __name__ == '__main__':
    experiment()
