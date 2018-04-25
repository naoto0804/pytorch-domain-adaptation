import os

import click
import torch
import torch.cuda
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import DADataset
from datasets import load_source_target_datasets
from net import LenetClassifier
from net import weights_init
from opt import exp_list
from opt import params
from preprocess import get_composed_transforms
from util import save_model

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('--exp', type=click.Choice(exp_list), required=True)
@click.option('--use_affine', is_flag=True)
@click.option('--num_epochs', type=int, default=200)
def experiment(exp, use_affine, num_epochs):
    log_dir = 'log/{:s}/unsup'.format(exp)
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device('cuda')

    src, tgt = load_source_target_datasets(exp)
    if src.train_X.shape[1] != tgt.train_X.shape[1]:
        raise RuntimeError('channel mismatch for source / target dataset')

    n_ch_t = tgt.train_X.shape[1]  # number of color channels
    res = src.train_X.shape[-1]  # size of image
    n_classes = src.n_classes

    cls = LenetClassifier(n_classes, n_ch_t, res).to(device)
    cls.apply(weights_init('kaiming'))

    config = {'lr': params['base_lr'], 'weight_decay': params['weight_decay']}
    optimizer = Adam(list(cls.parameters()), **config)

    train_tfs = get_composed_transforms(train=True, hflip=False)
    test_tfs = get_composed_transforms(train=False, hflip=False)
    src_train = DADataset(src.train_X, src.train_y, train_tfs, use_affine)
    tgt_test = DADataset(tgt.test_X, tgt.test_y, test_tfs, use_affine)

    src_train_loader = DataLoader(src_train, batch_size=params['batch_size'],
                                  num_workers=4, shuffle=True)
    tgt_test_loader = DataLoader(tgt_test, batch_size=params['batch_size'],
                                 num_workers=4)

    print('Training...')
    for epoch in range(1, num_epochs + 1):

        cls.train()
        for tgt_x, tgt_y in src_train_loader:
            loss = F.cross_entropy(cls(tgt_x.to(device)), tgt_y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0 and epoch > 0:
            cls.eval()
            n_err = 0
            with torch.no_grad():
                for tgt_x, tgt_y in tgt_test_loader:
                    prob_y = F.softmax(cls(tgt_x.to(device)), dim=1)
                    pred_y = torch.max(prob_y, dim=1)[1]
                    pred_y = pred_y.to(torch.device('cpu'))
                    n_err += (pred_y != tgt_y).sum().item()
            print('Epoch {:d}, Err {:f}'.format(epoch, n_err / len(tgt_test)))

        if epoch % 100 == 0 and epoch > 0:
            save_model(cls, '{:s}/epoch{:d}.tar'.format(log_dir, epoch))


if __name__ == '__main__':
    experiment()
