import click
import os
import torch
import torch.cuda
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import DADataset
from datasets import load_source_target_datasets
from util import save_model
from util.io import get_config
from util.net import LenetClassifier
from util.net import weights_init
from util.opt import exp_list
from util.preprocess import get_composed_transforms

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('--exp', type=click.Choice(exp_list), required=True)
@click.option('--use_affine', is_flag=True)
@click.option('--num_epochs', type=int, default=200)
@click.option('--train_type', type=click.Choice(['sup', 'unsup']))
def experiment(exp, use_affine, num_epochs, train_type):
    config = get_config('config.yaml')
    lr = float(config['lr'])
    weight_decay = float(config['weight_decay'])
    batch_size = int(config['batch_size'])

    log_dir = 'log/{:s}/{:s}'.format(exp, train_type)
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device('cuda')

    src, tgt = load_source_target_datasets(exp)
    train = tgt if train_type == 'sup' else src
    test = tgt

    n_ch = train.train_X.shape[1]  # number of color channels
    res = train.train_X.shape[-1]  # size of image
    n_classes = train.n_classes

    cls = LenetClassifier(n_classes, n_ch, res).to(device)
    cls.apply(weights_init('kaiming'))

    optimizer = Adam(list(cls.parameters()), lr=lr, weight_decay=weight_decay)

    train_tfs = get_composed_transforms(train=True)
    test_tfs = get_composed_transforms(train=False)
    train_data = DADataset(train.train_X, train.train_y, train_tfs, use_affine)
    test_data = DADataset(test.test_X, test.test_y, test_tfs, use_affine)

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              num_workers=4, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4)

    print('Training...')
    for epoch in range(1, num_epochs + 1):

        cls.train()
        for tgt_x, tgt_y in train_loader:
            loss = F.cross_entropy(cls(tgt_x.to(device)), tgt_y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0 and epoch > 0:
            cls.eval()
            n_err = 0
            with torch.no_grad():
                for tgt_x, tgt_y in test_loader:
                    prob_y = F.softmax(cls(tgt_x.to(device)), dim=1)
                    pred_y = torch.max(prob_y, dim=1)[1]
                    pred_y = pred_y.to(torch.device('cpu'))
                    n_err += (pred_y != tgt_y).sum().item()
            print('Epoch {:d}, Err {:f}'.format(epoch, n_err / len(test_data)))

        if epoch % 100 == 0 and epoch > 0:
            save_model(cls, '{:s}/epoch{:d}.tar'.format(log_dir, epoch))


if __name__ == '__main__':
    experiment()
