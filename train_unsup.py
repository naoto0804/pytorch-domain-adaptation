import os

import click
import numpy as np
import torch
import torch.cuda
from batchup import data_source
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import Adam

from datasets import load_source_target_datasets
from net import Classifier, weights_init_kaiming
from opt import params, exp_list
from util.io import save_model
from util.normalize import norm_cls_to_gan

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('--exp', type=click.Choice(exp_list), required=True)
@click.option('--seed', type=int, default=0,
              help='random seed (0 for time-based)')
def experiment(exp, seed):
    log_dir = 'log/{:s}/unsup'.format(exp)
    os.makedirs(log_dir, exist_ok=True)

    data_src, data_tgt = load_source_target_datasets(exp)
    del data_tgt.train_y
    del data_tgt.train_X

    n_color = data_src.train_X.shape[1]
    res = data_src.train_X.shape[-1]

    cls = Classifier(data_tgt.n_classes, n_color, res).cuda()
    cls.apply(weights_init_kaiming)

    config = {'lr': params['base_lr'], 'weight_decay': params['weight_decay']}
    optimizer = Adam(list(cls.parameters()), **config)

    if seed != 0:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    data_src.train_X = norm_cls_to_gan(data_src.train_X)
    data_tgt.test_X = norm_cls_to_gan(data_tgt.test_X)

    source_train_ds = data_source.ArrayDataSource(
        [data_src.train_X, data_src.train_y])
    target_test_ds = data_source.ArrayDataSource(
        [data_tgt.test_X, data_tgt.test_y])
    niter = 0

    print('Training...')
    for epoch in range(params['num_epochs']):

        cls.train()

        for (src_X, src_y) in source_train_ds.batch_iterator(
                batch_size=params['batch_size'], shuffle=rng):
            niter += 1
            src_X = Variable(torch.from_numpy(src_X).cuda())
            src_y = Variable(torch.from_numpy(src_y).long().cuda())

            loss = F.cross_entropy(cls(src_X), src_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cls.eval()

        def f_eval(X_sup, y_sup):
            X_var = Variable(torch.from_numpy(X_sup).cuda(),
                             requires_grad=False)
            y_t_prob = F.softmax(cls(X_var), dim=1).data.cpu().numpy()
            y_pred = np.argmax(y_t_prob, axis=1)
            return float((y_pred != y_sup).sum())

        tgt_test_err, = target_test_ds.batch_map_mean(
            f_eval, batch_size=params['batch_size'] * 4)

        fmt = '*** Epoch {} TGT TEST err={:.3%}'
        print(fmt.format(epoch, tgt_test_err))

        if (epoch + 1) % 100 == 0 and epoch > 0:
            save_model(cls, '{:s}/epoch{:d}.tar'.format(log_dir, epoch + 1))


if __name__ == '__main__':
    experiment()
