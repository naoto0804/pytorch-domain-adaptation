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
from net import Generator
from net import weights_init
from opt import exp_list
from opt import params
from preprocess import get_composed_transforms
from util.io import load_models_dict
from util.io import save_model
from util.sampler import InfiniteSampler

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('--exp', type=click.Choice(exp_list), required=True)
@click.option('--modelname', type=str, required=True)
def experiment(exp, modelname):
    num_epochs = 500
    log_dir = 'log/{:s}/generator'.format(exp)
    os.makedirs(log_dir, exist_ok=True)

    batch_size = params['batch_size']

    src, tgt = load_source_target_datasets(exp)

    n_ch_s = src.train_X.shape[1]  # number of color channels
    n_ch_t = tgt.train_X.shape[1]  # number of color channels
    res = src.train_X.shape[-1]  # size of image
    n_classes = src.n_classes

    train_tfs = get_composed_transforms(train=True, hflip=False)
    test_tfs = get_composed_transforms(train=False, hflip=False)

    src_train = DADataset(src.train_X, src.train_y, train_tfs, True)
    tgt_test = DADataset(tgt.test_X, tgt.test_y, test_tfs, False)
    del src, tgt

    n_sample = len(src_train)
    iter_per_epoch = n_sample // batch_size + 1

    cls_t = Classifier(n_classes, n_ch_t, res).cuda()
    cls_t.apply(weights_init('kaiming'))

    gen_s_t_params = {'res': res, 'n_c_in': n_ch_s, 'n_c_out': n_ch_t}
    gen_s_t = Generator(**{**params['gen_init'], **gen_s_t_params}).cuda()
    load_models_dict({'gen_s_t': gen_s_t}, modelname)

    config = {'lr': params['base_lr'], 'weight_decay': params['weight_decay'],
              'betas': params['betas']}
    opt_cls = Adam(list(cls_t.parameters()), **config)

    src_train_iter = iter(DataLoader(
        src_train, batch_size=batch_size, num_workers=4,
        sampler=InfiniteSampler(len(src_train))))
    tgt_test_loader = DataLoader(
        tgt_test, batch_size=batch_size * 4, num_workers=4)
    print('Training...')

    niter = 0
    while True:
        niter += 1
        src_x, src_y = next(src_train_iter)
        src_x = Variable(src_x.cuda(), requires_grad=False)
        src_y = Variable(src_y.cuda())

        fake_tgt_x = Variable(gen_s_t(src_x).data)
        loss = F.cross_entropy(cls_t(fake_tgt_x), src_y)

        opt_cls.zero_grad()
        loss.backward()
        opt_cls.step()

        if niter % iter_per_epoch == 0:
            n_err = 0
            epoch = niter // iter_per_epoch

            cls_t.eval()
            for batch_idx, (x, y) in enumerate(tgt_test_loader):
                x = Variable(x.cuda(), requires_grad=False)
                prob_y = F.softmax(cls_t(x), dim=1).data.cpu()
                pred_y = torch.max(prob_y, dim=1)[1]
                n_err += (pred_y != y).sum()
            cls_t.train()
            print('Epoch {:d}, Err {:f}'.format(epoch, n_err / len(tgt_test)))

            if epoch % 100 == 0:
                save_model(cls_t, '{:s}/epoch{:d}.tar'.format(log_dir, epoch))

            if epoch >= num_epochs:
                break


if __name__ == '__main__':
    experiment()
