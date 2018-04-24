import os
from itertools import chain

import click
import torch
import torch.cuda
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from datasets import DADataset
from datasets import load_source_target_datasets
from loss import GANLoss
from net import Classifier
from net import Discriminator
from net import Generator
from net import weights_init
from opt import exp_list
from opt import params
from preprocess import get_composed_transforms
from util.image_pool import ImagePool
from util.io import save_models_dict
from util.sampler import InfiniteSampler

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('--exp', type=click.Choice(exp_list), required=True)
@click.option('--affine', is_flag=True)
@click.option('--num_epochs', type=int, default=200)
def experiment(exp, affine, num_epochs):
    writer = SummaryWriter()
    log_dir = 'log/{:s}/sbada'.format(exp)
    os.makedirs(log_dir, exist_ok=True)

    alpha = params['weight']['alpha']
    beta = params['weight']['beta']
    gamma = params['weight']['gamma']
    mu = params['weight']['mu']
    new = params['weight']['new']
    eta = 0.0
    batch_size = params['batch_size']

    src, tgt = load_source_target_datasets(exp)

    n_ch_s = src.train_X.shape[1]  # number of color channels
    n_ch_t = tgt.train_X.shape[1]  # number of color channels
    res = src.train_X.shape[-1]  # size of image
    n_classes = src.n_classes

    train_tfs = get_composed_transforms(train=True, hflip=False)
    test_tfs = get_composed_transforms(train=False, hflip=False)

    src_train = DADataset(src.train_X, src.train_y, train_tfs, affine)
    tgt_train = DADataset(tgt.train_X, None, train_tfs, affine)
    tgt_test = DADataset(tgt.test_X, tgt.test_y, test_tfs, affine)
    del src, tgt

    n_sample = max(len(src_train), len(tgt_train))
    iter_per_epoch = n_sample // batch_size + 1

    weights_init_kaiming = weights_init('kaiming')
    weights_init_gaussian = weights_init('gaussian')

    cls_s = Classifier(n_classes, n_ch_s, res).cuda()
    cls_t = Classifier(n_classes, n_ch_t, res).cuda()

    cls_s.apply(weights_init_kaiming)
    cls_t.apply(weights_init_kaiming)

    gen_s_t_params = {'res': res, 'n_c_in': n_ch_s, 'n_c_out': n_ch_t}
    gen_t_s_params = {'res': res, 'n_c_in': n_ch_t, 'n_c_out': n_ch_s}
    gen_s_t = Generator(**{**params['gen_init'], **gen_s_t_params}).cuda()
    gen_t_s = Generator(**{**params['gen_init'], **gen_t_s_params}).cuda()
    gen_s_t.apply(weights_init_gaussian)
    gen_t_s.apply(weights_init_gaussian)

    dis_s_params = {'res': res, 'n_c_in': n_ch_s}
    dis_t_params = {'res': res, 'n_c_in': n_ch_t}
    dis_s = Discriminator(**{**params['dis_init'], **dis_s_params}).cuda()
    dis_t = Discriminator(**{**params['dis_init'], **dis_t_params}).cuda()
    dis_s.apply(weights_init_gaussian)
    dis_t.apply(weights_init_gaussian)

    config = {'lr': params['base_lr'], 'weight_decay': params['weight_decay'],
              'betas': params['betas']}
    opt_cls = Adam(chain(cls_s.parameters(), cls_t.parameters()), **config)
    opt_gen = Adam(chain(gen_s_t.parameters(), gen_t_s.parameters()), **config)
    opt_dis = Adam(chain(dis_s.parameters(), dis_t.parameters()), **config)

    calc_ls = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
    calc_ce = F.cross_entropy

    fake_src_x_pool = ImagePool(params['pool_size'] * batch_size)
    fake_tgt_x_pool = ImagePool(params['pool_size'] * batch_size)

    src_train_iter = iter(DataLoader(
        src_train, batch_size=batch_size, num_workers=4,
        sampler=InfiniteSampler(len(src_train))))
    tgt_train_iter = iter(DataLoader(
        tgt_train, batch_size=batch_size, num_workers=4,
        sampler=InfiniteSampler(len(tgt_train))))
    tgt_test_loader = DataLoader(
        tgt_test, batch_size=batch_size * 4, num_workers=4)
    print('Training...')

    niter = 0
    while True:
        niter += 1
        src_x, src_y = next(src_train_iter)
        tgt_x = next(tgt_train_iter)
        src_x = Variable(src_x.cuda())
        src_y = Variable(src_y.cuda())
        tgt_x = Variable(tgt_x.cuda())

        if niter >= num_epochs * 0.75 * iter_per_epoch:
            eta = params['weight']['eta']

        fake_tgt_x = gen_s_t(src_x)
        fake_back_src_x = gen_t_s(fake_tgt_x)
        fake_src_x = gen_t_s(tgt_x)

        fake_src_pseudo_y = Variable(
            torch.max(cls_s(fake_src_x).data, dim=1)[1])

        # eq2
        loss_gen = beta * calc_ce(cls_t(fake_tgt_x), src_y)

        # eq3
        loss_gen += gamma * calc_ls(dis_s(fake_src_x), True)
        loss_gen += alpha * calc_ls(dis_t(fake_tgt_x), True)

        # eq5
        loss_gen += eta * calc_ce(cls_s(fake_src_x), fake_src_pseudo_y)

        # eq6
        loss_gen += new * calc_ce(cls_s(fake_back_src_x), src_y)

        # do not backpropagate loss to generator
        fake_tgt_x = fake_tgt_x.detach()
        fake_src_x = fake_src_x.detach()
        fake_back_src_x = fake_back_src_x.detach()

        # eq2
        loss_cls_s = mu * calc_ce(cls_s(src_x), src_y)  # no feedback
        loss_cls_t = beta * calc_ce(cls_t(fake_tgt_x), src_y)

        # eq3
        loss_dis_s = gamma * calc_ls(
            dis_s(fake_src_x_pool.query(fake_src_x.data)), False)
        loss_dis_s += gamma * calc_ls(dis_s(src_x), True)
        loss_dis_t = alpha * calc_ls(
            dis_t(fake_tgt_x_pool.query(fake_tgt_x.data)), False)
        loss_dis_t += alpha * calc_ls(dis_t(tgt_x), True)

        # eq5
        loss_cls_s += eta * calc_ce(cls_s(fake_src_x), fake_src_pseudo_y)

        # eq6
        loss_cls_s += new * calc_ce(cls_s(fake_back_src_x), src_y)

        loss_cls = loss_cls_s + loss_cls_t
        loss_dis = loss_dis_s + loss_dis_t

        for opt, loss in zip([opt_dis, opt_cls, opt_gen],
                             [loss_dis, loss_cls, loss_gen]):
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

        if niter % 100 == 0 and niter > 0:
            writer.add_scalar('dis/src', loss_dis_s.data.cpu()[0], niter)
            writer.add_scalar('dis/tgt', loss_dis_t.data.cpu()[0], niter)
            writer.add_scalar('cls/src', loss_cls_s.data.cpu()[0], niter)
            writer.add_scalar('cls/tgt', loss_cls_t.data.cpu()[0], niter)
            writer.add_scalar('gen', loss_gen.data.cpu()[0], niter)

        if niter % iter_per_epoch == 0:
            epoch = niter // iter_per_epoch
            cls_s.eval()
            cls_t.eval()

            n_err = 0
            for batch_idx, (x, y) in enumerate(tgt_test_loader):
                x = Variable(x.cuda(), requires_grad=False)
                prob_y = F.softmax(cls_t(x), dim=1).data.cpu()
                pred_y = torch.max(prob_y, dim=1)[1]
                n_err += (pred_y != y).sum()

            writer.add_scalar('err_tgt', n_err / len(tgt_test), epoch)

            cls_s.train()
            cls_t.train()

            if epoch % 10 == 0:
                data = []
                for x in [src_x, fake_tgt_x, fake_back_src_x, tgt_x,
                          fake_src_x]:
                    x = x.data.cpu()
                    if x.size(1) == 1:
                        x = x.repeat(1, 3, 1, 1)  # grayscale2rgb
                    data.append(x)
                grid = make_grid(torch.cat(tuple(data), dim=0),
                                 normalize=True, range=(-1.0, 1.0))
                writer.add_image('generated', grid, epoch)

            if epoch % 50 == 0:
                models_dict = {
                    'cls_s': cls_s, 'cls_t': cls_t, 'dis_s': dis_s,
                    'dis_t': dis_t, 'gen_s_t': gen_s_t, 'gen_t_s': gen_t_s}
                filename = '{:s}/epoch{:d}.tar'.format(log_dir, epoch)
                save_models_dict(models_dict, filename)

            if epoch >= num_epochs:
                break


if __name__ == '__main__':
    experiment()
