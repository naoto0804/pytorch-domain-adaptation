import os
from itertools import chain

import click
import numpy as np
import torch
import torch.cuda
from batchup import data_source
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import Adam
from torchvision.utils import make_grid

from loader import load_source_target_datasets
from loss import GANLoss
from net import Generator, Discriminator, weights_init_normal, \
    Classifier, weights_init_kaiming
from opt import params, exp_list
from util import ImagePool, norm_cls_to_gan, save_model, load_model

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('--exp', type=click.Choice(exp_list), required=True)
@click.option('--seed', type=int, default=0,
              help='random seed (0 for time-based)')
@click.option('--modelname', type=str, default='')
def experiment(exp, seed, modelname):
    writer = SummaryWriter()
    log_dir = 'log/{:s}/sbada'.format(exp)
    os.makedirs(log_dir, exist_ok=True)

    data_src, data_tgt = load_source_target_datasets(exp)

    # Delete the training ground truths as we should not be using them
    del data_tgt.train_y

    n_sample = max(data_src.train_X.shape[0], data_tgt.train_X.shape[0])
    iter_per_epoch = n_sample // params['batch_size'] + 1

    n_color = data_src.train_X.shape[1]
    res = data_src.train_X.shape[-1]

    cls_s = Classifier(data_src.n_classes, n_color, res).cuda()
    cls_t = Classifier(data_src.n_classes, n_color, res).cuda()

    if modelname:
        load_model(cls_s, modelname)
        load_model(cls_t, modelname)
    else:
        cls_s.apply(weights_init_kaiming)
        cls_t.apply(weights_init_kaiming)

    gen_params_dict = params['gen_init']
    gen_params_dict.update({'res': res, 'n_color': n_color})
    gen_s_t = Generator(**gen_params_dict).cuda()
    gen_t_s = Generator(**gen_params_dict).cuda()
    gen_s_t.apply(weights_init_normal)
    gen_t_s.apply(weights_init_normal)

    dis_params_dict = params['dis_init']
    dis_params_dict.update({'res': res, 'n_color': n_color})
    dis_s = Discriminator(**dis_params_dict).cuda()
    dis_t = Discriminator(**dis_params_dict).cuda()
    dis_s.apply(weights_init_normal)
    dis_t.apply(weights_init_normal)

    config = {'lr': params['base_lr'], 'weight_decay': params['weight_decay']}
    opt_cls = Adam(chain(cls_s.parameters(), cls_t.parameters()), **config)
    opt_gen = Adam(chain(gen_t_s.parameters(), gen_s_t.parameters()), **config)
    opt_dis = Adam(chain(dis_s.parameters(), dis_t.parameters()), **config)

    calc_ls = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
    calc_ce = F.cross_entropy

    if seed != 0:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    data_src.train_X = norm_cls_to_gan(data_src.train_X)
    data_tgt.train_X = norm_cls_to_gan(data_tgt.train_X)
    data_tgt.test_X = norm_cls_to_gan(data_tgt.test_X)
    source_train_ds = data_source.ArrayDataSource(
        [data_src.train_X, data_src.train_y], repeats=-1)
    target_train_ds = data_source.ArrayDataSource(
        [data_tgt.train_X], repeats=-1)
    target_test_ds = data_source.ArrayDataSource(
        [data_tgt.test_X, data_tgt.test_y])
    ds = data_source.CompositeDataSource(
        [source_train_ds, target_train_ds])

    alpha = params['weight']['alpha']
    beta = params['weight']['beta']
    gamma = params['weight']['gamma']
    mu = params['weight']['mu']
    new = params['weight']['new']
    eta = 0.0

    fake_src_X_pool = ImagePool(params['pool_size'] * params['batch_size'])
    fake_tgt_X_pool = ImagePool(params['pool_size'] * params['batch_size'])

    print('Training...')

    niter = 0
    for (src_X, src_y, tgt_X) in \
            ds.batch_iterator(batch_size=params['batch_size'],
                              shuffle=rng):
        niter += 1
        if niter == params['num_epochs'] // 2 * iter_per_epoch:
            eta = params['weight']['eta']

        src_X = Variable(torch.from_numpy(src_X).cuda())
        src_y = Variable(torch.from_numpy(src_y).long().cuda())
        tgt_X = Variable(torch.from_numpy(tgt_X).cuda())

        fake_tgt_X = gen_s_t(src_X)
        fake_back_src_X = gen_t_s(fake_tgt_X)
        fake_src_X = gen_t_s(tgt_X)

        fake_src_pseudo_y = Variable(
            torch.max(cls_s(fake_src_X).data, dim=1)[1])

        # eq2
        loss_gen = beta * calc_ce(cls_t(fake_tgt_X), src_y)

        # eq3
        loss_gen += gamma * calc_ls(dis_s(fake_src_X), True)
        loss_gen += alpha * calc_ls(dis_t(fake_tgt_X), True)

        # eq5
        loss_gen += eta * calc_ce(cls_s(fake_src_X), fake_src_pseudo_y)

        # eq6
        loss_gen += new * calc_ce(cls_s(fake_back_src_X), src_y)

        # do not backpropagate loss to generator
        fake_tgt_X = fake_tgt_X.detach()
        fake_src_X = fake_src_X.detach()
        fake_back_src_X = fake_back_src_X.detach()

        # eq2
        loss_cls_s = mu * calc_ce(cls_s(src_X), src_y)  # no feedback
        loss_cls_t = beta * calc_ce(cls_t(fake_tgt_X), src_y)

        # eq3
        loss_dis_s = gamma * calc_ls(
            dis_s(fake_src_X_pool.query(fake_src_X.data)), False)
        loss_dis_s += gamma * calc_ls(dis_s(src_X), True)
        loss_dis_t = alpha * calc_ls(
            dis_t(fake_tgt_X_pool.query(fake_tgt_X.data)), False)
        loss_dis_t += alpha * calc_ls(dis_t(tgt_X), True)

        # eq5
        loss_cls_s += eta * calc_ce(cls_s(fake_src_X), fake_src_pseudo_y)

        # eq6
        loss_cls_s += new * calc_ce(cls_s(fake_back_src_X), src_y)

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

        if niter % iter_per_epoch == 0:
            def f_eval(X_sup, y_sup):
                X_var = Variable(torch.from_numpy(X_sup).cuda(),
                                 requires_grad=False)
                y_t_prob = F.softmax(cls_t(X_var), dim=1).data.cpu().numpy()
                # TODO: mix cls_t(X_var) and cls_s(gen_t_s(X_var))
                y_pred = np.argmax(y_t_prob, axis=1)
                return float((y_pred != y_sup).sum())

            cls_s.eval()
            cls_t.eval()

            epoch = niter // iter_per_epoch
            tgt_test_err, = target_test_ds.batch_map_mean(
                f_eval, batch_size=params['batch_size'] * 4)
            writer.add_scalar('err_tgt', tgt_test_err, epoch)

            cls_s.train()
            cls_t.train()

            if epoch % 5 == 0:
                tpl = tuple(
                    x.data.cpu() for x in [src_X, fake_tgt_X, fake_back_src_X])
                grid = make_grid(torch.cat(tpl, dim=0),
                                 normalize=True, range=(-1.0, 1.0))
                writer.add_image('generated', grid, epoch)
                save_model(gen_s_t,
                           '{:s}/epoch{:d}.tar'.format(log_dir, epoch))


if __name__ == '__main__':
    experiment()
