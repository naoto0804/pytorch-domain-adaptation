import sys
from itertools import chain

import click
import os
import shutil
import torch
import torch.cuda
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

sys.path.append('../..')

from util.opt import exp_list
from methods.acal.cyclegan.networks import define_D, define_G
from methods.acal.net import Classifier
from util.dataset import DADataset
from util.dataset import SubsetDataset
from util.dataset import load_source_target_datasets
from util.evaluate import evaluate_classifier
from util.image_pool import ImagePool
from util.io import load_model, save_model, save_models_dict, get_config
from util.loss import GANLoss
from util.sampler import InfiniteSampler
from util.transform import get_composed_transforms

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('--exp', type=click.Choice(exp_list), required=True)
@click.option('--num_epochs', type=int, default=200)
@click.option('--pretrain', is_flag=True)
@click.option('--consistency', type=str, default='augmented')
# @click.option('--identifier', type=str, default='default')
# def experiment(exp, num_epochs, pretrain, identifier):
# log_dir = 'log/{:s}/{:s}'.format(exp, identifier)
# snapshot_dir = 'snapshot/{:s}/{:s}'.format(exp, identifier)
def experiment(exp, num_epochs, pretrain, consistency):
    config = get_config('config.yaml')
    identifier = '{:s}_ndf{:d}_ngf{:d}'.format(
        consistency, config['dis']['ndf'], config['gen']['ngf'])
    log_dir = 'log/{:s}/{:s}'.format(exp, identifier)
    snapshot_dir = 'snapshot/{:s}/{:s}'.format(exp, identifier)
    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs(snapshot_dir, exist_ok=True)

    shutil.copy('config.yaml', '{:s}/{:s}'.format(snapshot_dir, 'config.yaml'))
    batch_size = int(config['batch_size'])
    pool_size = int(config['pool_size'])
    lr = float(config['lr'])
    weight_decay = float(config['weight_decay'])

    device = torch.device('cuda')

    src, tgt = load_source_target_datasets(exp)

    n_ch_s = src.train_X.shape[1]  # number of color channels
    n_ch_t = tgt.train_X.shape[1]  # number of color channels
    n_class = src.n_classes

    train_tfs = get_composed_transforms()
    test_tfs = get_composed_transforms()

    src_train = DADataset(src.train_X, src.train_y, train_tfs)
    src_test = DADataset(src.test_X, src.test_y, train_tfs)
    tgt_train = DADataset(tgt.train_X, tgt.train_y, train_tfs)
    tgt_train = SubsetDataset(tgt_train, range(1000))  # fix indices
    tgt_test = DADataset(tgt.test_X, tgt.test_y, test_tfs)
    del src, tgt

    n_sample = max(len(src_train), len(tgt_train))
    iter_per_epoch = n_sample // batch_size + 1

    cls_s = Classifier(n_class, n_ch_s).to(device)
    cls_t = Classifier(n_class, n_ch_t).to(device)

    if not pretrain:
        load_model(cls_s, 'snapshot/{:s}/pretrain_cls_s.tar'.format(exp))

    gen_s_t_params = {'input_nc': n_ch_s, 'output_nc': n_ch_t}
    gen_t_s_params = {'input_nc': n_ch_t, 'output_nc': n_ch_s}
    gen_s_t = define_G(**{**config['gen'], **gen_s_t_params}).to(device)
    gen_t_s = define_G(**{**config['gen'], **gen_t_s_params}).to(device)

    dis_s = define_D(**{**config['dis'], 'input_nc': n_ch_s}).to(device)
    dis_t = define_D(**{**config['dis'], 'input_nc': n_ch_t}).to(device)

    opt_config = {'lr': lr, 'weight_decay': weight_decay, 'betas': (0.5, 0.99)}
    opt_gen = Adam(chain(gen_s_t.parameters(), gen_t_s.parameters(), \
                         cls_s.parameters(), cls_t.parameters()), **opt_config)
    opt_dis = Adam(chain(dis_s.parameters(), dis_t.parameters()), **opt_config)

    calc_ls = GANLoss(device, use_lsgan=True).to(device)
    calc_ce = torch.nn.CrossEntropyLoss().to(device)
    calc_l1 = torch.nn.L1Loss().to(device)

    fake_src_x_pool = ImagePool(pool_size * batch_size)
    fake_tgt_x_pool = ImagePool(pool_size * batch_size)

    src_train_iter = iter(DataLoader(
        src_train, batch_size=batch_size, num_workers=4,
        sampler=InfiniteSampler(len(src_train))))
    tgt_train_iter = iter(DataLoader(
        tgt_train, batch_size=batch_size, num_workers=4,
        sampler=InfiniteSampler(len(tgt_train))))
    src_test_loader = DataLoader(
        src_test, batch_size=batch_size * 4, num_workers=4)
    tgt_test_loader = DataLoader(
        tgt_test, batch_size=batch_size * 4, num_workers=4)
    print('Training...')

    cls_s.train()
    cls_t.train()

    niter = 0

    if pretrain:
        while True:
            niter += 1
            src_x, src_y = next(src_train_iter)
            loss = calc_ce(cls_s(src_x.to(device)), src_y.to(device))
            opt_gen.zero_grad()
            loss.backward()
            opt_gen.step()

            if niter % iter_per_epoch == 0:
                epoch = niter // iter_per_epoch
                n_err = evaluate_classifier(cls_s, tgt_test_loader, device)
                print(epoch, n_err / len(tgt_test))

                # n_err = evaluate_classifier(cls_s, src_test_loader, device)
                # print(epoch, n_err / len(src_test))

                if epoch >= num_epochs:
                    save_model(cls_s,
                               '{:s}/pretrain_cls_s.tar'.format(snapshot_dir))
                    break
        exit()

    while True:
        niter += 1
        src_x, src_y = next(src_train_iter)
        tgt_x, tgt_y = next(tgt_train_iter)
        src_x, src_y = src_x.to(device), src_y.to(device)
        tgt_x, tgt_y = tgt_x.to(device), tgt_y.to(device)

        fake_tgt_x = gen_s_t(src_x)
        fake_back_src_x = gen_t_s(fake_tgt_x)
        fake_src_x = gen_t_s(tgt_x)
        fake_back_tgt_x = gen_s_t(fake_src_x)

        #################
        # discriminator #
        #################

        loss_dis_s = calc_ls(
            dis_s(fake_src_x_pool.query(fake_src_x.detach())), False)
        loss_dis_s += calc_ls(dis_s(src_x), True)
        loss_dis_t = calc_ls(
            dis_t(fake_tgt_x_pool.query(fake_tgt_x.detach())), False)
        loss_dis_t += calc_ls(dis_t(tgt_x), True)
        loss_dis = loss_dis_s + loss_dis_t

        ##########################
        # generator + classifier #
        ##########################

        # classification
        loss_gen_cls_s = calc_ce(cls_s(src_x), src_y)
        loss_gen_cls_t = calc_ce(cls_t(tgt_x), tgt_y)
        loss_gen_cls = loss_gen_cls_s + loss_gen_cls_t

        # augmented cycle consistency
        if consistency == 'augmented':
            loss_gen_aug_s = calc_ce(cls_s(fake_src_x), tgt_y)
            loss_gen_aug_s += calc_ce(cls_s(fake_back_src_x), src_y)
            loss_gen_aug_t = calc_ce(cls_t(fake_tgt_x), src_y)
            loss_gen_aug_t += calc_ce(cls_t(fake_back_tgt_x), tgt_y)
            loss_gen_aug = loss_gen_aug_s + loss_gen_aug_t
        elif consistency == 'relaxed':
            loss_gen_aug_s = calc_ce(cls_s(fake_back_src_x), src_y)
            loss_gen_aug_t = calc_ce(cls_t(fake_back_tgt_x), tgt_y)
            loss_gen_aug = loss_gen_aug_s + loss_gen_aug_t
        elif consistency == 'simple':
            loss_gen_aug_s = calc_ce(cls_s(fake_src_x), tgt_y)
            loss_gen_aug_t = calc_ce(cls_t(fake_tgt_x), src_y)
            loss_gen_aug = loss_gen_aug_s + loss_gen_aug_t
        elif consistency == 'cycle':
            loss_gen_aug_s = calc_l1(fake_back_src_x, src_x)
            loss_gen_aug_t = calc_l1(fake_back_tgt_x, tgt_x)
            loss_gen_aug = loss_gen_aug_s + loss_gen_aug_t
        else:
            raise NotImplementedError

        # deceive discriminator
        loss_gen_adv_s = calc_ls(dis_s(fake_src_x), True)
        loss_gen_adv_t = calc_ls(dis_t(fake_tgt_x), True)
        loss_gen_adv = loss_gen_adv_s + loss_gen_adv_t

        loss_gen = loss_gen_cls + loss_gen_aug + loss_gen_adv

        opt_dis.zero_grad()
        loss_dis.backward()
        opt_dis.step()

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if niter % 100 == 0 and niter > 0:
            writer.add_scalar('dis/src', loss_dis_s.item(), niter)
            writer.add_scalar('dis/tgt', loss_dis_t.item(), niter)
            writer.add_scalar('gen/cls_s', loss_gen_cls_s.item(), niter)
            writer.add_scalar('gen/cls_t', loss_gen_cls_t.item(), niter)
            writer.add_scalar('gen/aug_s', loss_gen_aug_s.item(), niter)
            writer.add_scalar('gen/aug_t', loss_gen_aug_t.item(), niter)
            writer.add_scalar('gen/adv_s', loss_gen_adv_s.item(), niter)
            writer.add_scalar('gen/adv_t', loss_gen_adv_t.item(), niter)

        if niter % iter_per_epoch == 0:
            epoch = niter // iter_per_epoch

            if epoch % 1 == 0:
                data = []
                for x in [src_x, fake_tgt_x, fake_back_src_x, tgt_x,
                          fake_src_x, fake_back_tgt_x]:
                    x = x.to(torch.device('cpu'))
                    if x.size(1) == 1:
                        x = x.repeat(1, 3, 1, 1)  # grayscale2rgb
                    data.append(x)
                grid = make_grid(torch.cat(tuple(data), dim=0), nrow=16,
                                 normalize=True, range=(-1.0, 1.0))
                writer.add_image('generated', grid, epoch)

            n_err = evaluate_classifier(cls_t, tgt_test_loader, device)
            writer.add_scalar('err_tgt', n_err / len(tgt_test), epoch)

            if epoch % 50 == 0:
                models_dict = {
                    'cls_s': cls_s, 'cls_t': cls_t, 'dis_s': dis_s,
                    'dis_t': dis_t, 'gen_s_t': gen_s_t, 'gen_t_s': gen_t_s}
                filename = '{:s}/epoch{:d}.tar'.format(snapshot_dir, epoch)
                save_models_dict(models_dict, filename)

            if epoch >= num_epochs:
                break


if __name__ == '__main__':
    experiment()
