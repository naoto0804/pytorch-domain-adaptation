import click
import numpy as np
import torch
import torch.cuda
from batchup import data_source
from torch.autograd import Variable
from torch.nn import functional as F

from loader import load_source_target_datasets
from net import Classifier, Generator
from opt import exp_list, params
from util import norm_cls_to_gan, load_models_dict

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('--exp', type=click.Choice(exp_list), required=True)
@click.option('--snapshot', type=str, required=True)
@click.option('--mix_ratio', type=str, default=1.0)
def experiment(exp, snapshot, mix_ratio):
    data_src, data_tgt = load_source_target_datasets(exp)

    n_color_s = data_src.train_X.shape[1]
    n_color_t = data_tgt.train_X.shape[1]
    res = data_tgt.train_X.shape[-1]
    del data_src

    cls_s = Classifier(data_tgt.n_classes, n_color_s, res).cuda()
    cls_t = Classifier(data_tgt.n_classes, n_color_t, res).cuda()
    add_params = {'res': res, 'n_c_in': n_color_t, 'n_c_out': n_color_s}
    gen_t_s = Generator(**{**params['gen_init'], **add_params}).cuda()

    models_dict = {'cls_s': cls_s, 'cls_t': cls_t, 'gen_t_s': gen_t_s}
    load_models_dict(models_dict, snapshot)

    data_tgt.test_X = norm_cls_to_gan(data_tgt.test_X)
    target_test_ds = data_source.ArrayDataSource(
        [data_tgt.test_X, data_tgt.test_y])

    print('Testing...')
    cls_t.eval()

    def f_eval(X_sup, y_sup):
        X_var = Variable(torch.from_numpy(X_sup).cuda(),
                         requires_grad=False)
        y_prob_t = F.softmax(cls_t(X_var), dim=1).data.cpu().numpy()
        y_prob_s = F.softmax(cls_s(gen_t_s(X_var)), dim=1).data.cpu().numpy()
        y_prob = (1 - mix_ratio) * y_prob_t + mix_ratio * y_prob_s
        y_pred = np.argmax(y_prob, axis=1)
        return float((y_pred != y_sup).sum())

    tgt_test_err, = target_test_ds.batch_map_mean(
        f_eval, batch_size=params['batch_size'] * 4)
    print('TEST err={:.3%}'.format(tgt_test_err))


if __name__ == '__main__':
    experiment()
