import click
import numpy as np
import torch
import torch.cuda
from batchup import data_source
from torch.autograd import Variable
from torch.nn import functional as F

from loader import load_source_target_datasets
from net import Classifier
from opt import params, exp_list
from util import load_model
from util import norm_cls_to_gan

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('--exp', type=click.Choice(exp_list), required=True)
@click.option('--model', type=str, required=True)
def experiment(exp, model):
    data_src, data_tgt = load_source_target_datasets(exp)
    del data_src

    n_color = data_tgt.train_X.shape[1]
    res = data_tgt.train_X.shape[-1]

    cls = Classifier(data_tgt.n_classes, n_color, res).cuda()
    load_model(cls, model)

    data_tgt.test_X = norm_cls_to_gan(data_tgt.test_X)

    target_test_ds = data_source.ArrayDataSource(
        [data_tgt.test_X, data_tgt.test_y])

    print('Testing...')
    cls.eval()

    def f_eval(X_sup, y_sup):
        X_var = Variable(torch.from_numpy(X_sup).cuda(),
                         requires_grad=False)
        y_t_prob = F.softmax(cls(X_var), dim=1).data.cpu().numpy()
        y_pred = np.argmax(y_t_prob, axis=1)
        return float((y_pred != y_sup).sum())

    tgt_test_err, = target_test_ds.batch_map_mean(
        f_eval, batch_size=params['batch_size'] * 4)
    print('TEST err={:.3%}'.format(tgt_test_err))


if __name__ == '__main__':
    experiment()
