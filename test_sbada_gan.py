import click
import torch.cuda
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets import DADataset
from datasets import load_source_target_datasets
from net import Classifier, Generator
from opt import exp_list, params
from preprocess import get_composed_transforms
from util.io import load_models_dict

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('--exp', type=click.Choice(exp_list), required=True)
@click.option('--snapshot', type=str, required=True)
@click.option('--mix_ratio', type=float, default=1.0)
def experiment(exp, snapshot, mix_ratio):
    src, tgt = load_source_target_datasets(exp)
    n_ch_s = src.train_X.shape[1]  # number of color channels
    n_ch_t = tgt.train_X.shape[1]  # number of color channels
    res = src.train_X.shape[-1]  # size of image
    n_classes = src.n_classes

    cls_s = Classifier(n_classes, n_ch_s, res).cuda()
    cls_t = Classifier(n_classes, n_ch_t, res).cuda()
    add_params = {'res': res, 'n_c_in': n_ch_t, 'n_c_out': n_ch_s}
    gen_t_s = Generator(**{**params['gen_init'], **add_params}).cuda()

    models_dict = {'cls_s': cls_s, 'cls_t': cls_t, 'gen_t_s': gen_t_s}
    load_models_dict(models_dict, snapshot)

    test_tfs = get_composed_transforms(train=False, hflip=False)
    tgt_test = DADataset(tgt.test_X, tgt.test_y, test_tfs, False)
    tgt_test_loader = DataLoader(
        tgt_test, batch_size=params['batch_size'] * 4, num_workers=4)

    print('Testing...')
    cls_t.eval()

    n_err = 0
    for batch_idx, (tgt_X, tgt_y) in enumerate(tgt_test_loader):
        tgt_X = Variable(tgt_X.cuda(), requires_grad=False)
        prob_y_from_t = F.softmax(cls_t(tgt_X), dim=1).data.cpu()
        prob_y_from_s = F.softmax(cls_s(gen_t_s(tgt_X)), dim=1).data.cpu()
        prob_y = (1 - mix_ratio) * prob_y_from_s + mix_ratio * prob_y_from_t
        pred_y = torch.max(prob_y, dim=1)[1]  # (values, indices)
        n_err += (pred_y != tgt_y).sum()
    print('Err:', n_err / len(tgt_test))


if __name__ == '__main__':
    experiment()
