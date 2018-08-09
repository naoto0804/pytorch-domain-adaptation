import click
import torch.cuda
from torch.nn import functional as F
from torch.utils.data import DataLoader

from util.datasets import DADataset, load_source_target_datasets
from util.io import load_models_dict, get_config
from util.net import LenetClassifier, Generator
from util.opt import exp_list
from util.preprocess import get_composed_transforms

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('--exp', type=click.Choice(exp_list), required=True)
@click.option('--snapshot', type=str, required=True)
@click.option('--mix_ratio', type=float, default=1.0)
def experiment(exp, snapshot, mix_ratio):
    config = get_config('config.yaml')
    batch_size = int(config['batch_size'])

    device = torch.device('cuda')
    src, tgt = load_source_target_datasets(exp)
    n_ch_s = src.train_X.shape[1]  # number of color channels
    n_ch_t = tgt.train_X.shape[1]  # number of color channels
    res = src.train_X.shape[-1]  # size of image
    n_classes = src.n_classes

    cls_s = LenetClassifier(n_classes, n_ch_s, res).to(device)
    cls_t = LenetClassifier(n_classes, n_ch_t, res).to(device)
    add_params = {'res': res, 'n_c_in': n_ch_t, 'n_c_out': n_ch_s}
    gen_t_s = Generator(**{**config['gen_init'], **add_params}).to(device)

    models_dict = {'cls_s': cls_s, 'cls_t': cls_t, 'gen_t_s': gen_t_s}
    load_models_dict(models_dict, snapshot)

    test_tfs = get_composed_transforms(train=False, hflip=False)
    tgt_test = DADataset(tgt.test_X, tgt.test_y, test_tfs, False)
    tgt_test_loader = DataLoader(
        tgt_test, batch_size=batch_size * 4, num_workers=4)

    print('Testing...')
    cls_t.eval()

    n_err = 0
    for tgt_x, tgt_y in tgt_test_loader:
        tgt_x = tgt_x.to(device)
        prob_y_from_t = F.softmax(cls_t(tgt_x), dim=1)
        prob_y_from_s = F.softmax(cls_s(gen_t_s(tgt_x)), dim=1)
        prob_y = (1 - mix_ratio) * prob_y_from_s + mix_ratio * prob_y_from_t
        pred_y = torch.max(prob_y, dim=1)[1].to(torch.device('cpu'))
        n_err += (pred_y != tgt_y).sum().item()
    print('Err:', n_err / len(tgt_test))


if __name__ == '__main__':
    experiment()
