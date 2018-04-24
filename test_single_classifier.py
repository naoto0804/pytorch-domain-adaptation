import click
import torch
import torch.cuda
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets import DADataset
from datasets import load_source_target_datasets
from net import Classifier
from opt import params, exp_list
from preprocess import get_composed_transforms
from util.io import load_model

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('--exp', type=click.Choice(exp_list), required=True)
@click.option('--modelname', type=str, required=True)
def experiment(exp, modelname):
    _, tgt = load_source_target_datasets(exp)
    n_ch_t = tgt.train_X.shape[1]  # number of color channels
    res = tgt.train_X.shape[-1]  # size of image
    n_classes = tgt.n_classes

    cls = Classifier(n_classes, n_ch_t, res).cuda()
    load_model(cls, modelname)

    test_tfs = get_composed_transforms(train=False, hflip=False)
    tgt_test = DADataset(tgt.test_X, tgt.test_y, test_tfs, False)
    tgt_test_loader = DataLoader(
        tgt_test, batch_size=params['batch_size'] * 4, num_workers=4)

    print('Testing...')
    cls.eval()

    n_err = 0
    for batch_idx, (tgt_x, tgt_y) in enumerate(tgt_test_loader):
        tgt_x = Variable(tgt_x.cuda(), requires_grad=False)
        prob_y = F.softmax(cls(tgt_x), dim=1).data.cpu()
        pred_y = torch.max(prob_y, dim=1)[1]  # (values, indices)
        n_err += (pred_y != tgt_y).sum()
    print('Err:', n_err / len(tgt_test))


if __name__ == '__main__':
    experiment()
