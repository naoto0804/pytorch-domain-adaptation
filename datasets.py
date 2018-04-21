import os
import sys

if sys.version_info[0] == 2:
    from ConfigParser import RawConfigParser
else:
    from configparser import RawConfigParser

import tables
from batchup.datasets import dataset
from batchup.image.utils import ImageArrayUInt8ToFloat32
import numpy as np
from batchup.datasets import mnist, fashion_mnist, cifar10, svhn, stl, usps
from skimage.transform import downscale_local_mean, resize
import torch
from PIL import Image
from preprocess import OriginalAffineTransform

_CONFIG = None


def get_config():
    global _CONFIG
    if _CONFIG is None:
        if os.path.exists('domain_datasets.cfg'):
            _CONFIG = RawConfigParser()
            _CONFIG.read('domain_datasets.cfg')
        else:
            raise ValueError(
                'Could not find configuration file domain_datasets.cfg')
    return _CONFIG


def get_data_dir(name):
    config = get_config()
    path = config.get('paths', name)
    if path is not None and path != '':
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise ValueError(
                'Configuration file entry for paths:{} does not exist'.format(
                    name))
        return path
    else:
        raise ValueError(
            'Configuration file did not have entry for paths:{}'.format(name))


def _syndigits_train_path():
    return os.path.join(get_data_dir('syn_digits'), 'synth_train_32x32.mat')


def _syndigits_test_path():
    return os.path.join(get_data_dir('syn_digits'), 'synth_test_32x32.mat')


def _syndigits_h5_path():
    return os.path.abspath(
        os.path.join(get_data_dir('syn_digits'), 'syn_digits.h5'))


_TRAIN_SRC = dataset.ExistingSourceFile(_syndigits_train_path, None)
_TEST_SRC = dataset.ExistingSourceFile(_syndigits_test_path, None)


@dataset.fetch_and_convert_dataset(
    [_TRAIN_SRC, _TEST_SRC], _syndigits_h5_path)
def fetch_syn_digits(source_paths, target_path):
    train_path, test_path = source_paths

    f_out = tables.open_file(target_path, mode='w')
    g_out = f_out.create_group(f_out.root, 'syn_digits', 'Syn-Digits data')

    # Load in the training data Matlab file
    print('Converting {} to HDF5...'.format(train_path))
    train_X_u8, train_y = svhn._read_svhn_matlab(train_path)
    f_out.create_array(g_out, 'train_X_u8', train_X_u8)
    f_out.create_array(g_out, 'train_y', train_y)
    del train_X_u8
    del train_y

    # Load in the test data Matlab file
    print('Converting {} to HDF5...'.format(test_path))
    test_X_u8, test_y = svhn._read_svhn_matlab(test_path)
    f_out.create_array(g_out, 'test_X_u8', test_X_u8)
    f_out.create_array(g_out, 'test_y', test_y)
    del test_X_u8
    del test_y

    f_out.close()

    return target_path


def delete_cache():  # pragma: no cover
    dataset.delete_dataset_cache(_syndigits_h5_path())


class SynDigits(object):
    def __init__(self, n_val=10000, val_lower=0.0, val_upper=1.0):
        h5_path = fetch_syn_digits()
        if h5_path is not None:
            f = tables.open_file(h5_path, mode='r')

            train_X_u8 = f.root.syn_digits.train_X_u8
            train_y = f.root.syn_digits.train_y
            self.test_X_u8 = f.root.syn_digits.test_X_u8
            self.test_y = f.root.syn_digits.test_y

            if n_val == 0 or n_val is None:
                self.train_X_u8 = train_X_u8
                self.train_y = train_y
                self.val_X_u8 = np.zeros((0, 3, 32, 32), dtype=np.uint8)
                self.val_y = np.zeros((0,), dtype=np.int32)
            else:
                self.train_X_u8 = train_X_u8[:-n_val]
                self.train_y = train_y[:-n_val]
                self.val_X_u8 = train_X_u8[-n_val:]
                self.val_y = train_y[-n_val:]
        else:
            raise RuntimeError('Could not load Syn-Digits dataset')

        self.train_X = ImageArrayUInt8ToFloat32(self.train_X_u8, val_lower,
                                                val_upper)
        self.val_X = ImageArrayUInt8ToFloat32(self.val_X_u8, val_lower,
                                              val_upper)
        self.test_X = ImageArrayUInt8ToFloat32(self.test_X_u8, val_lower,
                                               val_upper)


class GTSRB(object):
    def __init__(self, n_val=2935, shuffle_seed=12345, val_lower=0.0,
                 val_upper=1.0):
        h5_path = os.path.join(get_data_dir('gtsrb'), 'gtsrb.h5')
        if not os.path.exists(h5_path):
            raise RuntimeError('Could not load GTSRB from {}; please run '
                               '\'prepare_gtsrb.py\' to create it'.format(
                h5_path))

        f = tables.open_file(h5_path, mode='r')

        train_X_u8 = f.root.gtsrb.train_X_u8
        train_y = f.root.gtsrb.train_y
        test_X_u8 = f.root.gtsrb.test_X_u8
        test_y = f.root.gtsrb.test_y

        shuffle_rng = np.random.RandomState(shuffle_seed)
        train_ndx = shuffle_rng.permutation(len(train_X_u8))
        test_ndx = shuffle_rng.permutation(len(test_X_u8))
        train_X_u8 = train_X_u8[:][train_ndx]
        train_y = train_y[:][train_ndx]
        test_X_u8 = test_X_u8[:][test_ndx]
        test_y = test_y[:][test_ndx]
        if n_val == 0 or n_val is None:
            self.train_X_u8, self.train_y = train_X_u8, train_y
            self.val_X_u8 = np.zeros((0, 3, 40, 40), dtype=np.uint8)
            self.val_y = np.zeros((0,), dtype=np.int32)
        else:
            self.train_X_u8, self.val_X_u8 = train_X_u8[:-n_val], train_X_u8[
                                                                  -n_val:]
            self.train_y, self.val_y = train_y[:-n_val], train_y[-n_val:]
        self.test_X_u8 = test_X_u8
        self.test_y = test_y

        self.n_classes = 43

        self.train_X = ImageArrayUInt8ToFloat32(self.train_X_u8, val_lower,
                                                val_upper)
        self.val_X = ImageArrayUInt8ToFloat32(self.val_X_u8, val_lower,
                                              val_upper)
        self.test_X = ImageArrayUInt8ToFloat32(self.test_X_u8, val_lower,
                                               val_upper)


class SynSigns(object):
    def __init__(self, n_val=10000, n_test=10000, shuffle_seed=12345,
                 val_lower=0.0, val_upper=1.0):
        h5_path = os.path.join(get_data_dir('syn_signs'), 'syn_signs.h5')
        if not os.path.exists(h5_path):
            raise RuntimeError('Could not load Syn-Signs from {}; please run '
                               '\'prepare_synsigns.py\' to create it'.format(
                h5_path))

        f = tables.open_file(h5_path, mode='r')

        X_u8 = f.root.syn_signs.X_u8
        y = f.root.syn_signs.y

        shuffle_rng = np.random.RandomState(shuffle_seed)
        ndx = shuffle_rng.permutation(len(X_u8))
        X_u8 = X_u8[:][ndx]
        y = y[:][ndx]
        n_vt = n_val + n_test
        self.train_X_u8 = X_u8[:-n_vt]
        self.train_y = y[:-n_vt]
        valtest_X_u8 = X_u8[-n_vt:]
        valtest_y = y[-n_vt:]
        self.val_X_u8 = valtest_X_u8[:n_val]
        self.val_y = valtest_y[:n_val]
        self.test_X_u8 = valtest_X_u8[n_val:]
        self.test_y = valtest_y[n_val:]

        self.n_classes = 43

        self.train_X = ImageArrayUInt8ToFloat32(self.train_X_u8, val_lower,
                                                val_upper)
        self.val_X = ImageArrayUInt8ToFloat32(self.val_X_u8, val_lower,
                                              val_upper)
        self.test_X = ImageArrayUInt8ToFloat32(self.test_X_u8, val_lower,
                                               val_upper)


def rgb2grey_tensor(X):
    return (X[:, 0:1, :, :] * 0.2125) + (X[:, 1:2, :, :] * 0.7154) + (
            X[:, 2:3, :, :] * 0.0721)


# Dataset loading functions
def load_svhn(zero_centre=False, greyscale=False, val=False, extra=False):
    #
    #
    # Load SVHN
    #
    #

    print('Loading SVHN...')
    if val:
        d_svhn = svhn.SVHN(n_val=10000)
    else:
        d_svhn = svhn.SVHN(n_val=0)

    if extra:
        d_extra = svhn.SVHNExtra()
    else:
        d_extra = None

    d_svhn.train_X = d_svhn.train_X[:]
    d_svhn.val_X = d_svhn.val_X[:]
    d_svhn.test_X = d_svhn.test_X[:]
    d_svhn.train_y = d_svhn.train_y[:]
    d_svhn.val_y = d_svhn.val_y[:]
    d_svhn.test_y = d_svhn.test_y[:]

    if extra:
        d_svhn.train_X = np.append(d_svhn.train_X, d_extra.X[:], axis=0)
        d_svhn.train_y = np.append(d_svhn.train_y, d_extra.y[:], axis=0)

    if greyscale:
        d_svhn.train_X = rgb2grey_tensor(d_svhn.train_X)
        d_svhn.val_X = rgb2grey_tensor(d_svhn.val_X)
        d_svhn.test_X = rgb2grey_tensor(d_svhn.test_X)

    if zero_centre:
        d_svhn.train_X = d_svhn.train_X * 2.0 - 1.0
        d_svhn.val_X = d_svhn.val_X * 2.0 - 1.0
        d_svhn.test_X = d_svhn.test_X * 2.0 - 1.0

    print(
        'SVHN: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, test: X.shape={}, y.shape={}'.format(
            d_svhn.train_X.shape, d_svhn.train_y.shape, d_svhn.val_X.shape,
            d_svhn.val_y.shape, d_svhn.test_X.shape,
            d_svhn.test_y.shape))

    print('SVHN: train: X.min={}, X.max={}'.format(
        d_svhn.train_X.min(), d_svhn.train_X.max()))

    d_svhn.n_classes = 10

    return d_svhn


def load_mnist(invert=False, zero_centre=False, intensity_scale=1.0, val=False,
               pad32=False, downscale_x=1,
               rgb=False):
    #
    #
    # Load MNIST
    #
    #

    print('Loading MNIST...')

    if val:
        d_mnist = mnist.MNIST(n_val=10000)
    else:
        d_mnist = mnist.MNIST(n_val=0)

    d_mnist.train_X = d_mnist.train_X[:]
    d_mnist.val_X = d_mnist.val_X[:]
    d_mnist.test_X = d_mnist.test_X[:]
    d_mnist.train_y = d_mnist.train_y[:]
    d_mnist.val_y = d_mnist.val_y[:]
    d_mnist.test_y = d_mnist.test_y[:]

    if downscale_x != 1:
        d_mnist.train_X = downscale_local_mean(d_mnist.train_X,
                                               (1, 1, 1, downscale_x))
        d_mnist.val_X = downscale_local_mean(d_mnist.val_X,
                                             (1, 1, 1, downscale_x))
        d_mnist.test_X = downscale_local_mean(d_mnist.test_X,
                                              (1, 1, 1, downscale_x))

    if pad32:
        py = (32 - d_mnist.train_X.shape[2]) // 2
        px = (32 - d_mnist.train_X.shape[3]) // 2
        # Pad 28x28 to 32x32
        d_mnist.train_X = np.pad(d_mnist.train_X,
                                 [(0, 0), (0, 0), (py, py), (px, px)],
                                 mode='constant')
        d_mnist.val_X = np.pad(d_mnist.val_X,
                               [(0, 0), (0, 0), (py, py), (px, px)],
                               mode='constant')
        d_mnist.test_X = np.pad(d_mnist.test_X,
                                [(0, 0), (0, 0), (py, py), (px, px)],
                                mode='constant')

    if invert:
        # Invert
        d_mnist.train_X = 1.0 - d_mnist.train_X
        d_mnist.val_X = 1.0 - d_mnist.val_X
        d_mnist.test_X = 1.0 - d_mnist.test_X

    if intensity_scale != 1.0:
        d_mnist.train_X = (d_mnist.train_X - 0.5) * intensity_scale + 0.5
        d_mnist.val_X = (d_mnist.val_X - 0.5) * intensity_scale + 0.5
        d_mnist.test_X = (d_mnist.test_X - 0.5) * intensity_scale + 0.5

    if zero_centre:
        d_mnist.train_X = d_mnist.train_X * 2.0 - 1.0
        d_mnist.test_X = d_mnist.test_X * 2.0 - 1.0

    if rgb:
        d_mnist.train_X = np.concatenate([d_mnist.train_X] * 3, axis=1)
        d_mnist.val_X = np.concatenate([d_mnist.val_X] * 3, axis=1)
        d_mnist.test_X = np.concatenate([d_mnist.test_X] * 3, axis=1)

    print(
        'MNIST: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, test: X.shape={}, y.shape={}'.format(
            d_mnist.train_X.shape, d_mnist.train_y.shape,
            d_mnist.val_X.shape, d_mnist.val_y.shape,
            d_mnist.test_X.shape, d_mnist.test_y.shape))

    print('MNIST: train: X.min={}, X.max={}'.format(
        d_mnist.train_X.min(), d_mnist.train_X.max()))

    d_mnist.n_classes = 10

    return d_mnist


def load_fashion_mnist(invert=False, zero_centre=False, intensity_scale=1.0,
                       val=False, pad32=False, downscale_x=1):
    #
    #
    # Load MNIST
    #
    #

    print('Loading Fashion MNIST...')

    if val:
        d_fmnist = fashion_mnist.FashionMNIST(n_val=10000)
    else:
        d_fmnist = fashion_mnist.FashionMNIST(n_val=0)

    d_fmnist.train_X = d_fmnist.train_X[:]
    d_fmnist.val_X = d_fmnist.val_X[:]
    d_fmnist.test_X = d_fmnist.test_X[:]
    d_fmnist.train_y = d_fmnist.train_y[:]
    d_fmnist.val_y = d_fmnist.val_y[:]
    d_fmnist.test_y = d_fmnist.test_y[:]

    if downscale_x != 1:
        d_fmnist.train_X = downscale_local_mean(d_fmnist.train_X,
                                                (1, 1, 1, downscale_x))
        d_fmnist.val_X = downscale_local_mean(d_fmnist.val_X,
                                              (1, 1, 1, downscale_x))
        d_fmnist.test_X = downscale_local_mean(d_fmnist.test_X,
                                               (1, 1, 1, downscale_x))

    if pad32:
        py = (32 - d_fmnist.train_X.shape[2]) // 2
        px = (32 - d_fmnist.train_X.shape[3]) // 2
        # Pad 28x28 to 32x32
        d_fmnist.train_X = np.pad(d_fmnist.train_X,
                                  [(0, 0), (0, 0), (py, py), (px, px)],
                                  mode='constant')
        d_fmnist.val_X = np.pad(d_fmnist.val_X,
                                [(0, 0), (0, 0), (py, py), (px, px)],
                                mode='constant')
        d_fmnist.test_X = np.pad(d_fmnist.test_X,
                                 [(0, 0), (0, 0), (py, py), (px, px)],
                                 mode='constant')

    if invert:
        # Invert
        d_fmnist.train_X = 1.0 - d_fmnist.train_X
        d_fmnist.val_X = 1.0 - d_fmnist.val_X
        d_fmnist.test_X = 1.0 - d_fmnist.test_X

    if intensity_scale != 1.0:
        d_fmnist.train_X = (d_fmnist.train_X - 0.5) * intensity_scale + 0.5
        d_fmnist.val_X = (d_fmnist.val_X - 0.5) * intensity_scale + 0.5
        d_fmnist.test_X = (d_fmnist.test_X - 0.5) * intensity_scale + 0.5

    if zero_centre:
        d_fmnist.train_X = d_fmnist.train_X * 2.0 - 1.0
        d_fmnist.test_X = d_fmnist.test_X * 2.0 - 1.0

    print(
        'Fashion MNIST: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, '
        'test: X.shape={}, y.shape={}'.format(
            d_fmnist.train_X.shape, d_fmnist.train_y.shape,
            d_fmnist.val_X.shape, d_fmnist.val_y.shape,
            d_fmnist.test_X.shape, d_fmnist.test_y.shape))

    print('Fashion MNIST: train: X.min={}, X.max={}'.format(
        d_fmnist.train_X.min(), d_fmnist.train_X.max()))

    d_fmnist.n_classes = 10

    return d_fmnist


def load_usps(invert=False, zero_centre=False, val=False, scale28=False):
    #
    #
    # Load USPS
    #
    #

    print('Loading USPS...')

    if val:
        d_usps = usps.USPS()
    else:
        d_usps = usps.USPS(n_val=None)

    d_usps.train_X = d_usps.train_X[:]
    d_usps.val_X = d_usps.val_X[:]
    d_usps.test_X = d_usps.test_X[:]
    d_usps.train_y = d_usps.train_y[:]
    d_usps.val_y = d_usps.val_y[:]
    d_usps.test_y = d_usps.test_y[:]

    if scale28:
        def _resize_tensor(X):
            X_prime = np.zeros((X.shape[0], 1, 28, 28), dtype=np.float32)
            for i in range(X.shape[0]):
                X_prime[i, 0, :, :] = resize(X[i, 0, :, :], (28, 28),
                                             mode='constant')
            return X_prime

        # Scale 16x16 to 28x28
        d_usps.train_X = _resize_tensor(d_usps.train_X)
        d_usps.val_X = _resize_tensor(d_usps.val_X)
        d_usps.test_X = _resize_tensor(d_usps.test_X)

    if invert:
        # Invert
        d_usps.train_X = 1.0 - d_usps.train_X
        d_usps.val_X = 1.0 - d_usps.val_X
        d_usps.test_X = 1.0 - d_usps.test_X

    if zero_centre:
        d_usps.train_X = d_usps.train_X * 2.0 - 1.0
        d_usps.test_X = d_usps.test_X * 2.0 - 1.0

    print(
        'USPS: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, test: X.shape={}, y.shape={}'.format(
            d_usps.train_X.shape, d_usps.train_y.shape,
            d_usps.val_X.shape, d_usps.val_y.shape,
            d_usps.test_X.shape, d_usps.test_y.shape))

    print('USPS: train: X.min={}, X.max={}'.format(
        d_usps.train_X.min(), d_usps.train_X.max()))

    d_usps.n_classes = 10

    return d_usps


def load_cifar10(range_01=False, val=False):
    #
    #
    # Load CIFAR-10 for adaptation with STL
    #
    #

    print('Loading CIFAR-10...')
    if val:
        d_cifar = cifar10.CIFAR10(n_val=5000)
    else:
        d_cifar = cifar10.CIFAR10(n_val=0)

    d_cifar.train_X = d_cifar.train_X[:]
    d_cifar.val_X = d_cifar.val_X[:]
    d_cifar.test_X = d_cifar.test_X[:]
    d_cifar.train_y = d_cifar.train_y[:]
    d_cifar.val_y = d_cifar.val_y[:]
    d_cifar.test_y = d_cifar.test_y[:]

    # Remap class indices so that the frog class (6) has an index of -1 as it does not appear int the STL dataset
    cls_mapping = np.array([0, 1, 2, 3, 4, 5, -1, 6, 7, 8])
    d_cifar.train_y = cls_mapping[d_cifar.train_y]
    d_cifar.val_y = cls_mapping[d_cifar.val_y]
    d_cifar.test_y = cls_mapping[d_cifar.test_y]

    # Remove all samples from skipped classes
    train_mask = d_cifar.train_y != -1
    val_mask = d_cifar.val_y != -1
    test_mask = d_cifar.test_y != -1

    d_cifar.train_X = d_cifar.train_X[train_mask]
    d_cifar.train_y = d_cifar.train_y[train_mask]
    d_cifar.val_X = d_cifar.val_X[val_mask]
    d_cifar.val_y = d_cifar.val_y[val_mask]
    d_cifar.test_X = d_cifar.test_X[test_mask]
    d_cifar.test_y = d_cifar.test_y[test_mask]

    if range_01:
        d_cifar.train_X = d_cifar.train_X * 2.0 - 1.0
        d_cifar.val_X = d_cifar.val_X * 2.0 - 1.0
        d_cifar.test_X = d_cifar.test_X * 2.0 - 1.0

    print(
        'CIFAR-10: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, test: X.shape={}, y.shape={}'.format(
            d_cifar.train_X.shape, d_cifar.train_y.shape, d_cifar.val_X.shape,
            d_cifar.val_y.shape, d_cifar.test_X.shape,
            d_cifar.test_y.shape))

    print('CIFAR-10: train: X.min={}, X.max={}'.format(
        d_cifar.train_X.min(), d_cifar.train_X.max()))

    d_cifar.n_classes = 9

    return d_cifar


def load_stl(zero_centre=False, val=False):
    #
    #
    # Load STL for adaptation with CIFAR-10
    #
    #

    print('Loading STL...')
    if val:
        d_stl = stl.STL()
    else:
        d_stl = stl.STL(n_val_folds=0)

    d_stl.train_X = d_stl.train_X[:]
    d_stl.val_X = d_stl.val_X[:]
    d_stl.test_X = d_stl.test_X[:]
    d_stl.train_y = d_stl.train_y[:]
    d_stl.val_y = d_stl.val_y[:]
    d_stl.test_y = d_stl.test_y[:]

    # Remap class indices to match CIFAR-10:
    cls_mapping = np.array([0, 2, 1, 3, 4, 5, 6, -1, 7, 8])
    d_stl.train_y = cls_mapping[d_stl.train_y]
    d_stl.val_y = cls_mapping[d_stl.val_y]
    d_stl.test_y = cls_mapping[d_stl.test_y]

    d_stl.train_X = d_stl.train_X[:]
    d_stl.val_X = d_stl.val_X[:]
    d_stl.test_X = d_stl.test_X[:]

    # Remove all samples from class -1 (monkey) as it does not appear int the CIFAR-10 dataset
    train_mask = d_stl.train_y != -1
    val_mask = d_stl.val_y != -1
    test_mask = d_stl.test_y != -1

    d_stl.train_X = d_stl.train_X[train_mask]
    d_stl.train_y = d_stl.train_y[train_mask]
    d_stl.val_X = d_stl.val_X[val_mask]
    d_stl.val_y = d_stl.val_y[val_mask]
    d_stl.test_X = d_stl.test_X[test_mask]
    d_stl.test_y = d_stl.test_y[test_mask]

    # Downsample images from 96x96 to 32x32
    d_stl.train_X = downscale_local_mean(d_stl.train_X, (1, 1, 3, 3))
    d_stl.val_X = downscale_local_mean(d_stl.val_X, (1, 1, 3, 3))
    d_stl.test_X = downscale_local_mean(d_stl.test_X, (1, 1, 3, 3))

    if zero_centre:
        d_stl.train_X = d_stl.train_X * 2.0 - 1.0
        d_stl.val_X = d_stl.val_X * 2.0 - 1.0
        d_stl.test_X = d_stl.test_X * 2.0 - 1.0

    print(
        'STL: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, test: X.shape={}, y.shape={}'.format(
            d_stl.train_X.shape, d_stl.train_y.shape, d_stl.val_X.shape,
            d_stl.val_y.shape, d_stl.test_X.shape,
            d_stl.test_y.shape))

    print('STL: train: X.min={}, X.max={}'.format(
        d_stl.train_X.min(), d_stl.train_X.max()))

    d_stl.n_classes = 9

    return d_stl


def load_syn_digits(zero_centre=False, greyscale=False, val=False):
    #
    #
    # Load syn digits
    #
    #

    print('Loading Syn-digits...')
    if val:
        d_synd = SynDigits(n_val=10000)
    else:
        d_synd = SynDigits(n_val=0)

    d_synd.train_X = d_synd.train_X[:]
    d_synd.val_X = d_synd.val_X[:]
    d_synd.test_X = d_synd.test_X[:]
    d_synd.train_y = d_synd.train_y[:]
    d_synd.val_y = d_synd.val_y[:]
    d_synd.test_y = d_synd.test_y[:]

    if greyscale:
        d_synd.train_X = rgb2grey_tensor(d_synd.train_X)
        d_synd.val_X = rgb2grey_tensor(d_synd.val_X)
        d_synd.test_X = rgb2grey_tensor(d_synd.test_X)

    if zero_centre:
        d_synd.train_X = d_synd.train_X * 2.0 - 1.0
        d_synd.val_X = d_synd.val_X * 2.0 - 1.0
        d_synd.test_X = d_synd.test_X * 2.0 - 1.0

    print(
        'SynDigits: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, test: X.shape={}, y.shape={}'.format(
            d_synd.train_X.shape, d_synd.train_y.shape, d_synd.val_X.shape,
            d_synd.val_y.shape, d_synd.test_X.shape,
            d_synd.test_y.shape))

    print('SynDigits: train: X.min={}, X.max={}'.format(
        d_synd.train_X.min(), d_synd.train_X.max()))

    d_synd.n_classes = 10

    return d_synd


def load_syn_signs(zero_centre=False, greyscale=False, val=False):
    #
    #
    # Load syn digits
    #
    #

    print('Loading Syn-signs...')
    if val:
        d_syns = SynSigns(n_val=10000, n_test=10000)
    else:
        d_syns = SynSigns(n_val=0, n_test=10000)

    d_syns.train_X = d_syns.train_X[:]
    d_syns.val_X = d_syns.val_X[:]
    d_syns.test_X = d_syns.test_X[:]
    d_syns.train_y = d_syns.train_y[:]
    d_syns.val_y = d_syns.val_y[:]
    d_syns.test_y = d_syns.test_y[:]

    if greyscale:
        d_syns.train_X = rgb2grey_tensor(d_syns.train_X)
        d_syns.val_X = rgb2grey_tensor(d_syns.val_X)
        d_syns.test_X = rgb2grey_tensor(d_syns.test_X)

    if zero_centre:
        d_syns.train_X = d_syns.train_X * 2.0 - 1.0
        d_syns.val_X = d_syns.val_X * 2.0 - 1.0
        d_syns.test_X = d_syns.test_X * 2.0 - 1.0

    print(
        'SynSigns: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, '
        'test: X.shape={}, y.shape={}'.format(
            d_syns.train_X.shape, d_syns.train_y.shape, d_syns.val_X.shape,
            d_syns.val_y.shape, d_syns.test_X.shape,
            d_syns.test_y.shape))

    print('SynSigns: train: X.min={}, X.max={}'.format(
        d_syns.train_X.min(), d_syns.train_X.max()))

    d_syns.n_classes = 43

    return d_syns


def load_gtsrb(zero_centre=False, greyscale=False, val=False):
    #
    #
    # Load syn digits
    #
    #

    print('Loading GTSRB...')
    if val:
        d_gts = GTSRB(n_val=10000)
    else:
        d_gts = GTSRB(n_val=0)

    d_gts.train_X = d_gts.train_X[:]
    d_gts.val_X = d_gts.val_X[:]
    d_gts.test_X = d_gts.test_X[:]
    d_gts.train_y = d_gts.train_y[:]
    d_gts.val_y = d_gts.val_y[:]
    d_gts.test_y = d_gts.test_y[:]

    if greyscale:
        d_gts.train_X = rgb2grey_tensor(d_gts.train_X)
        d_gts.val_X = rgb2grey_tensor(d_gts.val_X)
        d_gts.test_X = rgb2grey_tensor(d_gts.test_X)

    if zero_centre:
        d_gts.train_X = d_gts.train_X * 2.0 - 1.0
        d_gts.val_X = d_gts.val_X * 2.0 - 1.0
        d_gts.test_X = d_gts.test_X * 2.0 - 1.0

    print('GTSRB: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, '
          'test: X.shape={}, y.shape={}'.format(
        d_gts.train_X.shape, d_gts.train_y.shape, d_gts.val_X.shape,
        d_gts.val_y.shape, d_gts.test_X.shape,
        d_gts.test_y.shape))

    print('GTSRB: train: X.min={}, X.max={}'.format(
        d_gts.train_X.min(), d_gts.train_X.max()))

    d_gts.n_classes = 43

    return d_gts


def load_source_target_datasets(exp):
    if exp == 'svhn_mnist':
        d_source = load_svhn(zero_centre=False, greyscale=False)
        d_target = load_mnist(invert=False, zero_centre=False,
                              pad32=True, rgb=False)
    elif exp == 'mnist_svhn':
        d_source = load_mnist(invert=False, zero_centre=False,
                              pad32=True, rgb=False)
        d_target = load_svhn(zero_centre=False, greyscale=False)
    elif exp == 'svhn_mnist_rgb':
        d_source = load_svhn(zero_centre=False, greyscale=False)
        d_target = load_mnist(invert=False, zero_centre=False,
                              pad32=True, rgb=True)
    elif exp == 'mnist_svhn_rgb':
        d_source = load_mnist(invert=False, zero_centre=False,
                              pad32=True, rgb=True)
        d_target = load_svhn(zero_centre=False, greyscale=False)
    elif exp == 'cifar_stl':
        d_source = load_cifar10(range_01=False)
        d_target = load_stl(zero_centre=False, val=False)
    elif exp == 'stl_cifar':
        d_source = load_stl(zero_centre=False)
        d_target = load_cifar10(range_01=False, val=False)
    elif exp == 'mnist_usps':
        d_source = load_mnist(zero_centre=False)
        d_target = load_usps(zero_centre=False, scale28=True,
                             val=False)
    elif exp == 'usps_mnist':
        d_source = load_usps(zero_centre=False, scale28=True)
        d_target = load_mnist(zero_centre=False, val=False)
    elif exp == 'syndigits_svhn':
        d_source = load_syn_digits(zero_centre=False)
        d_target = load_svhn(zero_centre=False, val=False)
    elif exp == 'synsigns_gtsrb':
        d_source = load_syn_signs(zero_centre=False)
        d_target = load_gtsrb(zero_centre=False, val=False)
    else:
        raise RuntimeError('{:s}:Not implemented'.format(exp))
    return d_source, d_target


class DADataset(torch.utils.data.Dataset):
    # wrapper to apply transform only on images
    def __init__(self, images, targets=None, transform=None, affine=False):
        self.images = np.uint8(images * 255.0).transpose(0, 2, 3, 1)
        if targets is not None:
            self.targets = torch.from_numpy(targets).long()
        else:
            self.targets = None
        self.transform = transform
        self.affine = affine
        self.affine_transform = OriginalAffineTransform()

    def __getitem__(self, index):
        img = self.images[index]
        # Points outside the boundaries of the input should be filled
        # However, affinetransform in torchvision does not support this
        if self.affine:
            img = self.affine_transform(img)
        img = Image.fromarray(img.squeeze())
        if self.transform is not None:
            img = self.transform(img)
        if self.targets is None:
            return img
        else:
            return img, self.targets[index]

    def __len__(self):
        return len(self.images)
