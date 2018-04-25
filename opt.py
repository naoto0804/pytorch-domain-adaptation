params = {
    'batch_size': 32, 'base_lr': 1e-4,
    'betas': (0.5, 0.999), 'weight_decay': 1e-5, 'pool_size': 50,
    'weight': {'alpha': 1.0, 'gamma': 1.0, 'beta': 10.0, 'mu': 10.0,
               'eta': 1.0, 'new': 1.0},
    'gen_init': {'n_ch': 64, 'n_hidden': 5, 'n_resblock': 4},
    'dis_init': {'n_ch': 64}
}

exp_list = ['svhn_mnist', 'mnist_svhn', 'svhn_mnist_rgb', 'mnist_svhn_rgb',
            'cifar_stl', 'stl_cifar', 'mnist_usps', 'usps_mnist',
            'syndigits_svhn', 'synsigns_gtsrb', 'mnist_mnistm']

arch_list = ['mnist-bn-32-64-256', 'grey-32-64-128-gp',
             'grey-32-64-128-gp-wn', 'grey-32-64-128-gp-nonorm',
             'rgb-128-256-down-gp', 'resnet18-32', 'rgb40-48-96-192-384-gp',
             'rgb40-96-192-384-gp'
             ]
