# params = {
#     'res_image': 32,
#     'n_class': 10,
#     'base_lr': 1e-3,
#     'weight_decay': 1e-5,
#     'betas': (0.5, 0.999),
#     'loss': {'dis': 1.0, 'gen': 1.0, 'task': 1.0},
#     'gen': {'n_ch': 32, 'n_hidden': 10, 'n_resblock': 6},
#     'dis': {'n_ch': 128, 'dr_prob': 0.1, 'noise_sigma': 0.2}
# }


# SBADA-GAN
params = {
    'batch_size': 32, 'base_lr': 1e-4, 'num_epochs': 500,
    'weight_decay': 1e-5, 'pool_size': 50,
    'weight': {'alpha': 1.0, 'gamma': 1.0, 'beta': 10.0, 'mu': 10.0,
               'eta': 1.0, 'new': 1.0},
    'gen_init': {'n_ch': 64, 'n_hidden': 5, 'n_resblock': 4},
    'dis_init': {'n_ch': 64}
}

exp_list = ['svhn_mnist', 'mnist_svhn', 'svhn_mnist_rgb', 'mnist_svhn_rgb',
            'cifar_stl', 'stl_cifar', 'mnist_usps', 'usps_mnist',
            'syndigits_svhn', 'synsigns_gtsrb']

arch_list = ['mnist-bn-32-64-256', 'grey-32-64-128-gp',
             'grey-32-64-128-gp-wn', 'grey-32-64-128-gp-nonorm',
             'rgb-128-256-down-gp', 'resnet18-32', 'rgb40-48-96-192-384-gp',
             'rgb40-96-192-384-gp'
             ]
