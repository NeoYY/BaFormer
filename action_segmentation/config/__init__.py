import torch

from .defaults import get_default_config
from .config_node import configurable


def update_config(config):
    if config.dataset.name in ['CIFAR10', 'CIFAR100']:
        dataset_dir = f'~/.torch/datasets/{config.dataset.name}'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 32
        config.dataset.n_channels = 3
        config.dataset.n_classes = int(config.dataset.name[5:])
    elif config.dataset.name in ['MNIST', 'FashionMNIST', 'KMNIST']:
        dataset_dir = '~/.torch/datasets'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 28
        config.dataset.n_channels = 1
        config.dataset.n_classes = 10

    # Dataset-dependent settings are resolved here so that selecting the dataset
    # via the CLI (e.g. `dataset.name breakfast`) updates these fields. They are
    # otherwise frozen at import time in defaults.py using the default dataset.
    if config.dataset.name == 'gtea':
        config.dataset.n_classes = 11
        config.dataset.sample_rate = 1
        config.dataset.roll = 4
        config.dataset.noise_weight = None
        config.dataset.num_query = 60
        config.dataset.guassian_sigma = 0.5
        config.dataset.pos_weight = 20
        config.dataset.threshold = 0.2
    elif config.dataset.name == '50salads':
        config.dataset.n_classes = 19
        config.dataset.sample_rate = 2
        config.dataset.roll = None
        config.dataset.noise_weight = None
        config.dataset.num_query = 100
        config.dataset.guassian_sigma = None
        config.dataset.pos_weight = 300
        config.dataset.threshold = 0.3
    elif config.dataset.name == 'breakfast':
        config.dataset.n_classes = 48
        config.dataset.sample_rate = 1
        config.dataset.roll = None
        config.dataset.noise_weight = None
        config.dataset.num_query = 100
        config.dataset.guassian_sigma = None
        config.dataset.pos_weight = 80
        config.dataset.threshold = 0.3

    if not torch.cuda.is_available():
        config.device = 'cpu'

    return config
