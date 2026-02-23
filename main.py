import argparse
import os
import yaml
import numpy as np
from engines.engine import Engine

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


def get_args_parser():
    parser = argparse.ArgumentParser('SAT-HMR', add_help=False)
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--mode', default='train', type=str)

    return parser


def update_args(args, cfg_path):
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
        args_dict = vars(args)
        args_dict.update(config)
        args = argparse.Namespace(**args_dict)
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SAT-HMR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    assert args.cfg is not None
    args = update_args(args, os.path.join('configs', 'run', f'{args.cfg}.yaml'))
    args.exp_name = args.cfg
    args = update_args(args, os.path.join('configs', 'models', f'{args.model}.yaml'))

    if args.mode.lower() == 'train':
        from accelerate.utils import set_seed
        seed = args.seed
        set_seed(args.seed)
        engine = Engine(args, mode='train')
        engine.train()

    elif args.mode.lower() == 'eval':
        engine = Engine(args, mode='eval')
        engine.eval()

    elif args.mode.lower() == 'infer':
        engine = Engine(args, mode='infer')
        engine.infer()

    else:
        print('Wrong mode!')
        exit(1)
