
from . import yacs
from .yacs import CfgNode as CN
import argparse
import os
import numpy as np
import pprint

cfg = CN()



def parse_cfg(cfg, opt):
    cfg.local_rank = opt.local_rank
    cfg.test_mode=opt.test
    cfg.seed=opt.seed
    cfg.iters=opt.iters
    cfg.ckpt=opt.ckpt
    cfg.test_view1=opt.test_view1
    cfg.test_view2=opt.test_view2


def make_cfg(opt):
    with open(opt.cfg_file, 'r') as f:
        current_cfg = yacs.load_cfg(f)

    if 'parent_cfg' in current_cfg.keys():
        with open(current_cfg.parent_cfg, 'r') as f:
            parent_cfg = yacs.load_cfg(f)
        cfg.merge_from_other_cfg(parent_cfg)

    cfg.merge_from_other_cfg(current_cfg)

    parse_cfg(cfg, opt)
    # pprint.pprint(cfg)
    return cfg

parser = argparse.ArgumentParser()
# parser.add_argument('path', type=str)

parser.add_argument('--test', action='store_true', help="test mode")

parser.add_argument('--test_view1', type=int, default=-1)

parser.add_argument('--test_view2', type=int, default=-1)

parser.add_argument('--seed', type=int, default=0)

### training options
parser.add_argument('--iters', type=int, default=10000, help="training iters")


parser.add_argument('--ckpt', type=str, default='latest')







### dataset options
parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
# (the default value is for the fox dataset)



parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)

parser.add_argument('--local_rank', type=int, default=0)

opt = parser.parse_args()

cfg = make_cfg(opt)
