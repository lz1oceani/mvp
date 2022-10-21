import os, os.path as osp
from isaacgym import gymapi
from isaacgym import gymutil

import hydra
import omegaconf
import torch
import time

from mvp.utils.hydra_utils import omegaconf_to_dict, print_dict, dump_cfg
from mvp.utils.hydra_utils import set_np_formatting, set_seed
from mvp.utils.hydra_utils import parse_sim_params, parse_task
import matplotlib.pyplot as plt


@hydra.main(config_name="config", config_path="../configs")
def train(cfg: omegaconf.DictConfig):

    # Assume no multi-gpu training
    assert cfg.num_gpus == 1

    # Parse the config
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # Create logdir and dump cfg
    if not cfg.test:
        os.makedirs(cfg.logdir, exist_ok=True)
        dump_cfg(cfg, cfg.logdir)

    # Set up python env
    set_np_formatting()
    set_seed(cfg.train.seed, cfg.train.torch_deterministic)

    # Construct task
    sim_params = parse_sim_params(cfg, cfg_dict)
    env = parse_task(cfg, cfg_dict, sim_params)
    
    obs = env.reset()
    print(obs.shape)
    num_envs = len(obs)
    for i in range(2):
        env.step(torch.rand((num_envs,) + env.action_space.shape, device="cuda:0"))
    torch.cuda.synchronize()
    
    st = time.time()
    num_envs = len(obs)
    num = 10
    for i in range(num):
        env.step(torch.rand((num_envs,) + env.action_space.shape, device="cuda:0"))
    torch.cuda.synchronize()
    total_time = time.time() - st
    print("FPS", num * num_envs / total_time, 'time', total_time)
    
        
    
if __name__ == '__main__':
    conda_path = osp.abspath("~/miniconda3/envs/rlgpu/lib")
    print(conda_path)
    
    os.environ["LD_LIBRARY_PATH"] = conda_path
    
    train()
