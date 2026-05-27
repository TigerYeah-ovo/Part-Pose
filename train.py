import os
import sys
import argparse
import logging
import random

import torch
import gorilla

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from provider.create_dataloaders import create_dataloaders
from utils.solver import Solver, get_logger
from model.Net import Net, Loss

def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation")

    # pretrain
    parser.add_argument("--gpus",
                        type=str,
                        default="1",
                        help="gpu num")
    parser.add_argument("--config",
                        type=str,
                        help="path to config file")
    args_cfg = parser.parse_args()

    return args_cfg

def init():
    args = get_parser()
    exp_name = args.config.split("/")[-1].split(".")[0]
    log_dir = os.path.join("/data/yecl24/ExperimentalResults/6dpe/part_registration/log", exp_name)
    
    os.makedirs(log_dir, exist_ok=True)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.log_dir = log_dir
    cfg.ckpt_dir = os.path.join(log_dir, 'ckpt')

    has_ckpt = False
    if not os.path.isdir(cfg.ckpt_dir):
        os.makedirs(cfg.ckpt_dir)
    else:
        ckpt_list = os.listdir(cfg.ckpt_dir)
        if len(ckpt_list) > 0:
            has_ckpt = True
        
    cfg.gpus = args.gpus
    logger = get_logger(
        level_print=logging.INFO, level_save=logging.WARNING, path_file=log_dir+"/training_logger.log")
    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)

    return logger, cfg, has_ckpt

if __name__ == "__main__":
    logger, cfg , has_ckpt= init()

    logger.warning(
        "************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed_all(cfg.rd_seed)

    # model
    logger.info("=> creating model ...")
    model = Net(cfg.pose_net)

    start_epoch = 1
    start_iter = 0

    model = model.cuda()
        
    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.warning("#Total parameters : {}".format(count_parameters))
    loss = Loss(cfg.loss).cuda()
    
    # dataloader
    dataloaders = create_dataloaders(cfg.train_dataset)

    for k in dataloaders.keys():
        dataloaders[k].dataset.reset()

    if has_ckpt:
        ckpt_list = os.listdir(cfg.ckpt_dir)
        ckpt_list = sorted(ckpt_list, key=lambda x:int("".join(filter(str.isdigit, x))))
        start_epoch = int(ckpt_list[-1].split(".")[0].split("_")[-1]) + 1
        print(f"Continual training {ckpt_list[-1]} from {start_epoch} epoch")
        ckpt_dict = torch.load(os.path.join(cfg.ckpt_dir, ckpt_list[-1]))
        model_state_dict = ckpt_dict["model_state_dict"]
        optim_state_dict = ckpt_dict["optim_state_dict"]
        model.load_state_dict(model_state_dict, strict=False)

    # solver
    Trainer = Solver(model=model, 
                     loss=loss,
                     dataloaders=dataloaders,
                     logger=logger,
                     cfg=cfg,
                     start_epoch=start_epoch,
                     start_iter=start_iter)
    
    if has_ckpt:
        Trainer.optimizer.load_state_dict(optim_state_dict)
    Trainer.solve()

    logger.info('\nFinish!\n')
