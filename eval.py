import json
import os
import numpy as np
import random
import subprocess
from datetime import datetime
import logging
import sys,signal
from torch.utils import data
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import datetime
import time
import dill

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import models, utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from dataset import ImageFolderDataset, VideoFolderDataset, VideoFolderDatasetRestricted, VideoFolderDatasetCachedForRecons, VideoFolderDatasetSplit, get_dataloader, VideoFolderDatasetSplitFixedSample, random_split_dataset, get_train_dataloader, get_test_dataloader
from models.models import MMDetWUniModal
from utils.torch_utils import eval_model,display_eval_tb,train_logging,get_lr_blocks,associate_param_with_lr,lrSched_monitor, ContrastiveLoss, set_seed
from utils.runjobs_utils import init_logger,Saver,DataConfig,torch_load_model,get_iter,get_data_to_copy_str
from options.test_options import TestOption

def get_val_transformation_cfg():
    cfg = {
        # 'resize': {
        #     'img_size': 224
        # },
        'post': {
            'blur': {
                'prob': 0.0,
                'sig': [0.0, 3.0]
            },
            'jpeg': {
                'prob': 0.0,
                'method': ['cv2', 'pil'],
                'qual': [30, 100]
            },
            'noise':{
                'prob': 0.0,
                'var': [0.01]
            }
        },
        'crop': {
            'img_size': 224,
            'type': 'center'    # ['center', 'random'], according to 'train' or 'test' mode
        },
        'flip': False,    # set false when testing
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    return cfg


def get_logger(name, config):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.flush = sys.stdout.flush
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)
    os.makedirs(os.path.join(config['out_dir'], config['expt']), exist_ok=True)
    out_handler = logging.FileHandler(filename=os.path.join(config['out_dir'], config['expt'], f'{config["mode"]}.log'))
    out_handler.setLevel(level=logging.INFO)
    logger.addHandler(out_handler)
    logger.setLevel(logging.INFO)
    return logger


def get_val_dataset_config(config):
    dataset_classes = config['classes']
    dataset_config = {}
    for dataset_class in dataset_classes:
        dataset_config[dataset_class] = {
            "data_root": f'{config["data_root"]}/{dataset_class}',
            "dataset_type": "VideoFolderDatasetWithFn",
            "mode": config["mode"],
            "selected_cls_labels": [("0_real", 0), ("1_fake", 1)]
        }
    return dataset_config


def test(config):
    starting_time = datetime.datetime.now()
    out_dir = args.out_dir
    exp_name = args.expt
    os.makedirs(os.path.join(out_dir, exp_name), exist_ok=True)
    logger = get_logger(__name__, config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['st_pretrained'] = False
    model = MMDetWUniModal(config)
    model = torch.nn.DataParallel(model).to(device)
    val_dataset_cfgs = get_val_dataset_config(config)
    
    if config['ckpt_path'] is not None and os.path.exists(config['ckpt_path']):
        load_model_path = config['ckpt_path']
    else:
        resume_ckpt_name = config['resume_ckpt_name']
        load_model_path = os.path.join(out_dir, exp_name, 'checkpoints', resume_ckpt_name)
    if not os.path.exists(load_model_path):
        raise ValueError(f'Checkpoint path not found in {load_model_path}.')
    else:
        logger.info(f'Loading weights from {load_model_path}...')
        with open(load_model_path, 'rb') as f:
            loaded_file = dill.load(f)
            model.load_state_dict(loaded_file['model_state_dict'], strict=False)

    window_size = config['window_size']
    model.eval()
    for dataset_id, (val_dataset_name, val_dataset_cfg) in enumerate(val_dataset_cfgs.items(), 1):
        print(f'Test on {dataset_id}/{len(val_dataset_cfgs)}: {val_dataset_name}')
        cur_dataset_config = {
            val_dataset_name: val_dataset_cfg
        }
        val_dataloader = get_test_dataloader(cur_dataset_config)
        with torch.no_grad():
            clip_val_probas = []
            clip_val_gts = []
            for idx, val_batch in tqdm(enumerate(val_dataloader, 1), total=len(val_dataloader), desc=f'Val {val_dataset_name}'):
                fns, val_data_batch, val_true_labels = val_batch
                val_true_labels = val_true_labels.long().to(device)
                B, L, C, H, W = val_data_batch.shape
                # use dense clips
                for index in range(L - window_size + 1):
                    if index % window_size == 0 or index == L - window_size:
                        clip = val_data_batch[:, index: index + window_size, :, :, :]
                        val_preds = model(clip)
                        clip_log_probs = F.softmax(val_preds, dim=-1)
                        clip_res = torch.argmax(clip_log_probs, dim=-1)
                        clip_samples = clip_res.shape[0]
                        
                        clip_val_probas.extend(clip_log_probs[:,0].tolist())
                        clip_fixed_labels = 1 - val_true_labels
                        clip_val_gts.extend(clip_fixed_labels[:].tolist())
            val_auc = roc_auc_score(clip_val_gts, clip_val_probas)
            logger.info(f'Val auc on {val_dataset_name}: {val_auc}')
        

if __name__ == '__main__':
    args = TestOption().parse()
    config = args.__dict__
    set_seed(args.seed)
    test(config)
