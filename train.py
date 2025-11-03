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
from options.train_options import TrainOption


def get_train_transformation_cfg():
    cfg = {
        # 'resize': {
        #     'img_size': 224
        # },
        'post': {
            'blur': {
                'prob': 0.1,
                'sig': [0.0, 3.0]
            },
            'jpeg': {
                'prob': 0.1,
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
            'type': 'random'    # ['center', 'random'], according to 'train' or 'test' mode
        },
        'flip': True,    # set false when testing
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    return cfg


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

def get_train_dataset_config(config):
    dataset_classes = config['classes']
    dataset_config = {}
    for dataset_class in dataset_classes:
        dataset_config[dataset_class] = {
            "data_root": f'{config["data_root"]}/{dataset_class}',
            "dataset_type": "VideoFolderDatasetWithFn",
            "mode": config["mode"],
            "selected_cls_labels": [("0_real", 0), ("1_fake", 1)]
        }
        if config['fix_split'] and config["mode"] == 'train':
            dataset_config[dataset_class]['split'] = {
                'train': json.load(open(os.path.join(config['split_path'], 'train.json')))[dataset_class],
                'val': json.load(open(os.path.join(config['split_path'], 'val.json')))[dataset_class]
            }
    return dataset_config


def create_optimizer(config, model):
    lr = config['lr']
    param_dict_list = model.module.assign_lr_dict_list(lr=lr)
    optimizer = torch.optim.Adam(param_dict_list, weight_decay=config['weight_decay'])
    return optimizer
    
    
def create_scheduler(config, optimizer):
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['step_factor'], min_lr=1e-08, patience=config['patience'], cooldown=config['cooldown'], verbose=True)
    return lr_scheduler


def train(config):
    starting_time = datetime.datetime.now()
    out_dir = args.out_dir
    exp_name = args.expt
    os.makedirs(os.path.join(out_dir, exp_name), exist_ok=True)
    logger = get_logger(__name__, config)
    config['datasets'] = get_train_dataset_config(config)
    train_dataloader, val_dataloader = get_train_dataloader(config)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = MMDetWUniModal(config)
    model = torch.nn.DataParallel(model).to(device)
    optimizer = create_optimizer(config, model)
    lr_scheduler = create_scheduler(config, optimizer)
    ce_loss = nn.CrossEntropyLoss()
    contra_loss = ContrastiveLoss(batch_size=config['bs'], temperature=config['contra_temperature'])
    
    epoch_init = 0
    resume_ckpt_name = config['resume_ckpt_name']
    load_model_path = os.path.join(out_dir, exp_name, 'checkpoints', resume_ckpt_name)
    val_loss = np.inf
    if os.path.exists(load_model_path):
        logger.info(f'Loading weights, optimizer and scheduler from {load_model_path}...')
        it_off, epoch_init, lr_scheduler, val_loss = torch_load_model(model, optimizer, load_model_path, strict=False)
    data_config = DataConfig(os.path.join(out_dir, exp_name, 'checkpoints'), exp_name)
    saver = Saver(model, optimizer, lr_scheduler, data_config, starting_time, hours_limit=23, mins_limit=0)
    
    tb_folder = os.path.join(os.path.join(out_dir, exp_name), 'tb_logs')
    writer = SummaryWriter(tb_folder)
    log_string_config = '  '.join([k + ':' + str(v) for k,v in config.items()])
    writer.add_text('config : %s' % exp_name, log_string_config, 0)
    
    if epoch_init == 0:
        model.zero_grad()

    tot_iter = 0
    window_size = config['window_size']
    patience = config['patience']
    for epoch in range(epoch_init, config['epoch']):
        logger.info(f"Epoch: {epoch}, learning_rate: {optimizer.param_groups[0]['lr']}")
        train_loss = 0
        total_loss = 0
        accu_contra_loss = 0
        accu_ce_loss = 0
        total_accu = 0
        for ib, (fns, data_batch, true_labels) in enumerate(train_dataloader, 1):
            model.train()
            data_batch = data_batch.float().to(device)
            true_labels = true_labels.long().to(device)
            optimizer.zero_grad()
            pred_labels, text_feat, vision_feat = model(data_batch, output_mm_states=True)
            loss_ce = ce_loss(pred_labels, true_labels)
            loss_contrastive = contra_loss(text_feat.squeeze(1), vision_feat.squeeze(1))
            total_loss += loss_ce.item()
            total_loss += loss_contrastive.item()
            accu_contra_loss += loss_contrastive.item()
            accu_ce_loss += loss_ce.item()
            loss = loss_ce + loss_contrastive
            log_probs = F.softmax(pred_labels, dim=-1)
            res_probs = torch.argmax(log_probs, dim=-1)
            summation = torch.sum(res_probs == true_labels)
            accu = summation / true_labels.shape[0]
            total_accu += accu
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if tot_iter % config['display_step'] == 0:
                train_logging(
                    'loss/train_loss_iter', writer, logger, epoch, saver, 
                    tot_iter, total_loss/config['display_step'], 
                    total_accu/config['display_step'], lr_scheduler
                )
                total_loss = 0
                total_accu = 0
                logging_dict = {
                    "contra_loss": accu_contra_loss / config['display_step'],
                    "cross_loss": accu_ce_loss / config['display_step']
                }
                for k in logging_dict:
                    logger.info(f'{k}: {logging_dict[k]}')
                accu_contra_loss = 0
                accu_ce_loss = 0
            if (tot_iter + 1) % config['val_step'] == 0:
                model.eval()
                with torch.no_grad():
                    clip_val_probas = []
                    clip_val_gts = []
                    clip_val_losses = []
                    for idx, val_batch in tqdm(enumerate(val_dataloader, 1), total=len(val_dataloader), desc='validation'):
                        fns, val_data_batch, val_true_labels = val_batch
                        val_true_labels = val_true_labels.long().to(device)
                        B, L, C, H, W = val_data_batch.shape
                        # use dense clips
                        for index in range(L - window_size + 1):
                            if index % window_size == 0 or index == L - window_size:
                                clip = val_data_batch[:, index: index + window_size, :, :, :]
                                val_preds = model(clip)
                                clip_val_loss = ce_loss(val_preds, val_true_labels)
                                clip_log_probs = F.softmax(val_preds, dim=-1)
                                clip_res = torch.argmax(clip_log_probs, dim=-1)
                                clip_samples = clip_res.shape[0]
                                
                                clip_val_probas.extend(clip_log_probs[:,0].tolist())
                                clip_fixed_labels = 1 - val_true_labels
                                clip_val_gts.extend(clip_fixed_labels[:].tolist())
                                clip_val_losses.append(clip_val_loss.item())
                    val_auc = roc_auc_score(clip_val_gts, clip_val_probas)
                    avg_val_losses = sum(clip_val_losses) / len(clip_val_losses)
                    lr_scheduler.step(avg_val_losses)
                    writer.add_scalar('loss/val_loss_iter', avg_val_losses, tot_iter)
                    logger.info(f'Average val loss: {avg_val_losses}')
                    logger.info(f'Average val auc: {val_auc}')
                    logger.info(f'Patience: {lr_scheduler.num_bad_epochs} / {patience}')
                model.train()
            tot_iter += 1
        if epoch % config['save_epoch'] == 0:
            saver.save_model(epoch, tot_iter, total_loss, 0, force_saving=True)
        

if __name__ == '__main__':
    args = TrainOption().parse()
    config = args.__dict__
    set_seed(args.seed)
    train(config)
