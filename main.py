import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from PIL import Image

import yaml
import argparse
import datetime

from intermdetr.matcher import build_matcher

from intermdetr.intermdetr import build_intermdetr


from ..unimatch.unimatch.unimatch import UniMatch

from helpers.dataloader_helper import build_dataloader
from helpers.optimizer_helper import build_optimizer
from helpers.scheduler_helper import build_lr_scheduler
from helpers.trainer_helper import Trainer
from helpers.tester_helper import Tester
from helpers.utils_helper import create_logger
from helpers.utils_helper import set_random_seed


from datasets.kitti.kitti_eval_python.eval import get_official_eval_result
from datasets.kitti.kitti_eval_python.eval import get_distance_eval_result
# import datasets.kitti.kitti_eval_python.kitti_common as kitti

from torch.utils.data import DataLoader
from datasets.kitti.kitti_dataset import KITTI_Dataset
import copy


if __name__ == '__main__':

    # cfg = {'root_dir': '/home/autonomy/stereo_camera/data',
    #        'random_flip': 0.0, 'random_crop': 1.0, 'scale': 0.8, 'shift': 0.1, 'use_dontcare': False,
    #        'class_merging': False, 'writelist':['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center':False,

    #        }
    cfg = {
        'random_seed': 444, 'dataset': {'type': 'KITTI', 'root_dir': '/home/autonomy/stereo_camera/data', 'train_split': 'train',
        'test_split': 'val', 'batch_size': 16, 'use_3d_center': True, 'class_merging': False,
        'use_dontcare': False, 'bbox2d_type': 'anno', 'meanshape': False, 'writelist': ['Pedestrian', 'Car', 'Cyclist'], 'clip_2d': False,
        'aug_pd': False, 'aug_crop': False, 'random_flip': 0.0, 'random_crop': 0.0, 'scale': 0.00, 'shift': 0.0, 'intermediate_layer': 'layer_5'}, 
        'model_name': 'unidetr_V1', 'model': {'num_classes': 3, 'return_intermediate_dec': True, 'device': 'cuda', 'backbone': 'resnet50', 
        'train_backbone': True, 'num_feature_levels': 1, 'dilation': False, 'position_embedding': 'sine', 'masks': False, 'mode': 'LID',
        'num_depth_bins': 80, 'depth_min': '1e-3', 'depth_max': 60.0, 'with_box_refine': True, 'two_stage': False, 'init_box': False, 
        'enc_layers': 1, 'dec_layers': 3, 'hidden_dim': 256, 'dim_feedforward': 256, 'dropout': 0.1, 'nheads': 8, 'num_queries': 50, 
        'enc_n_points': 4, 'dec_n_points': 4, 'aux_loss': False, 'cls_loss_coef': 2, 'focal_alpha': 0.25, 'bbox_loss_coef': 5, 
        'giou_loss_coef': 2, '3dcenter_loss_coef': 10, 'dim_loss_coef': 1, 'angle_loss_coef': 1, 'depth_loss_coef': 1, 
        'depth_map_loss_coef': 1, 'set_cost_class': 2, 'set_cost_bbox': 5, 'set_cost_giou': 2, 'set_cost_3dcenter': 10}, 
        'optimizer': {'type': 'adamw', 'lr': 0.00005, 'weight_decay': 0.00005}, 
        'lr_scheduler': {'type': 'step', 'warmup': False, 'decay_rate': 0.1, 'decay_list': [125, 165]}, 
        'trainer': {'max_epoch': 195, 'gpu_ids': '0', 'save_frequency': 1, 'save_path': './runs/', 'save_all': False},
        'tester': {'type': 'KITTI', 'mode': 'single', 'checkpoint': 195, 'threshold': 0.2, 'topk': 50}
        }

            #### INTERNAL CONFIG FILE ####

    dataset = KITTI_Dataset('train', cfg['dataset'])


    dataloader = DataLoader(dataset=dataset, batch_size=8)

    # config = "./config/params.yaml"
    # assert (os.path.exists(config))
    # cfg = yaml.load(open(config, 'r'), Loader=yaml.Loader)

    set_random_seed(cfg.get('random_seed', 444))

    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    # build dataloader

    train_loader, test_loader = build_dataloader(cfg['dataset'])###########



    detr_model, loss = build_intermdetr(cfg['model'])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))
    print(device)

    if cfg['dataset']['intermediate_layer'] != '':

        feature_channels = 128
        num_scales = 1
        upsample_factor = 8
        num_head = 1
        ffn_dim_expansion = 4
        num_transformer_layers = 6
        task = 'stereo'
        attn_type = "self_swin2d_cross_1d"
        attn_splits_list = [1]
        corr_radius_list = [-1]
        prop_radius_list = [-1]
        num_reg_refine = 1

        stereo_model = UniMatch(feature_channels=feature_channels,
                            num_scales=num_scales,
                            upsample_factor=upsample_factor,
                            num_head=num_head,
                            ffn_dim_expansion=ffn_dim_expansion,
                            num_transformer_layers=num_transformer_layers,
                            attn_type=attn_type,
                            attn_splits_list=attn_splits_list,
                            corr_radius_list=corr_radius_list,
                            prop_radius_list=prop_radius_list,
                            num_reg_refine=num_reg_refine,
                            task=task).to(device)
        

    if len(gpu_ids) == 1:
        detr_model = detr_model.to(device)
    else:
        detr_model = torch.nn.DataParallel(detr_model, device_ids=gpu_ids).to(device)

    if False:
        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        train_cfg=cfg['trainer'],
                        model_name=model_name)
        
        tester.test()

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], detr_model)

    # build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    trainer = Trainer(cfg=cfg['trainer'],
                      stereomodel=stereo_model,
                      model=detr_model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      loss=loss,
                      model_name=model_name)

    tester = Tester(cfg=cfg['tester'],
                    model=trainer.model,
                    dataloader=test_loader,
                    logger=logger,
                    train_cfg=cfg['trainer'],
                    model_name=model_name)
    if cfg['dataset']['test_split'] != 'test':
        trainer.tester = tester

    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f' % (cfg['optimizer']['lr']))

    trainer.train()

    if cfg['dataset']['test_split'] == 'test':
        tester.test()

    logger.info('###################  Testing  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Split: %s' % (cfg['dataset']['test_split']))

    tester.test()
