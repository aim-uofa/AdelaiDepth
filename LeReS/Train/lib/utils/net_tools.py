import importlib
import os
import dill
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import OrderedDict

from lib.configs.config import cfg

logger = logging.getLogger(__name__)

def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'lib.models.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to f1ind function: %s', func_name)
        raise

def load_ckpt(args, model, optimizer=None, scheduler=None, val_err=[]):
    """
    Load checkpoint.
    """
    if os.path.isfile(args.load_ckpt):
        logger.info("loading checkpoint %s", args.load_ckpt)
        checkpoint = torch.load(args.load_ckpt, map_location=lambda storage, loc: storage, pickle_module=dill)
        model_state_dict_keys = model.state_dict().keys()
        checkpoint_state_dict_noprefix = strip_prefix_if_present(checkpoint['model_state_dict'], "module.")

        if all(key.startswith('module.') for key in model_state_dict_keys):
            model.module.load_state_dict(checkpoint_state_dict_noprefix)
        else:
            model.load_state_dict(checkpoint_state_dict_noprefix)
        if args.resume:
            #args.batchsize = checkpoint['batch_size']
            args.start_step = checkpoint['step']
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scheduler.__setattr__('last_epoch', checkpoint['step'])
            if 'val_err' in checkpoint:  # For backward compatibility
                val_err[0] = checkpoint['val_err']
        del checkpoint
        torch.cuda.empty_cache()


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def save_ckpt(args, step, epoch, model, optimizer, scheduler, val_err={}):
    """Save checkpoint"""
    ckpt_dir = os.path.join(cfg.TRAIN.LOG_DIR, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'epoch%d_step%d.pth' %(epoch, step))
    if isinstance(model, nn.DataParallel):
        model = model.module
    torch.save({
        'step': step,
        'epoch': epoch,
        'batch_size': args.batchsize,
        'scheduler': scheduler.state_dict(),
        'val_err': val_err,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()},
        save_name, pickle_module=dill)
    logger.info('save model: %s', save_name)


# save image to the disk
def save_images(data, pred, scale=60000.):
    rgb = data['A_raw']
    gt = data['B_raw']
    if type(rgb).__module__ != np.__name__:
        rgb = rgb.cpu().numpy()
        rgb = np.squeeze(rgb)
        rgb = rgb[:, :, ::-1]
    if type(gt).__module__ != np.__name__:
        gt = gt.cpu().numpy()
        gt = np.squeeze(gt)
    if type(pred).__module__ != np.__name__:
        pred = pred.cpu().numpy()
        pred = np.squeeze(pred)
    model_name = (cfg.DATA.LOAD_MODEL_NAME.split('/')[-1]).split('.')[0]
    image_dir = os.path.join(cfg.TRAIN.OUTPUT_ROOT_DIR, '../evaluation', model_name)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)


    if 'kitti' in cfg.DATASET:
        name = data['A_paths'][0].split('/')[-4] + '-' + data['A_paths'][0].split('/')[-1].split('.')[0]
    else:
        name = data['A_paths'][0].split('/')[-1].split('.')[0]
    rgb_name = '%s_%s.png' % (name, 'rgb')
    gt_name = '%s_%s.png' % (name, 'gt')
    gt_raw_name = '%s_%s.png' % (name, 'gt-raw')
    pred_name = '%s_%s.png' % (name, 'pred')
    pred_raw_name = '%s_%s.png' % (name, 'pred-raw')

    plt.imsave(os.path.join(image_dir, rgb_name), rgb)
    if len(data['B_raw'].shape) != 2:
        plt.imsave(os.path.join(image_dir, gt_name), gt, cmap='rainbow')
        gt_scale = gt * scale
        gt_scale = gt_scale.astype('uint16')
        cv2.imwrite(os.path.join(image_dir, gt_raw_name), gt_scale)
    plt.imsave(os.path.join(image_dir, pred_name), pred, cmap='rainbow')
    pred_raw = pred * scale
    pred_raw = pred_raw.astype('uint16')
    cv2.imwrite(os.path.join(image_dir, pred_raw_name), pred_raw)