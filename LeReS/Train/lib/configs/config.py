from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from multiprocessing.sharedctypes import Value

import os
import six
import yaml
import copy
import numpy as np
from ast import literal_eval

from lib.utils.collections import AttrDict
from lib.utils.misc import get_run_name


__C = AttrDict()
# Consumers can get config by:
cfg = __C

# Root directory of project
__C.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------- #
# Data configurations
# ---------------------------------------------------------------------------- #
__C.DATASET = AttrDict()
__C.DATASET.NAME = 'diversedepth'
__C.DATASET.RGB_PIXEL_MEANS = (0.485, 0.456, 0.406)  # (102.9801, 115.9465, 122.7717)
__C.DATASET.RGB_PIXEL_VARS = (0.229, 0.224, 0.225)  # (1, 1, 1)
# Scale the depth map
#__C.DATASET.DEPTH_SCALE = 10.0
__C.DATASET.CROP_SIZE = (448, 448)  # (height, width)

# Camera Parameters
__C.DATASET.FOCAL_X = 256.0
__C.DATASET.FOCAL_Y = 256.0

# ---------------------------------------------------------------------------- #
# Models configurations
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()
__C.MODEL.INIT_TYPE = 'xavier'
# Configure the model type for the encoder, e.g.ResNeXt50_32x4d_body_stride16
__C.MODEL.ENCODER = 'resnet50_stride32'
__C.MODEL.MODEL_REPOSITORY = 'datasets/pretrained_model'
__C.MODEL.PRETRAINED_WEIGHTS = 'resnext50_32x4d.pth'
__C.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = True

# Configure resnext
__C.MODEL.RESNET_BOTTLENECK_DIM = [64, 256, 512, 1024, 2048]
__C.MODEL.RESNET_BLOCK_DIM = [64, 64, 128, 256, 512]

# Configure the decoder
__C.MODEL.FCN_DIM_IN = [512, 256, 256, 256, 256, 256]
__C.MODEL.FCN_DIM_OUT = [256, 256, 256, 256, 256]
__C.MODEL.LATERAL_OUT = [512, 256, 256, 256]

# Configure input and output channel of the model
__C.MODEL.ENCODER_INPUT_C = 3
__C.MODEL.DECODER_OUTPUT_C = 1
# Configure weight for different losses
__C.MODEL.FREEZE_BACKBONE_BN = False
#__C.MODEL.USE_SYNCBN = True
__C.MODEL.DEVICE = "cuda"
# ---------------------------------------------------------------------------- #
# Training configurations
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()
# Load run name, which is the combination of running time and host name
__C.TRAIN.RUN_NAME = get_run_name()
__C.TRAIN.OUTPUT_DIR = './outputs'
# Dir for checkpoint and logs
__C.TRAIN.LOG_DIR = os.path.join(__C.TRAIN.OUTPUT_DIR, cfg.TRAIN.RUN_NAME)
# Differ the learning rate between encoder and decoder
__C.TRAIN.SCALE_DECODER_LR = 1
__C.TRAIN.BASE_LR = 0.001
__C.TRAIN.MAX_ITER = 0
# Set training epoches, end at the last epoch of list
__C.TRAIN.EPOCH = 50
__C.TRAIN.MAX_ITER = 0
__C.TRAIN.LR_SCHEDULER_MULTISTEPS = [30000, 120000, 200000]
__C.TRAIN.LR_SCHEDULER_GAMMA = 0.1
__C.TRAIN.WARMUP_FACTOR = 1.0 / 3
__C.TRAIN.WARMUP_ITERS = 500  # Need to change
__C.TRAIN.WARMUP_METHOD = "linear"
# Snapshot (model checkpoint) period
__C.TRAIN.SNAPSHOT_ITERS = 5000
__C.TRAIN.VAL_STEP = 5000
__C.TRAIN.BATCHSIZE = 4
__C.TRAIN.GPU_NUM = 1
# Steps for LOG interval
__C.TRAIN.LOG_INTERVAL = 10
__C.TRAIN.LOAD_CKPT = None
# Configs for loss
__C.TRAIN.LOSS_MODE = '_vnl_ssil_ranking_'
__C.TRAIN.LOSS_AUXI_WEIGHT = 0.5
#__C.TRAIN.DIFF_LOSS_WEIGHT = 5

__C.TRAIN.OPTIM = 'SGD'

def print_configs(cfg):
    import logging
    logger = logging.getLogger(__name__)

    message = ''
    message += '----------------- Configs ---------------\n'
    for k, kv in cfg.items():
        if isinstance(kv, AttrDict):
            name1 = '.'.join(['cfg', k])
            message += '{:>50}: \n'.format(name1)
            for k_, v_ in kv.items():
                name2 = '.'.join([name1, k_])
                message += '{:>50}: {:<30}\n'.format(str(name2), str(v_))
        else:
            message += '{:>50}: {:<30}\n'.format(k, kv)

    message += '----------------- End -------------------'
    logger.info(message)


def merge_cfg_from_file(train_args):
    """Load a yaml config file and merge it into the global config."""
    # cfg_filename = train_args.cfg_file
    # cfg_file = os.path.join(__C.ROOT_DIR, cfg_filename + '.yaml')
    # with open(cfg_file, 'r') as f:
    #     yaml_cfg = AttrDict(yaml.safe_load(f))
    # _merge_a_into_b(yaml_cfg, __C)
    # __C.DATASET.DEPTH_MIN_LOG = np.log10(__C.DATASET.DEPTH_MIN)
    # # Modify some configs
    # __C.DATASET.DEPTH_BIN_INTERVAL = (np.log10(__C.DATASET.DEPTH_MAX) - np.log10(
    #     __C.DATASET.DEPTH_MIN)) / __C.MODEL.DECODER_OUTPUT_C

    # # The boundary of each bin
    # __C.DATASET.DEPTH_BIN_BORDER = np.array([__C.DATASET.DEPTH_MIN_LOG + __C.DATASET.DEPTH_BIN_INTERVAL * (i + 0.5)
    #                                          for i in range(__C.MODEL.DECODER_OUTPUT_C)])
    # __C.DATASET.WCE_LOSS_WEIGHT = [[np.exp(-0.2 * (i - j) ** 2) for i in range(__C.MODEL.DECODER_OUTPUT_C)]
    #                                for j in np.arange(__C.MODEL.DECODER_OUTPUT_C)]

    if train_args.backbone == 'resnet50':
        __C.MODEL.ENCODER = 'resnet50_stride32'
        __C.MODEL.PRETRAINED_WEIGHTS = 'resnext50_32x4d.pth'
    elif train_args.backbone == 'resnext101':
        __C.MODEL.ENCODER = 'resnext101_stride32x8d'
        __C.MODEL.PRETRAINED_WEIGHTS = 'resnext101_stride32x8d.pth'
    else:
        raise ValueError

    for k, v in vars(train_args).items():
        if k.upper() in __C.TRAIN.keys():
            __C.TRAIN[k.upper()] = getattr(train_args, k)
    
    __C.TRAIN.LOG_DIR = os.path.join(__C.TRAIN.OUTPUT_DIR, cfg.TRAIN.RUN_NAME)


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            # if _key_is_deprecated(full_key):
            #     continue
            # elif _key_is_renamed(full_key):
            #     _raise_key_rename_error(full_key)
            # else:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, six.string_types):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, six.string_types):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
