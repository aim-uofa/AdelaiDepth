import os
import cv2
import json
import scipy.io as sio
import numpy as np
import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

from lib.utils.logging import setup_logging, SmoothedValue
from lib.models.multi_depth_model_auxiv2 import RelDepthModel
from lib.utils.net_tools import load_ckpt
from lib.utils.evaluate_depth_error import evaluate_rel_err, recover_metric_depth
from lib.configs.config import cfg, merge_cfg_from_file
from tools.parse_arg_test import TestOptions


logger = setup_logging(__name__)

def scale_torch(img, scale):
    """
    Scale the image and output it in torch.tensor.
    :param img: input image. [C, H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32)
    img /= scale
    img = torch.from_numpy(img.copy())
    img = transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)(img)
    return img


if __name__ == '__main__':
    test_args = TestOptions().parse()
    test_args.thread = 1
    test_args.batchsize = 1

    # load model
    model = RelDepthModel()
    model.eval()

    # load checkpoint
    if test_args.load_ckpt:
        load_ckpt(test_args, model)
    model.cuda()
    model = torch.nn.DataParallel(model)

    # base_dir = '/home/yvan/DeepLearning/all-datasets'
    # annos_path = os.path.join(base_dir, test_args.dataset_list[0], 'annotations/test_annotations.json')
    # f = open(annos_path)
    # annos = json.load(f)
    # f.close()

    imgs = sio.loadmat('../datasets/test.mat')
    rgbs = imgs['rgbs']
    depths = imgs['depths']

    test_datasize = rgbs.shape[0]
    logger.info('{:>15}: {:<30}'.format('test_data_size', test_datasize))
    # test
    smoothed_absRel = SmoothedValue(test_datasize)
    smoothed_rms = SmoothedValue(test_datasize)
    smoothed_logRms = SmoothedValue(test_datasize)
    smoothed_squaRel = SmoothedValue(test_datasize)
    smoothed_silog = SmoothedValue(test_datasize)
    smoothed_silog2 = SmoothedValue(test_datasize)
    smoothed_log10 = SmoothedValue(test_datasize)
    smoothed_delta1 = SmoothedValue(test_datasize)
    smoothed_delta2 = SmoothedValue(test_datasize)
    smoothed_delta3 = SmoothedValue(test_datasize)
    smoothed_whdr = SmoothedValue(test_datasize)

    smoothed_criteria = {'err_absRel': smoothed_absRel, 'err_squaRel': smoothed_squaRel, 'err_rms': smoothed_rms,
                         'err_silog': smoothed_silog, 'err_logRms': smoothed_logRms, 'err_silog2': smoothed_silog2,
                         'err_delta1': smoothed_delta1, 'err_delta2': smoothed_delta2, 'err_delta3': smoothed_delta3,
                         'err_log10': smoothed_log10, 'err_whdr': smoothed_whdr}

    for i in range(test_datasize):
        if i % 100 == 0:
            logger.info('processing : ' + str(i) + ' / ' + str(test_datasize))
        rgb = rgbs[i].transpose((2, 1, 0))  #rgb
        depth = depths[i].transpose((1, 0))
        mask_invalid = depth < 1e-8
        mask_invalid[45:471, 41:601] = 1
        mask_invalid = mask_invalid.astype(np.bool)

        # resize input to [385, 385], same to training setting
        rgb_resize = cv2.resize(rgb, (448, 448))

        img_torch = scale_torch(rgb_resize, 255)
        img_torch = img_torch[None, :, :, :].cuda()
        with torch.no_grad():
            pred_depth, pred_disp = model.module.depth_model(img_torch)
        pred_depth_resize = cv2.resize(pred_depth.cpu().numpy().squeeze(), (rgb.shape[1], rgb.shape[0]))

        # Recover metric depth
        pred_depth_metric = recover_metric_depth(pred_depth_resize, depth)
        # evaluate
        smoothed_criteria = evaluate_rel_err(pred_depth_metric, depth, smoothed_criteria)

        model_name = test_args.load_ckpt.split('/')[-1].split('.')[0]
        image_dir = os.path.join(cfg.ROOT_DIR, './evaluation', cfg.MODEL.ENCODER, model_name + '_nyu')
        os.makedirs(image_dir, exist_ok=True)
        img_name = '%04d.png' %i

        plt.imsave(os.path.join(image_dir, img_name.replace('.png', '_pred.png')), pred_depth_metric, cmap='rainbow')
        cv2.imwrite(os.path.join(image_dir, img_name.replace('.png', '_rgb.png')), np.squeeze(rgb)[:, :, ::-1])
        plt.imsave(os.path.join(image_dir, img_name.replace('.png', '_gt.png')), np.squeeze(depth), cmap='rainbow')
        # cv2.imwrite(os.path.join(image_dir, img_name.replace('.png', '_gtraw.png')), (pred_depth_metric * 6000).astype(np.uint16))



    print("###############WHDR ERROR: %f", smoothed_criteria['err_whdr'].GetGlobalAverageValue())
    print("###############absREL ERROR: %f", smoothed_criteria['err_absRel'].GetGlobalAverageValue())
    print("###############silog ERROR: %f", np.sqrt(smoothed_criteria['err_silog2'].GetGlobalAverageValue() - (
        smoothed_criteria['err_silog'].GetGlobalAverageValue()) ** 2))
    print("###############log10 ERROR: %f", smoothed_criteria['err_log10'].GetGlobalAverageValue())
    print("###############RMS ERROR: %f", np.sqrt(smoothed_criteria['err_rms'].GetGlobalAverageValue()))
    print("###############delta_1 ERROR: %f", smoothed_criteria['err_delta1'].GetGlobalAverageValue())
    print("###############delta_2 ERROR: %f", smoothed_criteria['err_delta2'].GetGlobalAverageValue())
    print("###############delta_3 ERROR: %f", smoothed_criteria['err_delta3'].GetGlobalAverageValue())
    print("###############squaRel ERROR: %f", smoothed_criteria['err_squaRel'].GetGlobalAverageValue())
    print("###############logRms ERROR: %f", np.sqrt(smoothed_criteria['err_logRms'].GetGlobalAverageValue()))