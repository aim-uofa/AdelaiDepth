import cv2
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from lib.test_utils import refine_focal, refine_shift
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt
from lib.spvcnn_classsification import SPVCNN_CLASSIFICATION
from lib.test_utils import reconstruct_depth

def parse_args():
    parser = argparse.ArgumentParser(
        description='Configs for LeReS')
    parser.add_argument('--load_ckpt', default='./res50.pth', help='Checkpoint path to load')
    parser.add_argument('--backbone', default='resnext101', help='Checkpoint path to load')

    args = parser.parse_args()
    return args

def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img

def make_shift_focallength_models():
    shift_model = SPVCNN_CLASSIFICATION(input_channel=3,
                                        num_classes=1,
                                        cr=1.0,
                                        pres=0.01,
                                        vres=0.01
                                        )
    focal_model = SPVCNN_CLASSIFICATION(input_channel=5,
                                        num_classes=1,
                                        cr=1.0,
                                        pres=0.01,
                                        vres=0.01
                                        )
    shift_model.eval()
    focal_model.eval()
    return shift_model, focal_model

def reconstruct3D_from_depth(rgb, pred_depth, shift_model, focal_model):
    cam_u0 = rgb.shape[1] / 2.0
    cam_v0 = rgb.shape[0] / 2.0
    pred_depth_norm = pred_depth - pred_depth.min() + 0.5

    dmax = np.percentile(pred_depth_norm, 98)
    pred_depth_norm = pred_depth_norm / dmax

    # proposed focal length, FOV is 60', Note that 60~80' are acceptable.
    proposed_scaled_focal = (rgb.shape[0] // 2 / np.tan((60/2.0)*np.pi/180))

    # recover focal
    focal_scale_1 = refine_focal(pred_depth_norm, proposed_scaled_focal, focal_model, u0=cam_u0, v0=cam_v0)
    predicted_focal_1 = proposed_scaled_focal / focal_scale_1.item()

    # recover shift
    shift_1 = refine_shift(pred_depth_norm, shift_model, predicted_focal_1, cam_u0, cam_v0)
    shift_1 = shift_1 if shift_1.item() < 0.6 else torch.tensor([0.6])
    depth_scale_1 = pred_depth_norm - shift_1.item()

    # recover focal
    focal_scale_2 = refine_focal(depth_scale_1, predicted_focal_1, focal_model, u0=cam_u0, v0=cam_v0)
    predicted_focal_2 = predicted_focal_1 / focal_scale_2.item()

    return shift_1, predicted_focal_2, depth_scale_1

if __name__ == '__main__':

    args = parse_args()

    # create depth model
    depth_model = RelDepthModel(backbone=args.backbone)
    depth_model.eval()

    # create shift and focal length model
    shift_model, focal_model = make_shift_focallength_models()

    # load checkpoint
    load_ckpt(args, depth_model, shift_model, focal_model)
    depth_model.cuda()
    shift_model.cuda()
    focal_model.cuda()

    image_dir = os.path.dirname(os.path.dirname(__file__)) + '/test_images/'
    imgs_list = os.listdir(image_dir)
    imgs_list.sort()
    imgs_path = [os.path.join(image_dir, i) for i in imgs_list if i != 'outputs']
    image_dir_out = image_dir + '/outputs'
    os.makedirs(image_dir_out, exist_ok=True)

    for i, v in enumerate(imgs_path):
        print('processing (%04d)-th image... %s' % (i, v))
        rgb = cv2.imread(v)
        rgb_c = rgb[:, :, ::-1].copy()
        gt_depth = None
        A_resize = cv2.resize(rgb_c, (448, 448))
        rgb_half = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_LINEAR)

        img_torch = scale_torch(A_resize)[None, :, :, :]
        pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
        pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

        # recover focal length, shift, and scale-invariant depth
        shift, focal_length, depth_scaleinv = reconstruct3D_from_depth(rgb, pred_depth_ori,
                                                                       shift_model, focal_model)
        disp = 1 / depth_scaleinv
        disp = (disp / disp.max() * 60000).astype(np.uint16)

        # if GT depth is available, uncomment the following part to recover the metric depth
        #pred_depth_metric = recover_metric_depth(pred_depth_ori, gt_depth)

        img_name = v.split('/')[-1]
        cv2.imwrite(os.path.join(image_dir_out, img_name), rgb)
        # save depth
        plt.imsave(os.path.join(image_dir_out, img_name[:-4]+'-depth.png'), pred_depth_ori, cmap='rainbow')
        cv2.imwrite(os.path.join(image_dir_out, img_name[:-4]+'-depth_raw.png'), (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))
        # save disp
        cv2.imwrite(os.path.join(image_dir_out, img_name[:-4]+'.png'), disp)

        # reconstruct point cloud from the depth
        reconstruct_depth(depth_scaleinv, rgb[:, :, ::-1], image_dir_out, img_name[:-4]+'-pcd', focal=focal_length)
