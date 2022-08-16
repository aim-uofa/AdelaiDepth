import os
import os.path
import json
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from imgaug import augmenters as iaa

from lib.configs.config import cfg


class MultiDataset(Dataset):
    def __init__(self, opt, dataset_name=None):
        super(MultiDataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        self.dataset_name = dataset_name
        self.dir_anno = os.path.join(cfg.ROOT_DIR,
                                     opt.dataroot,
                                     dataset_name,
                                     'annotations',
                                     opt.phase_anno + '_annotations.json')
        self.dir_teacher_list = None
        self.rgb_paths, self.depth_paths, self.disp_paths, self.sem_masks, self.ins_paths, self.cam_intrinsics, self.all_annos, self.curriculum_list = self.getData()
        self.data_size = len(self.all_annos)
        self.focal_length_dict = {'diml_ganet': 1380.0 / 2.0, 'taskonomy': 512.0, 'online': 256.0,
                                  'apolloscape2': 2304.0 / 2.0, '3d-ken-burns': 512.0}

    def getData(self):
        with open(self.dir_anno, 'r') as load_f:
            all_annos = json.load(load_f)

        curriculum_list = list(np.random.choice(len(all_annos), len(all_annos), replace=False))

        rgb_paths = [
            os.path.join(cfg.ROOT_DIR, self.root, all_annos[i]['rgb_path']) 
            for i in range(len(all_annos))
        ]
        depth_paths = [
            os.path.join(cfg.ROOT_DIR, self.root, all_annos[i]['depth_path'])
            if 'depth_path' in all_annos[i]
            else None
            for i in range(len(all_annos))
        ]
        disp_paths = [
            os.path.join(cfg.ROOT_DIR, self.root, all_annos[i]['disp_path'])
            if 'disp_path' in all_annos[i]
            else None
            for i in range(len(all_annos))
        ]
        mask_paths = [
            (
                os.path.join(cfg.ROOT_DIR, self.root, all_annos[i]['mask_path'])
                if all_annos[i]['mask_path'] is not None 
                else None
            )
            if 'mask_path' in all_annos[i]
            else None
            for i in range(len(all_annos))
        ]
        ins_paths = [
            (
                os.path.join(cfg.ROOT_DIR, self.root, all_annos[i]['ins_planes_path'])
                if all_annos[i]['ins_planes_path'] is not None 
                else None
            )
            if 'ins_planes_path' in all_annos[i]
            else None
            for i in range(len(all_annos))
        ]
        cam_intrinsics = [
            (
                all_annos[i]['cam_intrinsic_para']
                if 'cam_intrinsic_para' in all_annos[i]
                else None
            )
            for i in range(len(all_annos))
        ] # [f, cx, cy]

        return rgb_paths, depth_paths, disp_paths, mask_paths, ins_paths, cam_intrinsics, all_annos, curriculum_list

    def __getitem__(self, anno_index):
        if 'train' in self.opt.phase:
            data = self.online_aug(anno_index)
        else:
            data = self.load_test_data(anno_index)
        return data

    def load_test_data(self, anno_index):
        """
        Augment data for training online randomly. The invalid parts in the depth map are set to -1.0, while the parts
        in depth bins are set to cfg.MODEL.DECODER_OUTPUT_C + 1.
        :param anno_index: data index.
        """
        rgb_path = self.rgb_paths[anno_index]

        rgb = cv2.imread(rgb_path)[:, :, ::-1]  # bgr, H*W*C
        depth, sky_mask, mask_valid = self.load_depth(anno_index, rgb)

        rgb_resize = cv2.resize(rgb, (cfg.DATASET.CROP_SIZE[1], cfg.DATASET.CROP_SIZE[0]),
                              interpolation=cv2.INTER_LINEAR)
        # to torch, normalize
        rgb_torch = self.scale_torch(rgb_resize.copy())
        # normalize disp and depth
        depth_normal = depth / (depth.max() + 1e-8)
        depth_normal[~mask_valid.astype(np.bool)] = 0

        data = {'rgb': rgb_torch, 'gt_depth': depth_normal}
        return data

    def online_aug(self, anno_index):
        """
        Augment data for training online randomly.
        :param anno_index: data index.
        """
        rgb_path = self.rgb_paths[anno_index]
        # depth_path = self.depth_paths[anno_index]
        rgb = cv2.imread(rgb_path)[:, :, ::-1]   # rgb, H*W*C

        disp, depth, \
        invalid_disp, invalid_depth, \
        ins_planes_mask, sky_mask, \
        ground_mask, depth_path = self.load_training_data(anno_index, rgb)
        rgb_aug = self.rgb_aug(rgb)

        # resize rgb, depth, disp
        flip_flg, resize_size, crop_size, pad, resize_ratio = self.set_flip_resize_crop_pad(rgb_aug)

        # focal length
        cam_intrinsic = self.cam_intrinsics[anno_index] if self.cam_intrinsics[anno_index] is not None \
                        else [float(self.focal_length_dict[self.dataset_name.lower()]), cfg.DATASET.CROP_SIZE[1] / 2, cfg.DATASET.CROP_SIZE[0] / 2] if self.dataset_name.lower() in self.focal_length_dict \
                        else [256.0, cfg.DATASET.CROP_SIZE[1] / 2, cfg.DATASET.CROP_SIZE[0] / 2]
        focal_length = cam_intrinsic[0] * (resize_size[0] + resize_size[1]) / ((cam_intrinsic[1] + cam_intrinsic[2]) * 2)
        focal_length = focal_length * resize_ratio
        focal_length = float(focal_length)

        rgb_resize = self.flip_reshape_crop_pad(rgb_aug, flip_flg, resize_size, crop_size, pad, 0)
        depth_resize = self.flip_reshape_crop_pad(depth, flip_flg, resize_size, crop_size, pad, -1, resize_method='nearest')
        disp_resize = self.flip_reshape_crop_pad(disp, flip_flg, resize_size, crop_size, pad, -1, resize_method='nearest')

        # resize sky_mask, and invalid_regions
        sky_mask_resize = self.flip_reshape_crop_pad(sky_mask.astype(np.uint8),
                                                     flip_flg,
                                                     resize_size,
                                                     crop_size,
                                                     pad,
                                                     0,
                                                     resize_method='nearest')
        invalid_disp_resize = self.flip_reshape_crop_pad(invalid_disp.astype(np.uint8),
                                                         flip_flg,
                                                         resize_size,
                                                         crop_size,
                                                         pad,
                                                         0,
                                                         resize_method='nearest')
        invalid_depth_resize = self.flip_reshape_crop_pad(invalid_depth.astype(np.uint8),
                                                          flip_flg,
                                                          resize_size,
                                                          crop_size,
                                                          pad,
                                                          0,
                                                          resize_method='nearest')
        # resize ins planes
        ins_planes_mask[ground_mask] = int(np.unique(ins_planes_mask).max() + 1)
        ins_planes_mask_resize = self.flip_reshape_crop_pad(ins_planes_mask.astype(np.uint8),
                                                            flip_flg,
                                                            resize_size,
                                                            crop_size,
                                                            pad,
                                                            0,
                                                            resize_method='nearest')

        # normalize disp and depth
        depth_resize = depth_resize / (depth_resize.max() + 1e-8) * 10
        disp_resize = disp_resize / (disp_resize.max() + 1e-8) * 10

        # invalid regions are set to -1, sky regions are set to 0 in disp and 10 in depth
        disp_resize[invalid_disp_resize.astype(np.bool) | (disp_resize > 1e7) | (disp_resize < 0)] = -1
        depth_resize[invalid_depth_resize.astype(np.bool) | (depth_resize > 1e7) | (depth_resize < 0)] = -1
        disp_resize[sky_mask_resize.astype(np.bool)] = 0  # 0
        depth_resize[sky_mask_resize.astype(np.bool)] = 20

        # to torch, normalize
        rgb_torch = self.scale_torch(rgb_resize.copy())
        depth_torch = self.scale_torch(depth_resize)
        disp_torch = self.scale_torch(disp_resize)
        ins_planes = torch.from_numpy(ins_planes_mask_resize)
        focal_length = torch.tensor(focal_length)

        if ('taskonomy' in self.dataset_name.lower()) or ('3d-ken-burns' in self.dataset_name.lower()):
            quality_flg = np.array(3)
        elif ('diml' in self.dataset_name.lower()):
            quality_flg = np.array(2)
        else:
            quality_flg = np.array(1)

        data = {'rgb': rgb_torch, 'depth': depth_torch, 'disp': disp_torch,
                'A_paths': rgb_path, 'B_paths': depth_path, 'quality_flg': quality_flg,
                'planes': ins_planes, 'focal_length': focal_length}
        return data

    def rgb_aug(self, rgb):
        # data augmentation for rgb
        img_aug = transforms.ColorJitter(brightness=0.0, contrast=0.3, saturation=0.1, hue=0)(Image.fromarray(rgb))
        rgb_aug_gray_compress = iaa.Sequential([iaa.MultiplyAndAddToBrightness(mul=(0.6, 1.25), add=(-20, 20)),
                                                iaa.Grayscale(alpha=(0.0, 1.0)),
                                                iaa.JpegCompression(compression=(0, 70)),
                                                ], random_order=True)
        rgb_aug_blur1 = iaa.AverageBlur(k=((0, 5), (0, 6)))
        rgb_aug_blur2 = iaa.MotionBlur(k=9, angle=[-45, 45])
        img_aug = rgb_aug_gray_compress(image=np.array(img_aug))
        blur_flg = np.random.uniform(0.0, 1.0)
        img_aug = rgb_aug_blur1(image=img_aug) if blur_flg > 0.7 else img_aug
        img_aug = rgb_aug_blur2(image=img_aug) if blur_flg < 0.3 else img_aug
        rgb_colorjitter = np.array(img_aug)
        return rgb_colorjitter

    def set_flip_resize_crop_pad(self, A):
        """
        Set flip, padding, reshaping and cropping flags.
        :param A: Input image, [H, W, C]
        :return: Data augamentation parameters
        """
        # flip
        flip_prob = np.random.uniform(0.0, 1.0)
        flip_flg = True if flip_prob > 0.5 and 'train' in self.opt.phase else False

        # crop
        if 'train' in self.opt.phase:
            image_h, image_w = A.shape[:2]
            croph, cropw = np.random.randint(image_h // 2, image_h + 1), np.random.randint(image_w // 2, image_w + 1)
            h0 = np.random.randint(image_h - croph + 1)
            w0 = np.random.randint(image_w - cropw + 1)
            crop_size = [w0, h0, cropw, croph]
        else:
            crop_size = [0, 0, resize_size[1], resize_size[0]]

        # # crop
        # start_y = 0 if resize_size[0] <= cfg.DATASET.CROP_SIZE[0] else np.random.randint(0, resize_size[0] - cfg.DATASET.CROP_SIZE[0])
        # start_x = 0 if resize_size[1] <= cfg.DATASET.CROP_SIZE[1] else np.random.randint(0, resize_size[1] - cfg.DATASET.CROP_SIZE[1])
        # crop_height = resize_size[0] if resize_size[0] <= cfg.DATASET.CROP_SIZE[0] else cfg.DATASET.CROP_SIZE[0]
        # crop_width = resize_size[1] if resize_size[1] <= cfg.DATASET.CROP_SIZE[1] else cfg.DATASET.CROP_SIZE[1]
        # crop_size = [start_x, start_y, crop_width, crop_height] if 'train' in self.opt.phase else [0, 0, resize_size[1], resize_size[0]]

        # # reshape
        # ratio_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  #
        # if 'train' in self.opt.phase:
        #     resize_ratio = ratio_list[np.random.randint(len(ratio_list))]
        # else:
        #     resize_ratio = 1.0

        # resize_size = [int(A.shape[0] * resize_ratio + 0.5),
        #                int(A.shape[1] * resize_ratio + 0.5)]  # [height, width]

        # reshape
        resize_size = [cfg.DATASET.CROP_SIZE[0], cfg.DATASET.CROP_SIZE[1]]
        resize_ratio = (cfg.DATASET.CROP_SIZE[0] + cfg.DATASET.CROP_SIZE[1]) / (cropw + croph)

        # # pad
        # pad_height = 0 if resize_size[0] > cfg.DATASET.CROP_SIZE[0] else cfg.DATASET.CROP_SIZE[0] - resize_size[0]
        # pad_width = 0 if resize_size[1] > cfg.DATASET.CROP_SIZE[1] else cfg.DATASET.CROP_SIZE[1] - resize_size[1]
        # # [up, down, left, right]
        # pad = [pad_height, 0, pad_width, 0] if 'train' in self.opt.phase else [0, 0, 0, 0]
        pad = [0, 0, 0, 0]
        return flip_flg, resize_size, crop_size, pad, resize_ratio

    def flip_reshape_crop_pad(self, img, flip, resize_size, crop_size, pad, pad_value=0, resize_method='bilinear'):
        """
        Flip, pad, reshape, and crop the image.
        :param img: input image, [C, H, W]
        :param flip: flip flag
        :param crop_size: crop size for the image, [x, y, width, height]
        :param pad: pad the image, [up, down, left, right]
        :param pad_value: padding value
        :return:
        """
        # Flip
        if flip:
            img = np.flip(img, axis=1)
        
        # Crop the image
        img_crop = img[
            crop_size[1]:crop_size[1] + crop_size[3],
            crop_size[0]:crop_size[0] + crop_size[2]
        ]

        # Resize the raw image
        if resize_method == 'nearest':
            img_resize = cv2.resize(img_crop, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)
        else:
            img_resize = cv2.resize(img_crop, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_LINEAR)

        # Pad the raw image
        if len(img.shape) == 3:
            img_pad = np.pad(img_resize, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), 'constant',
                             constant_values=(pad_value, pad_value))
        else:
            img_pad = np.pad(img_resize, ((pad[0], pad[1]), (pad[2], pad[3])), 'constant',
                             constant_values=(pad_value, pad_value))

        # # Resize the raw image
        # if resize_method == 'bilinear':
        #     img_resize = cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_LINEAR)
        # elif resize_method == 'nearest':
        #     img_resize = cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)
        # else:
        #     raise ValueError

        # # Crop the resized image
        # img_crop = img_resize[crop_size[1]:crop_size[1] + crop_size[3], crop_size[0]:crop_size[0] + crop_size[2]]

        # # Pad the raw image
        # if len(img.shape) == 3:
        #     img_pad = np.pad(img_crop, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), 'constant',
        #                      constant_values=(pad_value, pad_value))
        # else:
        #     img_pad = np.pad(img_crop, ((pad[0], pad[1]), (pad[2], pad[3])), 'constant',
        #                      constant_values=(pad_value, pad_value))

        return img_pad

    # def depth_to_bins(self, depth):
    #     """
    #     Discretize depth into depth bins
    #     Mark invalid padding area as cfg.MODEL.DECODER_OUTPUT_C + 1
    #     :param depth: 1-channel depth, [1, h, w]
    #     :return: depth bins [1, h, w]
    #     """
    #     invalid_mask = depth < 1e-8
    #     depth[depth < cfg.DATASET.DEPTH_MIN] = cfg.DATASET.DEPTH_MIN
    #     depth[depth > cfg.DATASET.DEPTH_MAX] = cfg.DATASET.DEPTH_MAX
    #     bins = ((torch.log10(depth) - cfg.DATASET.DEPTH_MIN_LOG) / cfg.DATASET.DEPTH_BIN_INTERVAL).to(torch.int)
    #     bins[invalid_mask] = cfg.MODEL.DECODER_OUTPUT_C + 1
    #     bins[bins == cfg.MODEL.DECODER_OUTPUT_C] = cfg.MODEL.DECODER_OUTPUT_C - 1
    #     depth[invalid_mask] = -1.0
    #     return bins

    def scale_torch(self, img):
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
                                            transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)])
            img = transform(img)
        else:
            img = img.astype(np.float32)
            img = torch.from_numpy(img)
        return img

    def load_depth(self, anno_index, rgb):
        """
        Load disparity, depth, and mask maps
        :return
            disp: disparity map,  np.float
            depth: depth map, np.float
            sem_mask: semantic masks, including sky, road, np.uint8
            ins_mask: plane instance masks, np.uint8
        """
        # load depth
        depth = cv2.imread(self.depth_paths[anno_index], -1)
        depth, mask_valid = self.preprocess_depth(depth, self.depth_paths[anno_index])

        # load semantic mask, such as road, sky
        if len(self.rgb_paths) == len(self.sem_masks) and self.sem_masks[anno_index] is not None:
            sem_mask = cv2.imread(self.sem_masks[anno_index], -1).astype(np.uint8)
        else:
            sem_mask = np.zeros(depth.shape, dtype=np.uint8)
        sky_mask = sem_mask == 17

        return depth, sky_mask, mask_valid

    def load_training_data(self, anno_index, rgb):
        """
        Load disparity, depth, and mask maps
        :return
            disp: disparity map,  np.float
            depth: depth map, np.float
            sem_mask: semantic masks, including sky, road, np.uint8
            ins_mask: plane instance masks, np.uint8
        """
        # load depth, rgb, disp
        if (self.depth_paths[anno_index] != None) and (self.disp_paths[anno_index] != None):
            # dataset has both depth and disp
            disp = cv2.imread(self.disp_paths[anno_index], -1)
            disp = (disp / (disp.max() + 1e-8) * 60000).astype(np.uint16)
            depth = cv2.imread(self.depth_paths[anno_index], -1)
            depth = (depth / (depth.max() + 1e-8) * 60000).astype(np.uint16)
            depth_path = self.depth_paths[anno_index]
        elif self.disp_paths[anno_index] != None:
            # dataset only has disparity
            disp = cv2.imread(self.disp_paths[anno_index], -1)
            disp_mask = disp < 1e-8
            depth = 1 / (disp + 1e-8)
            depth[disp_mask] = 0
            depth = (depth / (depth.max() + 1e-8) * 60000).astype(np.uint16)
            depth_path = self.disp_paths[anno_index]
        elif self.depth_paths[anno_index] != None:
            # dataset only has depth
            depth_path = self.depth_paths[anno_index]
            depth = cv2.imread(self.depth_paths[anno_index], -1)
            depth = (self.loading_check(depth, depth_path)).astype(np.uint16)
            depth_mask = depth < 1e-8
            disp = 1 / (depth + 1e-8)
            disp[depth_mask] = 0
            disp = (disp / (disp.max() + 1e-8) * 60000).astype(np.uint16)
        else:
            depth = np.full((rgb.shape[0], rgb.shape[1]), 0, dtype=np.uint16)
            disp = np.full((rgb.shape[0], rgb.shape[1]), 0, dtype=np.uint16)
            depth_path = 'None'

        # load semantic mask, such as road, sky
        if len(self.rgb_paths) == len(self.sem_masks) and self.sem_masks[anno_index] is not None:
            sem_mask = cv2.imread(self.sem_masks[anno_index], -1).astype(np.uint8)
        else:
            sem_mask = np.zeros(disp.shape, dtype=np.uint8)

        # load planes mask
        if len(self.rgb_paths) == len(self.ins_paths) and self.ins_paths[anno_index] is not None:
            ins_planes_mask = cv2.imread(self.ins_paths[anno_index], -1).astype(np.uint8)
        else:
            ins_planes_mask = np.zeros(disp.shape, dtype=np.uint8)

        sky_mask = sem_mask == 17
        road_mask = sem_mask == 49

        invalid_disp = disp < 1e-8
        invalid_depth = depth < 1e-8
        return disp, depth, invalid_disp, invalid_depth, ins_planes_mask, sky_mask, road_mask, depth_path

        #return disp, depth, sem_mask, depth_path, ins_planes_mask

    def preprocess_depth(self, depth, img_path):
        if 'diml' in img_path.lower():
            drange = 65535.0
        elif 'taskonomy' in img_path.lower():
            depth[depth > 23000] = 0
            drange = 23000.0
        else:
            #depth_filter1 = depth[depth > 1e-8]
            #drange = (depth_filter1.max() - depth_filter1.min())
            drange = depth.max()
        depth_norm = depth / (drange + 1e-8)
        mask_valid = (depth_norm > 1e-8).astype(np.float)
        return depth_norm, mask_valid

    def loading_check(self, depth, depth_path):
        if 'taskonomy' in depth_path:
            # invalid regions in taskonomy are set to 65535 originally
            depth[depth >= 28000] = 0
        if '3d-ken-burns' in depth_path:
            # maybe sky regions
            depth[depth >= 47000] = 0
        return depth

    def __len__(self):
        return self.data_size

    # def name(self):
    #     return 'DiverseDepth'

