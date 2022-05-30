import torch
import torch.nn.functional

from . import network_auxi as network
from lib.configs.config import cfg
from lib.utils.net_tools import *
from lib.models.PWN_planes import PWNPlanesLoss
from lib.models.PWN_edges import EdgeguidedNormalRegressionLoss
from lib.models.ranking_loss import EdgeguidedRankingLoss
from lib.models.ILNR_loss import MEADSTD_TANH_NORM_Loss
from lib.models.MSGIL_loss import MSGIL_NORM_Loss


class RelDepthModel(nn.Module):
    def __init__(self):
        super(RelDepthModel, self).__init__()
        self.depth_model = DepthModel()
        self.losses = ModelLoss()

    def forward(self, data, is_train=True):
        # Input data is a_real, predicted data is b_fake, groundtruth is b_real
        self.inputs = data['rgb'].cuda()
        self.logit, self.auxi = self.depth_model(self.inputs)
        if is_train:
            self.losses_dict = self.losses.criterion(self.logit, self.auxi, data)
        else:
            self.losses_dict = {'total_loss': torch.tensor(0.0, dtype=torch.float).cuda()}
        return {'decoder': self.logit, 'auxi': self.auxi, 'losses': self.losses_dict}

    def inference(self, data):
        with torch.no_grad():
            out = self.forward(data, is_train=False)
            pred_depth = out['decoder']
            pred_disp = out['auxi']
            pred_depth_normalize = (pred_depth - pred_depth.min() + 1) / (pred_depth.max() - pred_depth.min()) #pred_depth - pred_depth.min() #- pred_depth.max()
            pred_depth_out = pred_depth
            pred_disp_normalize = (pred_disp - pred_disp.min() + 1) / (pred_disp.max() - pred_disp.min())
            return {'pred_depth': pred_depth_out, 'pred_depth_normalize': pred_depth_normalize,
                    'pred_disp': pred_disp, 'pred_disp_normalize': pred_disp_normalize,
                    }


class ModelLoss(nn.Module):
    def __init__(self):
        super(ModelLoss, self).__init__()

        ################Loss for the main branch, i.e. on the depth map#################
        # Geometry Loss
        self.pn_plane = PWNPlanesLoss(focal_x=cfg.DATASET.FOCAL_X, focal_y=cfg.DATASET.FOCAL_Y,
                                            input_size=cfg.DATASET.CROP_SIZE, sample_groups=5000, xyz_mode='xyz')
        self.pn_edge = EdgeguidedNormalRegressionLoss(mask_value=-1e-8, max_threshold=10.1)
        # self.surface_normal_loss = SurfaceNormalLoss()

        # the scale can be adjusted
        self.msg_normal_loss = MSGIL_NORM_Loss(scale=4, valid_threshold=-1e-8)

        # Scale shift invariant. SSIMAEL_Loss is MIDAS loss. MEADSTD_TANH_NORM_Loss is our normalization loss.
        self.meanstd_tanh_loss = MEADSTD_TANH_NORM_Loss(valid_threshold=-1e-8)

        self.ranking_edge_loss = EdgeguidedRankingLoss(mask_value=-1e-8)


        ################Loss for the auxi branch, i.e. on the disp map#################
        # the scale can be adjusted
        self.msg_normal_auxiloss = MSGIL_NORM_Loss(scale=4, valid_threshold=-1e-8)

        # Scale shift invariant. SSIMAEL_Loss is MIDAS loss. MEADSTD_TANH_NORM_Loss is our normalization loss.
        self.meanstd_tanh_auxiloss = MEADSTD_TANH_NORM_Loss(valid_threshold=-1e-8)

        self.ranking_edge_auxiloss = EdgeguidedRankingLoss(mask_value=-1e-8)

    def criterion(self, pred_logit, auxi, data):
        loss1 = self.decoder_loss(pred_logit, data)
        loss2 = self.auxi_loss(auxi, data)
        loss = {}
        loss.update(loss1)
        loss.update(loss2)
        loss['total_loss'] = loss1['total_loss'] + loss2['total_loss']
        return loss

    def auxi_loss(self, auxi, data):
        loss = {}
        if 'disp' not in data:
            return {'total_loss': torch.tensor(0.0).cuda()}

        gt = data['disp'].to(device=auxi.device)

        if '_ranking-edge-auxi_' in cfg.TRAIN.LOSS_MODE.lower():
            loss['ranking-edge_auxiloss'] = self.ranking_edge_auxiloss(auxi, gt, data['rgb'])

        if '_msgil-normal-auxi_' in cfg.TRAIN.LOSS_MODE.lower():
            loss['msg_normal_auxiloss'] = (self.msg_normal_auxiloss(auxi, gt) * 0.5).float()

        if '_meanstd-tanh-auxi_' in cfg.TRAIN.LOSS_MODE.lower():
            loss['meanstd-tanh_auxiloss'] = self.meanstd_tanh_auxiloss(auxi, gt)

        total_loss = sum(loss.values())
        loss['total_loss'] = total_loss * cfg.TRAIN.LOSS_AUXI_WEIGHT
        return loss

    def decoder_loss(self, pred_logit, data):
        pred_depth = pred_logit

        gt_depth = data['depth'].to(device=pred_depth.device)

        # High-quality data, except webstereo data
        mask_high_quality = data['quality_flg'] ==3
        mask_mid_quality = data['quality_flg'] >= 2
        # gt_depth_high = gt_depth[mask_high_quality]
        # pred_depth_high = pred_depth[mask_high_quality]

        gt_depth_mid = gt_depth[mask_mid_quality]
        pred_depth_mid = pred_depth[mask_mid_quality]


        #gt_depth_filter = data['mask_highquality']]
        #pred_depth_filter = pred_depth[data['mask_highquality']]
        #focal_length_filter = data['focal_length'][data['mask_highquality']]

        # if gt_depth_high.ndim == 3:
        #     gt_depth_high = gt_depth_high[None, :, :, :]
        #     pred_depth_high = pred_depth_high[None, :, :, :]
        if gt_depth_mid.ndim == 3:
            gt_depth_mid = gt_depth_mid[None, :, :, :]
            pred_depth_mid = pred_depth_mid[None, :, :, :]
        loss = {}

        if '_pairwise-normal-regress-edge_' in cfg.TRAIN.LOSS_MODE.lower() or \
                '_pairwise-normal-regress-plane_' in cfg.TRAIN.LOSS_MODE.lower():
            pred_ssinv = recover_scale_shift_depth(pred_depth, gt_depth, min_threshold=-1e-8, max_threshold=10.1)
        else:
            pred_ssinv = None

        # Geometry Loss
        if '_pairwise-normal-regress-plane_' in cfg.TRAIN.LOSS_MODE.lower():
            focal_length = data['focal_length'] if 'focal_length' in data else None
            loss['pairwise-normal-regress-plane_loss'] = self.pn_plane(gt_depth,
                                                                       pred_ssinv,
                                                                       data['planes'],
                                                                       focal_length)
        if '_pairwise-normal-regress-edge_' in cfg.TRAIN.LOSS_MODE.lower():
            if mask_high_quality.sum():
                loss['pairwise-normal-regress-edge_loss'] = self.pn_edge(pred_ssinv[mask_high_quality],
                                                                         gt_depth[mask_high_quality],
                                                                         data['rgb'][mask_high_quality],
                                                                         focal_length=data['focal_length'][mask_high_quality])
            else:
                loss['pairwise-normal-regress-edge_loss'] = pred_ssinv.sum() * 0.

        # Scale-shift Invariant Loss
        if '_meanstd-tanh_' in cfg.TRAIN.LOSS_MODE.lower():
            loss_ssi = self.meanstd_tanh_loss(pred_depth_mid, gt_depth_mid)
            loss['meanstd-tanh_loss'] = loss_ssi

        if '_ranking-edge_' in cfg.TRAIN.LOSS_MODE.lower():
            loss['ranking-edge_loss'] = self.ranking_edge_loss(pred_depth, gt_depth, data['rgb'])

        # Multi-scale Gradient Loss
        if '_msgil-normal_' in cfg.TRAIN.LOSS_MODE.lower():
            loss['msg_normal_loss'] = (self.msg_normal_loss(pred_depth, gt_depth) * 0.1).float()

        total_loss = sum(loss.values())
        loss['total_loss'] = total_loss
        return loss


class ModelOptimizer(object):
    def __init__(self, model):
        super(ModelOptimizer, self).__init__()
        encoder_params = []
        encoder_params_names = []
        decoder_params = []
        decoder_params_names = []
        nograd_param_names = []

        for key, value in model.named_parameters():
            if value.requires_grad:
                if 'res' in key:
                    encoder_params.append(value)
                    encoder_params_names.append(key)
                else:
                    decoder_params.append(value)
                    decoder_params_names.append(key)
            else:
                nograd_param_names.append(key)

        lr_encoder = cfg.TRAIN.BASE_LR
        lr_decoder = cfg.TRAIN.BASE_LR * cfg.TRAIN.SCALE_DECODER_LR
        weight_decay = 0.0005

        net_params = [
            {'params': encoder_params,
             'lr': lr_encoder,
             'weight_decay': weight_decay},
            {'params': decoder_params,
             'lr': lr_decoder,
             'weight_decay': weight_decay},
        ]
        self.optimizer = torch.optim.SGD(net_params, momentum=0.9)
        self.model = model

    def optim(self, loss):
        self.optimizer.zero_grad()
        loss_all = loss['total_loss']
        loss_all.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()


class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        backbone = network.__name__.split('.')[-1] + '.' + cfg.MODEL.ENCODER
        self.encoder_modules = get_func(backbone)()
        self.decoder_modules = network.Decoder()
        self.auxi_modules = network.AuxiNetV2()

    def forward(self, x):
        lateral_out = self.encoder_modules(x)
        out_logit, auxi_input = self.decoder_modules(lateral_out)
        out_auxi = self.auxi_modules(auxi_input)
        return out_logit, out_auxi


def recover_scale_shift_depth(pred, gt, min_threshold=1e-8, max_threshold=1e8):
    b, c, h, w = pred.shape
    mask = (gt > min_threshold) & (gt < max_threshold)  # [b, c, h, w]
    EPS = 1e-6 * torch.eye(2, dtype=pred.dtype, device=pred.device)
    scale_shift_batch = []
    ones_img = torch.ones((1, h, w), dtype=pred.dtype, device=pred.device)
    for i in range(b):
        mask_i = mask[i, ...]
        pred_valid_i = pred[i, ...][mask_i]
        ones_i = ones_img[mask_i]
        pred_valid_ones_i = torch.stack((pred_valid_i, ones_i), dim=0)  # [c+1, n]
        A_i = torch.matmul(pred_valid_ones_i, pred_valid_ones_i.permute(1, 0))  # [2, 2]
        A_inverse = torch.inverse(A_i + EPS)

        gt_i = gt[i, ...][mask_i]
        B_i = torch.matmul(pred_valid_ones_i, gt_i)[:, None]  # [2, 1]
        scale_shift_i = torch.matmul(A_inverse, B_i)  # [2, 1]
        scale_shift_batch.append(scale_shift_i)
    scale_shift_batch = torch.stack(scale_shift_batch, dim=0)  # [b, 2, 1]
    ones = torch.ones_like(pred)
    pred_ones = torch.cat((pred, ones), dim=1)  # [b, 2, h, w]
    pred_scale_shift = torch.matmul(pred_ones.permute(0, 2, 3, 1).reshape(b, h * w, 2), scale_shift_batch)  # [b, h*w, 1]
    pred_scale_shift = pred_scale_shift.permute(0, 2, 1).reshape((b, c, h, w))
    return pred_scale_shift
