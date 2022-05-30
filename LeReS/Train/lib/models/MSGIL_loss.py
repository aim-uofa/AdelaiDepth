import torch
import torch.nn as nn



class MSGIL_NORM_Loss(nn.Module):
    """
    Our proposed GT normalized Multi-scale Gradient Loss Fuction.
    """
    def __init__(self, scale=4, valid_threshold=-1e-8, max_threshold=1e8):
        super(MSGIL_NORM_Loss, self).__init__()
        self.scales_num = scale
        self.valid_threshold = valid_threshold
        self.max_threshold = max_threshold
        self.EPSILON = 1e-6

    def one_scale_gradient_loss(self, pred_scale, gt, mask):
        mask_float = mask.to(dtype=pred_scale.dtype, device=pred_scale.device)

        d_diff = pred_scale - gt

        v_mask = torch.mul(mask_float[:, :, :-2, :], mask_float[:, :, 2:, :])
        v_gradient = torch.abs(d_diff[:, :, :-2, :] - d_diff[:, :, 2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(d_diff[:, :, :, :-2] - d_diff[:, :, :, 2:])
        h_mask = torch.mul(mask_float[:, :, :, :-2], mask_float[:, :, :, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        valid_num = torch.sum(h_mask) + torch.sum(v_mask)

        gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        gradient_loss = gradient_loss / (valid_num + 1e-8)

        return gradient_loss

    def transform(self, gt):
        # Get mean and standard deviation
        data_mean = []
        data_std_dev = []
        for i in range(gt.shape[0]):
            gt_i = gt[i]
            mask = gt_i > 0
            depth_valid = gt_i[mask]
            depth_valid = depth_valid[:5]
            if depth_valid.shape[0] < 10:
                data_mean.append(torch.tensor(0).cuda())
                data_std_dev.append(torch.tensor(1).cuda())
                continue
            size = depth_valid.shape[0]
            depth_valid_sort, _ = torch.sort(depth_valid, 0)
            depth_valid_mask = depth_valid_sort[int(size*0.1): -int(size*0.1)]
            data_mean.append(depth_valid_mask.mean())
            data_std_dev.append(depth_valid_mask.std())
        data_mean = torch.stack(data_mean, dim=0).cuda()
        data_std_dev = torch.stack(data_std_dev, dim=0).cuda()

        return data_mean, data_std_dev

    def forward(self, pred, gt):
        mask = gt > self.valid_threshold
        grad_term = 0.0
        gt_mean, gt_std = self.transform(gt)
        gt_trans = (gt - gt_mean[:, None, None, None]) / (gt_std[:, None, None, None] + 1e-8)
        for i in range(self.scales_num):
            d_gt = gt_trans[:, :, ::2, ::2]
            d_pred = pred[:, :, ::2, ::2]
            d_mask = mask[:, :, ::2, ::2]
            grad_term += self.one_scale_gradient_loss(d_pred, d_gt, d_mask)
        return grad_term


if __name__ == '__main__':
    msgi_loss = MSGIL_NORM_Loss()
    pred_depth = torch.rand([2, 1, 385, 513]).cuda()
    gt_depth = torch.rand([2, 1, 385, 513]).cuda()
    loss = msgi_loss(pred_depth, gt_depth)
    print(loss)


