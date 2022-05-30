import torch
import torch.nn as nn


class MEADSTD_TANH_NORM_Loss(nn.Module):
    """
    loss = MAE((d-u)/s - d') + MAE(tanh(0.01*(d-u)/s) - tanh(0.01*d'))
    """
    def __init__(self, valid_threshold=-1e-8, max_threshold=1e8):
        super(MEADSTD_TANH_NORM_Loss, self).__init__()
        self.valid_threshold = valid_threshold
        self.max_threshold = max_threshold
        #self.thres1 = 0.9

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
        """
        Calculate loss.
        """
        mask = (gt > self.valid_threshold) & (gt < self.max_threshold)   # [b, c, h, w]
        mask_sum = torch.sum(mask, dim=(1, 2, 3))
        # mask invalid batches
        mask_batch = mask_sum > 100
        if True not in mask_batch:
            return torch.tensor(0.0, dtype=torch.float).cuda()
        mask_maskbatch = mask[mask_batch]
        pred_maskbatch = pred[mask_batch]
        gt_maskbatch = gt[mask_batch]

        gt_mean, gt_std = self.transform(gt)
        gt_trans = (gt_maskbatch - gt_mean[:, None, None, None]) / (gt_std[:, None, None, None] + 1e-8)

        B, C, H, W = gt_maskbatch.shape
        loss = 0
        loss_tanh = 0
        for i in range(B):
            mask_i = mask_maskbatch[i, ...]
            pred_depth_i = pred_maskbatch[i, ...][mask_i]
            gt_trans_i = gt_trans[i, ...][mask_i]

            depth_diff = torch.abs(gt_trans_i - pred_depth_i)
            loss += torch.mean(depth_diff)

            tanh_norm_gt = torch.tanh(0.01*gt_trans_i)
            tanh_norm_pred = torch.tanh(0.01*pred_depth_i)
            loss_tanh += torch.mean(torch.abs(tanh_norm_gt - tanh_norm_pred))
        loss_out = loss/B + loss_tanh/B
        return loss_out.float()

if __name__ == '__main__':
    ilnr_loss = MEADSTD_TANH_NORM_Loss()
    pred_depth = torch.rand([3, 1, 385, 513]).cuda()
    gt_depth = torch.rand([3, 1, 385, 513]).cuda()

    loss = ilnr_loss(pred_depth, gt_depth)
    print(loss)
