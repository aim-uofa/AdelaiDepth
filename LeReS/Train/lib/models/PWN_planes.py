import torch
import torch.nn as nn
import numpy as np


class PWNPlanesLoss(nn.Module):
    """
    Virtual Normal Loss Function.
    """
    def __init__(self, focal_x, focal_y, input_size,
                 delta_cos=0.867, delta_diff_x=0.007,
                 delta_diff_y=0.007, sample_groups=5000, xyz_mode = 'uvd'):
        """
        Virtual normal planes loss, which constrain points to be on the same 3D plane.
        :para focal_x: folcal length fx
        :para focal_y: folcal length fy
        :para input_size: input image size
        :para delta_cos: a threshold for the angle among three point, three points should not be on the same plane
        :para  delta_diff_x: a threshold for the distance among three points along the x axis
        :para delta_diff_y: a threshold for the distance among three points along the y axis
        :para sample_groups: sample groups number, each group with 3 points can construct a plane
        :para xyz_mode: using (u, v, d) or (x, y, z) to construct the virtual planes
        """
        super(PWNPlanesLoss, self).__init__()
        self.fx = torch.tensor([focal_x], dtype=torch.float32).cuda()
        self.fy = torch.tensor([focal_y], dtype=torch.float32).cuda()
        self.input_size = input_size
        self.u0 = torch.tensor(input_size[1] // 2, dtype=torch.float32).cuda()
        self.v0 = torch.tensor(input_size[0] // 2, dtype=torch.float32).cuda()
        self.init_image_coor()
        self.delta_cos = delta_cos
        self.delta_diff_x = delta_diff_x
        self.delta_diff_y = delta_diff_y
        self.sample_groups = sample_groups
        self.xyz_mode = xyz_mode

    def init_image_coor(self):
        x_row = np.arange(0, self.input_size[1])
        x = np.tile(x_row, (self.input_size[0], 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy()).cuda()
        self.u_u0 = x - self.u0

        y_col = np.arange(0, self.input_size[0])  # y_col = np.arange(0, height)
        y = np.tile(y_col, (self.input_size[1], 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy()).cuda()
        self.v_v0 = y - self.v0

    def transfer_uvz(self, depth):
        max_uv = self.u_u0.max()
        u = self.u_u0.repeat((depth.shape[0], 1, 1, 1)) / max_uv
        v = self.v_v0.repeat((depth.shape[0], 1, 1, 1)) / max_uv
        z = depth
        pw = torch.cat([u, v, z], 1).permute(0, 2, 3, 1) # [b, h, w, c]
        return pw

    def transfer_xyz(self, depth):
        x = self.u_u0 * torch.abs(depth) / self.fx
        y = self.v_v0 * torch.abs(depth) / self.fy
        z = depth
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1) # [b, h, w, c]
        return pw

    def select_index(self, mask_kp):
        x, _, h, w = mask_kp.shape

        select_size = int(3 * self.sample_groups)
        p1_x = []
        p1_y = []
        p2_x = []
        p2_y = []
        p3_x = []
        p3_y = []
        valid_batch = torch.ones((x, 1), dtype=torch.bool)
        for i in range(x):
            mask_kp_i = mask_kp[i, 0, :, :]
            valid_points = torch.nonzero(mask_kp_i)

            if valid_points.shape[0] < select_size * 0.6:
                valid_points = torch.nonzero(~mask_kp_i.to(torch.uint8))
                valid_batch[i, :] = False
            elif valid_points.shape[0] < select_size:
                repeat_idx = torch.randperm(valid_points.shape[0])[:select_size - valid_points.shape[0]]
                valid_repeat = valid_points[repeat_idx]
                valid_points = torch.cat((valid_points, valid_repeat), 0)
            else:
                valid_points = valid_points
            """
            
            if valid_points.shape[0] <= select_size:
                valid_points = torch.nonzero(~mask_kp_i.to(torch.uint8))
                valid_batch[i, :] = False
            """
            select_indx = torch.randperm(valid_points.size(0))

            p1 = valid_points[select_indx[0:select_size:3]]
            p2 = valid_points[select_indx[1:select_size:3]]
            p3 = valid_points[select_indx[2:select_size:3]]

            p1_x.append(p1[:, 1])
            p1_y.append(p1[:, 0])

            p2_x.append(p2[:, 1])
            p2_y.append(p2[:, 0])

            p3_x.append(p3[:, 1])
            p3_y.append(p3[:, 0])
        p123 = {'p1_x': torch.stack(p1_x), 'p1_y': torch.stack(p1_y),
                'p2_x': torch.stack(p2_x), 'p2_y': torch.stack(p2_y),
                'p3_x': torch.stack(p3_x), 'p3_y': torch.stack(p3_y),
                'valid_batch': valid_batch}
        return p123

    def form_pw_groups(self, p123, pw):
        """
        Form 3D points groups, with 3 points in each grouup.
        :param p123: points index
        :param pw: 3D points, # [1, h, w, c]
        :return:
        """
        p1_x = p123['p1_x']
        p1_y = p123['p1_y']
        p2_x = p123['p2_x']
        p2_y = p123['p2_y']
        p3_x = p123['p3_x']
        p3_y = p123['p3_y']
        batch_list = np.arange(0, p1_x.shape[0])[:, np.newaxis]
        pw = pw.repeat((p1_x.shape[0], 1, 1, 1))
        pw1 = pw[batch_list, p1_y, p1_x, :]
        pw2 = pw[batch_list, p2_y, p2_x, :]
        pw3 = pw[batch_list, p3_y, p3_x, :]
        
        # [B, N, 3(x,y,z), 3(p1,p2,p3)]
        pw_groups = torch.cat([pw1[:, :, :, np.newaxis], pw2[:, :, :, np.newaxis], pw3[:, :, :, np.newaxis]], 3)
        return pw_groups

    def filter_mask(self, pw_pred):
        """
        :param pw_pred: constructed 3d vector (x, y, disp), [B, N, 3(x,y,z), 3(p1,p2,p3)]
        """
        xy12 = pw_pred[:, :, 0:2, 1] - pw_pred[:, :, 0:2, 0]
        xy13 = pw_pred[:, :, 0:2, 2] - pw_pred[:, :, 0:2, 0]
        xy23 = pw_pred[:, :, 0:2, 2] - pw_pred[:, :, 0:2, 1]
        # Ignore linear
        xy_diff = torch.cat([xy12[:, :, :, np.newaxis], xy13[:, :, :, np.newaxis], xy23[:, :, :, np.newaxis]],
                            3)  # [b, n, 2(xy), 3]
        m_batchsize, groups, coords, index = xy_diff.shape
        proj_query = xy_diff.view(m_batchsize * groups, -1, index).permute(0, 2, 1)  # [bn, 3(p123), 2(xy)]
        proj_key = xy_diff.view(m_batchsize * groups, -1, index)  # [bn, 2(xy), 3(p123)]
        q_norm = proj_query.norm(2, dim=2)  # [bn, 3(p123)]
        nm = torch.bmm(q_norm.view(m_batchsize * groups, index, 1), q_norm.view(m_batchsize * groups, 1, index))  # []
        energy = torch.bmm(proj_query, proj_key)  # transpose check [bn, 3(p123), 3(p123)]
        norm_energy = energy / (nm + 1e-8)
        norm_energy = norm_energy.view(m_batchsize * groups, -1) # [bn, 9(p123)]
        mask_cos = torch.sum((norm_energy > self.delta_cos) + (norm_energy < -self.delta_cos), 1) > 3  # igonre
        mask_cos = mask_cos.view(m_batchsize, groups)  # [b, n]  # igonre

        #ignore near
        mask_x = torch.sum(torch.abs(xy_diff[:, :, 0, :]) < self.delta_diff_x, 2) > 0
        mask_y = torch.sum(torch.abs(xy_diff[:, :, 1, :]) < self.delta_diff_y, 2) > 0
        mask_near = mask_x & mask_y
        mask_valid_pts = ~(mask_cos | mask_near)
        return mask_valid_pts

    def select_points_groups(self, pred_depth, mask_kp):
        p123 = self.select_index(mask_kp) # p1_x: [x, N]

        uvz_pred = self.transfer_uvz(pred_depth) #[1, h, w, 3(xyz)]
        uvz_groups_pred = self.form_pw_groups(p123, uvz_pred) # [x, N, 3(x,y,z), 3(p1,p2,p3)]
        # mask:[b, n]
        mask_valid_pts = (self.filter_mask(uvz_groups_pred)).to(torch.bool)  # [x, n]
        mask_valid_batch = p123['valid_batch'].repeat(1, mask_valid_pts.shape[1])  # [x, n]
        mask_valid = mask_valid_pts & mask_valid_batch.cuda() # [x, n]

        if self.xyz_mode == 'uvd':
            pw_groups_pred = uvz_groups_pred
        else:
            xyz_pred = self.transfer_xyz(pred_depth)  # [1, h, w, 3(xyz)]
            pw_groups_pred = self.form_pw_groups(p123, xyz_pred)  # [x, N, 3(x,y,z), 3(p1,p2,p3)]
        return pw_groups_pred, mask_valid

    def constrain_a_plane_loss(self, pw_groups_pre_i, mask_valid_i):
        """
        pw_groups_pre: selected points groups for the i-th plane,
        """
        if torch.sum(mask_valid_i) < 2:
            return [0.0, 0]
        pw_groups_pred_i = pw_groups_pre_i[mask_valid_i]  # [n, 3, 3]
        p12 = pw_groups_pred_i[:, :, 1] - pw_groups_pred_i[:, :, 0]
        p13 = pw_groups_pred_i[:, :, 2] - pw_groups_pred_i[:, :, 0]
        virtual_normal = torch.cross(p12, p13, dim=1)  # [n, 3]
        norm = torch.norm(virtual_normal, 2, dim=1, keepdim=True)
        virtual_normal = virtual_normal / (norm + 1e-8)

        # re-orient normals consistently
        orient_mask = torch.sum(torch.squeeze(virtual_normal) * torch.squeeze(pw_groups_pred_i[:, :, 0]), dim=1) > 0
        virtual_normal[orient_mask] *= -1
        #direct = virtual_normal[:, 2] / torch.abs(virtual_normal[:, 2])
        #virtual_normal = virtual_normal / direct[:, None]  # [n, 3]

        aver_normal = torch.sum(virtual_normal, dim=0)
        aver_norm = torch.norm(aver_normal, 2, dim=0, keepdim=True)
        aver_normal = aver_normal / (aver_norm + 1e-5)  # [3]

        cos_diff = 1.0 - torch.sum(virtual_normal * aver_normal, dim=1)
        loss = torch.sum(cos_diff, dim=0)
        valid_num = cos_diff.numel()
        return loss, valid_num

    def forward(self, gt_depth, pred_depth, mask, focal_length=None):
        """
        Virtual normal loss.
        :param pred_depth: predicted depth map, [B,C,H,W]
        :param mask: mask for planes, each plane is noted with a value, [B, C, H, W]
        :param focal_length: focal length
        """
        B, _, _, _ = pred_depth.shape
        loss = torch.tensor(0.0).cuda()
        valid_planes_num = 0
        for i in range(B):
            self.fx = focal_length[i] if focal_length is not None else 256.0
            self.fy = focal_length[i] if focal_length is not None else 256.0

            pred_depth_i = pred_depth[i, :]
            mask_i = mask[i, :][None, :, :]
            unique_planes = torch.unique(mask_i)
            planes = [mask_i == m for m in unique_planes if m != 0] #[x, 1, h, w] x is the planes number
            if len(planes) == 0:
                continue
            mask_planes = torch.stack(planes, dim=0) #torch.cat(planes, dim=0) #
            pw_groups_pred, mask_valid = self.select_points_groups(pred_depth_i[None, :, :, :], mask_planes) # [x, N, 3(x,y,z), 3(p1,p2,p3)]
            for j in range(unique_planes.numel()-1):
                mask_valid_j = mask_valid[j, :]
                pw_groups_pred_j = pw_groups_pred[j, :]
                loss_tmp, valid_angles = self.constrain_a_plane_loss(pw_groups_pred_j, mask_valid_j)
                valid_planes_num += valid_angles
                loss += loss_tmp
        loss /= (valid_planes_num + 1e-8)
        return loss


if __name__ == '__main__':
    import cv2
    vnl_loss = PWN_Planes_Loss(500.0, 500.0, (385, 513), xyz_mode='uvd')
    pred_depth = torch.rand([2, 1, 385, 513]).cuda()
    gt_depth = torch.rand([2, 1, 385, 513]).cuda()
    gt_depth[:, :, 3:20, 40:60] = 0
    mask_kp1 = pred_depth > 0.9
    mask_kp2 = pred_depth < 0.5
    mask = torch.zeros_like(gt_depth, dtype=torch.uint8)
    mask = 1*mask_kp1 + 2* mask_kp2
    #gt_depth = cv2.imread('/hardware/yifanliu/SUNRGBD/sunrgbd-meta-data/sunrgbd_test_depth/2.png', -1)
    #gt_depth = gt_depth[None, :, :, None]
    #pred_depth = gt_depth[:, :, ::-1, :]
    loss = vnl_loss(gt_depth, gt_depth, mask)
    print(loss)
