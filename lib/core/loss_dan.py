import torch
from torch import nn
from lib.core.evaluate import SegmentationMetric


class L_exp_z(nn.Module):
    def __init__(self, patch_size):
        super(L_exp_z, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, x, mean_val):
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean - torch.FloatTensor([mean_val]).to(x.device), 2))
        return d


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class L_pair(nn.Module):

    def __init__(self, cfg, device):
        super().__init__()
        self.cfg= cfg
        self.BCEseg = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.])).to(device)

    def forward(self, pred_d, pred_n, shapes):
        drive_area_seg_pred_d = pred_d[1].reshape(-1)
        drive_area_seg_pred_n = pred_n[1].reshape(-1)
        lseg_da = self.BCEseg(drive_area_seg_pred_d, drive_area_seg_pred_n)

        lane_line_seg_pred_d = pred_d[2].reshape(-1)
        lane_line_seg_pred_n = pred_n[2].reshape(-1)
        lseg_ll = self.BCEseg(lane_line_seg_pred_d, lane_line_seg_pred_n)

        metric = SegmentationMetric(2)
        nb, _, height, width = pred_n[1].shape
        pad_w, pad_h = shapes[0][1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        _, lane_line_pred_d = torch.max(pred_d[2], 1)
        _, lane_line_pred_n = torch.max(pred_n[2], 1)
        lane_line_pred_d = lane_line_pred_d[:, pad_h:height - pad_h, pad_w:width - pad_w]
        lane_line_pred_n = lane_line_pred_n[:, pad_h:height - pad_h, pad_w:width - pad_w]
        metric.reset()
        metric.addBatch(lane_line_pred_d.cpu(), lane_line_pred_n.cpu())
        IoU = metric.IntersectionOverUnion()
        liou_ll = 1 - IoU

        lseg_da *= self.cfg.LOSS.DA_SEG_GAIN
        lseg_ll *= self.cfg.LOSS.LL_SEG_GAIN
        liou_ll *= self.cfg.LOSS.LL_IOU_GAIN

        loss = lseg_da + lseg_ll + liou_ll
        return loss


if __name__ == '__main__':
    x= torch.rand([2,3,32,32])
    y= torch.rand([2,3,32,32])
    loss= L_exp_z(16)
    ltv= L_TV()
    lssim= SSIM()
    print(loss(x,x.mean()))
    print(ltv(x))
    print(lssim(x,y))