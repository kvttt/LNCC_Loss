import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np


class NCC(nn.Module):
    """
    Normalized cross correlation loss (reimplemented using cumsum for acceleration).
    """

    def __init__(self, win=9, eps=1e-5, use_double=True, safe_cumsum=True):
        """
        :param win: window size for local patch. Default: 9.
        :param eps: epsilon to avoid zero division. Default: 1e-5.
        :param use_double: whether to use double precision to prevent overflow. Default: False.
        :param safe_cumsum: whether to use the safe cumsum implementation to prevent overflow. Default: True.
        """

        super(NCC, self).__init__()

        self.win = 9 if win is None else win
        self.eps = eps
        self.use_double = use_double
        self.safe_cumsum = safe_cumsum

    def cumsum(self, I):
        """
        Compute the sum within each sliding window using cumsum.
        :param I: input tensor
        :return: window-wise sum (same size as the input tensor)
        """

        # Get dimension of volume
        n_dims = len(list(I.size())) - 2
        assert n_dims in (2, 3), 'Input tensor has to be 2D or 3D.'

        # Compute padding
        pad = self.win // 2
        pad = [pad + 1, pad] * n_dims

        # Pad input tensor
        I_pad = F.pad(I, pad=pad, mode='constant', value=0)

        if n_dims == 3:
            I_cs_xyz = I_pad.cumsum(2).cumsum(3).cumsum(4)
            x, y, z = I.shape[2:]
            I_win = I_cs_xyz[:, :, self.win:, self.win:, self.win:] \
                - I_cs_xyz[:, :, self.win:, self.win:, :z] \
                - I_cs_xyz[:, :, self.win:, :y, self.win:] \
                - I_cs_xyz[:, :, :x, self.win:, self.win:] \
                + I_cs_xyz[:, :, :x, :y, self.win:] \
                + I_cs_xyz[:, :, :x, self.win:, :z] \
                + I_cs_xyz[:, :, self.win:, :y, :z] \
                - I_cs_xyz[:, :, :x, :y, :z]

        else:
            I_cs_xy = I_pad.cumsum(2).cumsum(3)
            x, y = I.shape[2:]
            I_win = I_cs_xy[:, :, self.win:, self.win:] \
                - I_cs_xy[:, :, self.win:, :y] \
                - I_cs_xy[:, :, :x, self.win:] \
                + I_cs_xy[:, :, :x, :y]

        return I_win

    def cumsum_safe(self, I):
        """
        Compute the sum within each sliding window using cumsum.
        :param I: input tensor
        :return: window-wise sum (same size as the input tensor)
        """

        # Get dimension of volume
        n_dims = len(list(I.size())) - 2
        assert n_dims in (2, 3), 'Input tensor has to be 2D or 3D.'

        # Compute padding
        pad = self.win // 2
        pad = [pad + 1, pad] * n_dims

        # Pad input tensor
        I_pad = F.pad(I, pad=pad, mode='constant', value=0)

        if n_dims == 3:
            x, y, z = I.shape[2:]
            I_pad_clone = I_pad.clone()  # Cloning to prevent in-place operation
            I_pad[:, :, self.win:, :, :] -= I_pad_clone[:, :, :x, :, :]
            I_pad_clone = I_pad.clone()
            I_pad[:, :, :, self.win:, :] -= I_pad_clone[:, :, :, :y, :]
            I_pad_clone = I_pad.clone()
            I_pad[:, :, :, :, self.win:] -= I_pad_clone[:, :, :, :, :z]

            return I_pad.cumsum(2)[:, :, self.win:, :, :].cumsum(3)[:, :, :, self.win:, :].cumsum(4)[:, :, :, :, self.win:]

        else:
            x, y = I.shape[2:]
            I_pad_clone = I_pad.clone()
            I_pad[:, :, self.win:, :] -= I_pad_clone[:, :, :x, :]
            I_pad_clone = I_pad.clone()
            I_pad[:, :, :, self.win:] -= I_pad_clone[:, :, :, :y]

            return I_pad.cumsum(2)[:, :, self.win:, :].cumsum(3)[:, :, :, self.win:]

    def forward(self, I, J):
        """
        :param I: input tensor of shape (B, C, H, W) or (B, C, D, H, W).
        :param J: input tensor of shape (B, C, H, W) or (B, C, D, H, W).
        :return: NCC loss.
        """

        # Get dimension of volume
        n_dims = len(list(I.size())) - 2
        assert n_dims in (2, 3), 'Input tensor has to be 2D or 3D.'

        # Optionally convert input to double precision
        if self.use_double:
            I = I.double()
            J = J.double()

        # Element wise product
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # Window-wise sum
        if self.safe_cumsum:
            I_sum = self.cumsum_safe(I)
            J_sum = self.cumsum_safe(J)
            I2_sum = self.cumsum_safe(I2)
            J2_sum = self.cumsum_safe(J2)
            IJ_sum = self.cumsum_safe(IJ)
        else:
            I_sum = self.cumsum(I)
            J_sum = self.cumsum(J)
            I2_sum = self.cumsum(I2)
            J2_sum = self.cumsum(J2)
            IJ_sum = self.cumsum(IJ)

        # Window-wise average
        win_size = self.win ** n_dims
        mu_I = I_sum / win_size
        mu_J = J_sum / win_size

        cross = IJ_sum - mu_J * I_sum - mu_I * J_sum + mu_I * mu_J * win_size
        I_var = I2_sum - 2 * mu_I * I_sum + mu_I * mu_I * win_size
        J_var = J2_sum - 2 * mu_J * J_sum + mu_J * mu_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        return - torch.mean(cc).float()


class NCC_vxm(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    (https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/losses.py)
    """

    def __init__(self, win=None, n_channel=1):
        super(NCC_vxm, self).__init__()
        self.win = win
        self.n_channel = n_channel

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, self.n_channel, *win]).to(y_true.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return - torch.mean(cc)


if __name__ == '__main__':
    import torch

    a = torch.randn(1, 1, 160, 192, 160).cuda()
    b = torch.randn(1, 1, 160, 192, 160).cuda()
    ncc_vxm = NCC_vxm(n_channel=1)
    ncc = NCC(safe_cumsum=False)
    ncc_safe = NCC(safe_cumsum=True)

    l = ncc(a, b)
    # l_vxm = 0
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     profile_memory=True,
    #     with_flops=True,
    # ) as p_vxm:
    #     for i in range(10):
    #         l_vxm += ncc_vxm(a, b)
    #
    # l = 0
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     profile_memory=True,
    #     with_flops=True,
    # ) as p:
    #     for i in range(10):
    #         l += ncc(a, b)
    #
    # l_safe = 0
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     profile_memory=True,
    #     with_flops=True,
    # ) as p_safe:
    #     for i in range(10):
    #         l_safe += ncc_safe(a, b)
    #
    # print(f'NCC_vxm: {l_vxm}')
    # print(f'NCC: {l}')
    # print(f'NCC_safe: {l_safe}')
    #
    # with open('./ncc_vxm.txt', 'w') as f:
    #     f.write(p_vxm.key_averages().table(sort_by='cuda_time_total', row_limit=-1))
    #
    # with open('./ncc.txt', 'w') as f:
    #     f.write(p.key_averages().table(sort_by='cuda_time_total', row_limit=-1))
    #
    # with open('./ncc_safe.txt', 'w') as f:
    #     f.write(p_safe.key_averages().table(sort_by='cuda_time_total', row_limit=-1))
