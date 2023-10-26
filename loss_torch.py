import torch
import numpy as np
import torch.nn.functional as F
import math


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self,
                 win=None,
                 eps=1e-5,
                 use_cumsum=False,
                 use_double=False,
                 safe_cumsum=True,):
        """
        :param win: window size for local patch. Default: 9.
        :param eps: epsilon to avoid zero division. Default: 1e-5.
        :param use_cumsum: whether to use cumsum for acceleration. Default: False.
        :param use_double: whether to use double precision to prevent overflow. Default: False.
        :param safe_cumsum: whether to use the safe cumsum implementation to prevent overflow. Default: True.
        """

        self.win = 9 if win is None else win
        self.eps = eps
        self.use_cumsum = use_cumsum
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


    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        if self.use_double:
            Ii = Ii.double()
            Ji = Ji.double()

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        if not self.use_cumsum:
            # set window size
            if self.win is None:
                self.win = [9] * ndims
            elif not isinstance(self.win, list):  # user specified a single number not a list
                self.win = [self.win] * ndims

            win_size = np.prod(self.win)

            # compute filters
            sum_filt = torch.ones([1, 1, *self.win]).to("cuda")

            pad_no = math.floor(self.win[0] / 2)

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

            I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
            J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
            I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
            J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
            IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        else:
            if self.safe_cumsum:
                I_sum = self.cumsum_safe(Ii)
                J_sum = self.cumsum_safe(Ji)
                I2_sum = self.cumsum_safe(I2)
                J2_sum = self.cumsum_safe(J2)
                IJ_sum = self.cumsum_safe(IJ)
            else:
                I_sum = self.cumsum(Ii)
                J_sum = self.cumsum(Ji)
                I2_sum = self.cumsum(I2)
                J2_sum = self.cumsum(J2)
                IJ_sum = self.cumsum(IJ)

            win_size = self.win ** ndims

        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


if __name__ == '__main__':
    import time

    a = torch.randn(1, 1, 160, 192, 160).cuda()
    b = torch.randn(1, 1, 160, 192, 160).cuda()

    t0 = time.time()
    for i in range(10):
        cs_safe = NCC(use_cumsum=True, safe_cumsum=True, use_double=False).loss(a, b)
    t1 = time.time()
    for i in range(10):
        cs = NCC(use_cumsum=True, safe_cumsum=False, use_double=False).loss(a, b)
    t2 = time.time()
    for i in range(10):
        no_cs = NCC(use_cumsum=False, use_double=False).loss(a, b)
    t3 = time.time()

    print(f'Safe cumsum impl: {t1 - t0} s.')
    print(f'Unsafe cumsum impl: {t2 - t1} s.')
    print(f'Original impl: {t3 - t2} s.')

    print(f'Safe cumsum vs original: {torch.abs(cs_safe - no_cs)}.')
    print(f'Unsafe cumsum vs original: {torch.abs(cs - no_cs)}.')

