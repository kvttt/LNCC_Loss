import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self,
                 win=None,
                 eps=1e-5,
                 use_cumsum=False,
                 signed=False,
                 safe_cumsum=True,
                 use_double=False,):

        self.win = 9 if win is None else win
        self.eps = eps
        self.use_cumsum = use_cumsum
        self.signed = signed
        self.safe_cumsum = safe_cumsum
        self.use_double = use_double

    def cumsum(self, I):
        n_dims = len(I.get_shape().as_list()) - 2
        assert n_dims in [2, 3], "volumes should be 2 to 3 dimensions. found: %d" % n_dims

        pad = self.win // 2

        if n_dims == 2:
            pad = tf.constant([
                [0, 0],
                [pad + 1, pad],
                [pad + 1, pad],
                [0, 0],
            ])
        else:
            pad = tf.constant([
                [0, 0],
                [pad + 1, pad],
                [pad + 1, pad],
                [pad + 1, pad],
                [0, 0],
            ])

        I_pad = tf.pad(I, paddings=pad, mode='CONSTANT', constant_values=0)

        I_cs_x = tf.cumsum(I_pad, axis=1)
        I_cs_xy = tf.cumsum(I_cs_x, axis=2)

        if n_dims == 2:
            x, y = I.get_shape().as_list()[1:3]
            I_win = I_cs_xy[:, self.win:, self.win:] \
                - I_cs_xy[:, self.win:, :y] \
                - I_cs_xy[:, :x, self.win:] \
                + I_cs_xy[:, :x, :y]

        else:
            x, y, z = I.get_shape().as_list()[1:4]
            I_cs_xyz = tf.cumsum(I_cs_xy, axis=3)
            I_win = I_cs_xyz[:, self.win:, self.win:, self.win:] \
                - I_cs_xyz[:, self.win:, self.win:, :z] \
                - I_cs_xyz[:, self.win:, :y, self.win:] \
                - I_cs_xyz[:, :x, self.win:, self.win:] \
                + I_cs_xyz[:, :x, :y, self.win:] \
                + I_cs_xyz[:, :x, self.win:, :z] \
                + I_cs_xyz[:, self.win:, :y, :z] \
                - I_cs_xyz[:, :x, :y, :z]

        return I_win

    def cumsum_safe(self, I):
        n_dims = len(I.get_shape().as_list()) - 2
        assert n_dims in [2, 3], "volumes should be 2 to 3 dimensions. found: %d" % n_dims

        pad = self.win // 2

        if n_dims == 2:
            pad = tf.constant([
                [0, 0],
                [pad + 1, pad],
                [pad + 1, pad],
                [0, 0],
            ])
        else:
            pad = tf.constant([
                [0, 0],
                [pad + 1, pad],
                [pad + 1, pad],
                [pad + 1, pad],
                [0, 0],
            ])

        I_pad = tf.pad(I, paddings=pad, mode='CONSTANT', constant_values=0)
        I_pad = tf.Variable(I_pad)

        if n_dims == 2:
            x, y = I.get_shape().as_list()[1:3]

            I_pad[:, self.win:, :].assign(I_pad[:, self.win:, :] - I_pad[:, :x, :])
            I_pad[:, :, self.win:].assign(I_pad[:, :, self.win:] - I_pad[:, :, :y])

            I_pad = tf.cumsum(I_pad, axis=1)[:, self.win:, :]
            I_pad = tf.cumsum(I_pad, axis=2)[:, :, self.win:]

            return I_pad

        else:
            x, y, z = I.get_shape().as_list()[1:4]
            I_pad[:, self.win:, :, :].assign(I_pad[:, self.win:, :, :] - I_pad[:, :x, :, :])
            I_pad[:, :, self.win:, :].assign(I_pad[:, :, self.win:, :] - I_pad[:, :, :y, :])
            I_pad[:, :, :, self.win:].assign(I_pad[:, :, :, self.win:] - I_pad[:, :, :, :z])

            I_pad = tf.cumsum(I_pad, axis=1)[:, self.win:, :, :]
            I_pad = tf.cumsum(I_pad, axis=2)[:, :, self.win:, :]
            I_pad = tf.cumsum(I_pad, axis=3)[:, :, :, self.win:]

            return I_pad

    def ncc_cumsum(self, Ii, Ji):
        ndims = len(Ii.get_shape().as_list()) - 2
        assert ndims in [2, 3], "volumes should be 2 to 3 dimensions. found: %d" % ndims

        if self.use_double:
            Ii = tf.cast(Ii, tf.float64)
            Ji = tf.cast(Ji, tf.float64)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

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

        # compute cross correlation
        win_size = self.win ** ndims
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = tf.maximum(cross, self.eps)
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = tf.maximum(I_var, self.eps)
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = tf.maximum(J_var, self.eps)

        if self.signed:
            cc = cross / tf.sqrt(I_var * J_var + self.eps)
        else:
            # cc = (cross * cross) / (I_var * J_var)
            cc = (cross / I_var) * (cross / J_var)

        return tf.cast(cc, tf.float32)

    def ncc(self, Ii, Ji):
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(Ii.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims
        elif not isinstance(self.win, list):  # user specified a single number not a list
            self.win = [self.win] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # compute filters
        in_ch = Ji.get_shape().as_list()[-1]
        # sum_filt = tf.ones([*self.win, in_ch, 1])
        sum_filt = tf.ones([*self.win, in_ch, 1], dtype=Ii.dtype)
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)

        # compute local sums via convolution
        padding = 'SAME'
        I_sum = conv_fn(Ii, sum_filt, strides, padding)
        J_sum = conv_fn(Ji, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win) * in_ch
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        # TODO: simplify this
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = tf.maximum(cross, self.eps)
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = tf.maximum(I_var, self.eps)
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = tf.maximum(J_var, self.eps)

        if self.signed:
            cc = cross / tf.sqrt(I_var * J_var + self.eps)
        else:
            # cc = (cross * cross) / (I_var * J_var)
            cc = (cross / I_var) * (cross / J_var)

        return cc

    def loss(self, y_true, y_pred, reduce='mean'):
        if self.use_cumsum:
            cc = self.ncc_cumsum(y_true, y_pred)
        else:
            cc = self.ncc(y_true, y_pred)
        # reduce
        if reduce == 'mean':
            cc = tf.reduce_mean(K.batch_flatten(cc), axis=-1)
        elif reduce == 'max':
            cc = tf.reduce_max(K.batch_flatten(cc), axis=-1)
        elif reduce is not None:
            raise ValueError(f'Unknown NCC reduction type: {reduce}')
        # loss
        return - cc


if __name__ == '__main__':
    a = tf.random.normal([1, 160, 160, 160, 1])
    b = tf.random.normal([1, 160, 160, 160, 1])

    t0 = tf.timestamp()
    for i in range(10):
        cs_safe = NCC(use_cumsum=True, safe_cumsum=True, use_double=True).loss(a, b)
    t1 = tf.timestamp()
    for i in range(10):
        cs = NCC(use_cumsum=True, safe_cumsum=False, use_double=True).loss(a, b)
    t2 = tf.timestamp()
    for i in range(10):
        no_cs = NCC(use_cumsum=False).loss(a, b)
    t3 = tf.timestamp()

    print(f'Safe cumsum impl: {t1 - t0} s.')
    print(f'Unsafe cumsum impl: {t2 - t1} s.')
    print(f'Original impl: {t3 - t2} s.')

    print(f'Safe cumsum vs original: {tf.abs(cs_safe - no_cs)}.')
    print(f'Unsafe cumsum vs original: {tf.abs(cs - no_cs)}.')

