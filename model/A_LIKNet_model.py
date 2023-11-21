
import mri
import merlintf
import tensorflow as tf

from ksp_network import KspNetAttention
from info_share_layer import InfoShareLayer
from image_lowrank_net import Scalar, ComplexAttentionLSNet


def get_dc_layer():
    A = mri.MulticoilForwardOp(center=True)
    AH = mri.MulticoilAdjointOp(center=True)
    return merlintf.keras.layers.DCGD(A, AH)


class A_LIKNet(tf.keras.Model):
    def __init__(self, num_iter=8, name='A_LIKNet'):
        super().__init__(name=name)

        # unroll/iteration numbers
        self.S_end = num_iter

        # information sharing layers
        self.ISL = []
        # kspace networks
        self.KspNet = []
        # Image and low-rank block
        self.ImgLrNet = []
        # dc layer for image
        self.ImgDC = []

        self.ksp_dc_weight = Scalar(init=0.5)

        for i in range(self.S_end):
            self.ISL.append(InfoShareLayer())
            self.KspNet.append(KspNetAttention(output_filter=25))
            self.ImgLrNet.append(ComplexAttentionLSNet())
            self.ImgDC.append(get_dc_layer())

    def ksp_dc(self, kspace, mask, sub_ksp):
        masked_pred_ksp = merlintf.complex_scale(kspace, mask)
        scaled_pred_ksp = self.ksp_dc_weight(masked_pred_ksp)
        yu_weight = 1.0 - tf.squeeze(self.ksp_dc_weight.weight)
        scaled_sampled_ksp = merlintf.complex_scale(sub_ksp, yu_weight)
        other_points = merlintf.complex_scale(kspace, (1-mask))
        out = scaled_pred_ksp + scaled_sampled_ksp + other_points
        return out

    def update_xy(self, x, y, i, num_iter, constants):
        sub_y, mask, smap = constants

        # kspace network
        ksp_net = self.KspNet[i]
        y = ksp_net(y)  # (1, 25, 176, 132, 15)

        # image and low-rank network
        img_lr_net = self.ImgLrNet[i]
        x = img_lr_net(x, num_iter)  # (1, 25, 176, 132, 1)

        # dc operation
        y = self.ksp_dc(y, mask, sub_y)
        img_dc_layer = self.ImgDC[i]
        x = img_dc_layer([x] + list(constants))

        # information sharing
        info_share_layer = self.ISL[i]
        y, x = info_share_layer(y, x, mask, smap)

        return x, y

    def call(self, inputs):
        x, y, mask, smaps = inputs
        constants = inputs[1:]
        for i in range(self.S_end):
            x, y = self.update_xy(x, y, i, num_iter=self.S_end, constants=constants)
        return y, x
