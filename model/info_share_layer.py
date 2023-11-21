
import merlintf
import mri_no_mask
import tensorflow as tf


class Scalar(tf.keras.layers.Layer):
    def __init__(self, init=1.0, train_scale=1.0, name=None):
        super().__init__(name=name)
        self.init = init
        self.train_scale = train_scale

    def build(self, input_shape):
        self._weight = self.add_weight(name='scalar',
                                       shape=(1,),
                                       constraint=tf.keras.constraints.NonNeg(),
                                       initializer=tf.keras.initializers.Constant(self.init))

    @property
    def weight(self):
        return self._weight * self.train_scale

    def call(self, inputs):
        return merlintf.complex_scale(inputs, self.weight)


class InfoShareLayer(tf.keras.layers.Layer):
    def __init__(self, name='InfoShareLayer'):
        super().__init__(name=name)
        # two scalar for add operation
        self.tau_ksp = Scalar(init=0.5, name='ksp_weight')
        self.tau_img = Scalar(init=0.5, name='img_weight')

    def call(self, kspace, image, mask, smap):

        # transform image to kspace
        img_fft = mri_no_mask.MulticoilForwardOp(center=True)(image, mask, smap)
        ksp_1 = kspace
        ksp_2 = img_fft

        # weighted combined k-space
        weighted_ksp_1 = self.tau_ksp(ksp_1)
        ksp_2_weight = 1.0 - tf.squeeze(self.tau_ksp.weight)
        weighted_ksp_2 = merlintf.complex_scale(ksp_2, ksp_2_weight)
        new_ksp = weighted_ksp_1 + weighted_ksp_2

        # transform kspace to image
        ksp_ifft = mri_no_mask.MulticoilAdjointOp(center=True)(kspace, mask, smap)
        img_1 = image
        img_2 = ksp_ifft

        # weighted combined image
        weighted_img_1 = self.tau_img(img_1)
        img_2_weight = 1.0 - tf.squeeze(self.tau_img.weight)
        weighted_img_2 = merlintf.complex_scale(img_2, img_2_weight)
        new_img = weighted_img_1 + weighted_img_2

        return new_ksp, new_img
      
