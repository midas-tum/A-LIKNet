import merlintf
import tensorflow as tf
from image_net import ComplexUNet_2Dt
from low_rank_net import LNet_xyt_Batch


def extract_patches(x):
    # x: (t, x, y, 1)
    nb, nx, ny, nc = x.shape
    # Divide it into 4*4=16 blocks
    # Calculate the size of overlapping based on the remainder of the length divided by 4
    b_x = nx % 4 + 4
    b_y = ny % 4 + 4
    patch_x = (nx + 3 * b_x) // 4
    patch_y = (ny + 3 * b_y) // 4
    patches = tf.image.extract_patches(x,
                                       sizes=[1, patch_x, patch_y, 1],
                                       strides=[1, patch_x - b_x, patch_y - b_y, 1],
                                       rates=[1, 1, 1, 1],
                                       padding="VALID")  # (nb, 4, 4, patch_x * patch_y)
    return patches, patch_x, patch_y


@tf.function
def extract_patches_inverse(original_x, patches):
    _x = tf.zeros(original_x.shape, dtype=original_x.dtype)
    _y = extract_patches(_x)[0]
    temp = tf.ones(shape=_y.shape, dtype=tf.complex64)
    grad = tf.gradients(_y, _x, grad_ys=tf.convert_to_tensor(temp, dtype=tf.complex64))[0]
    # Divide by grad, to "average" together the overlapping patches
    # otherwise they would simply sum up
    return tf.gradients(_y, _x, grad_ys=patches)[0] / grad


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



def get_complex_attention_CNN():
    return ComplexUNet_2Dt(dim='2Dt', filters=12, kernel_size_2d=(1, 5, 5), kernel_size_t=(3, 1, 1), downsampling='mp',
                           num_level=2, num_layer_per_level=1, activation_last=None)


class ComplexAttentionLSNet(tf.keras.Model):
    def __init__(self, name='ComplexAttentionImageLRNet'):
        super().__init__(name=name)

        # define CNN block (sparse regularizer)
        self.R = get_complex_attention_CNN()

        # define low rank operator D
        self.D = LNet_xyt_Batch(num_patches=80)

        # denoiser strength
        self.tau = Scalar(init=0.1, name='tau')

        # image branch weight
        self.p_weight = Scalar(init=0.5, name='p_weight')

    def reshape_patch(self, patch, patch_x, patch_y):
        # patch: (t, 4, 4, px*py)
        nb = patch.shape[0]
        patches_stacked = tf.reshape(patch, (nb, 4, 4, patch_x, patch_y))  # (25, 4, 4, px, py)
        patches_stacked = tf.expand_dims(tf.reshape(patches_stacked, (nb, 16, patch_x, patch_y)),
                                         axis=0)  # (1, 25, 16, px, py)
        return patches_stacked

    def reshape_xyt_patch(self, patch):
        # patch: (5, 5, 16, patch_x, patch_y)
        patches_stacked_split = tf.transpose(patch, (0, 2, 1, 3, 4))  # (5, 16, t, x, y)
        nb_t, nb_xy, pt, px, py = patches_stacked_split.shape
        patches_stacked_split = tf.reshape(patches_stacked_split, (nb_t * nb_xy, pt, px, py))  # (80, pt, px, py)
        return patches_stacked_split

    def split_time(self, x):
        nt = x.shape[1]
        split_num = 5  # temporal patch size
        interval = nt // split_num
        for i in range(split_num):
            x_interval = x[:, i * interval:(i + 1) * interval, :, :, :]
            if i == 0:
                x_new = x_interval
            else:
                x_new = tf.concat((x_new, x_interval), axis=0)
        return x_new

    def recover_time_sequence(self, x):
        # x: (5, 5, nx, ny, 1)
        nb = x.shape[0]
        for i in range(nb):
            x_i = x[i, :, :, :, :]  # (5, nx, ny, 1)
            if i == 0:
                new_x = x_i
            else:
                new_x = tf.concat((new_x, x_i), axis=0)
        x = tf.expand_dims(new_x, axis=0)
        return x

    def recover_xyt_to_xy_patch(self, patch):
        # patch: (80, 5, px, px)
        nb, pt, px, py = patch.shape
        patches_stacked_split = tf.reshape(patch, (5, 16, pt, px, py))  # (5, 16, t, x, y)
        patches_stacked_split = tf.transpose(patches_stacked_split, (0, 2, 1, 3, 4))  # (5, 5, 16, x, y)
        patches_stacked = self.recover_time_sequence(patches_stacked_split)  # (1, 25, 16, x, y)
        patches_stacked = tf.reshape(tf.squeeze(patches_stacked, axis=0), (25, 4, 4, px, py))
        patches = tf.reshape(patches_stacked, (25, 4, 4, px * py))
        return patches


    def call(self, image, num_iter):
        x = image

        # denoiser operation
        den = self.R(x)
        p = x - merlintf.complex_scale(self.tau(den), 1 / num_iter)

        # low rank operation (D)
        patches, patch_x, patch_y = extract_patches(tf.squeeze(x, axis=0))
        x_patches_stacked = self.reshape_patch(patches, patch_x, patch_y)  # (1, 25, 16, patch_x, patch_y)
        x_patches_stacked_split = self.split_time(x_patches_stacked)  # (5, 5, 16, patch_x, patch_y)
        x_patches_stacked_split = self.reshape_xyt_patch(x_patches_stacked_split)  # (80, 5, px, py)
        q_patches_stacked_split = self.D(x_patches_stacked_split)  # (80, 5, px, px)
        q_patches = self.recover_xyt_to_xy_patch(q_patches_stacked_split)
        q = extract_patches_inverse(tf.squeeze(x, axis=0), q_patches)
        q = tf.expand_dims(q, axis=0)

        # weighted combination
        weighted_p = self.p_weight(p)
        q_weight = 1.0 - tf.squeeze(self.p_weight.weight)
        weighted_q = merlintf.complex_scale(q, q_weight)
        x = weighted_p + weighted_q

        return x
