
import merlintf
import tensorflow as tf


class complex_SE_coil_layer(tf.keras.layers.Layer):
    def __init__(self, coil_size, bottle_size=2, name='complex_SE_coil_layer'):
        super(complex_SE_coil_layer, self).__init__(name=name)
        self.coil_size = coil_size
        self.SE_coil = []
        self.SE_coil.append(tf.keras.layers.GlobalMaxPool3D(data_format='channels_last'))  # 2D tensor with shape (batch_size, coil)
        self.SE_coil.append(tf.keras.layers.Dense(bottle_size*2))
        self.SE_coil.append(tf.keras.layers.Dense(coil_size*2))  # real, imag
        self.SE_coil.append(tf.keras.layers.Activation('sigmoid'))

    def call(self, x):
        x_excitation = tf.concat((tf.math.real(x), tf.math.imag(x)), axis=-1)
        for op in self.SE_coil:
            x_excitation = op(x_excitation)
        x_excitation = tf.expand_dims(tf.expand_dims(tf.expand_dims(x_excitation, axis=1), axis=2), axis=3)
        x = tf.complex(tf.math.real(x) * x_excitation[:, :, :, :, :self.coil_size],
                       tf.math.imag(x) * x_excitation[:, :, :, :, self.coil_size:])
        return x


class KspNetAttention(tf.keras.Model):
    def __init__(self, output_filter=25, activation='ModReLU', use_bias=False, name='kspace network'):
        super().__init__(name=name)
        self.dilation_rate = 1
        self.use_bias = use_bias
        self.activation = activation
        self.conv_layer = merlintf.keras.layers.ComplexConv3D
        self.SE_coil = complex_SE_coil_layer

        self.Nw = []
        self.Nw.append(self.conv_layer(filters=32, kernel_size=(5, 5, 3), dilation_rate=self.dilation_rate,
                                       padding='same', use_bias=self.use_bias, activation=self.activation,
                                       data_format='channels_first'))
        self.Nw.append(self.SE_coil(coil_size=15))

        # second layer
        self.Nw.append(self.conv_layer(filters=8, kernel_size=(5, 5, 3), dilation_rate=self.dilation_rate,
                                       padding='same', use_bias=self.use_bias, activation=self.activation,
                                       data_format='channels_first'))
        self.Nw.append(self.SE_coil(coil_size=15))

        # output_filter depends on time dimension
        self.Nw.append(self.conv_layer(filters=output_filter, kernel_size=(3, 3, 3), dilation_rate=self.dilation_rate,
                                       padding='same', use_bias=self.use_bias, activation=None,
                                       data_format='channels_first'))

    def call(self, inputs):
        x = inputs
        for op in self.Nw:
            x = op(x)
        return x
      
