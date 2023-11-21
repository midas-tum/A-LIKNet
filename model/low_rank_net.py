
import tensorflow as tf

class LNet_xyt_Batch(tf.keras.layers.Layer):
    def __init__(self, num_patches, name='LowRankNet'):
        super().__init__(num_patches, name=name)
        # initialization of SVD threshold, default value = -2
        self.thres_coef = tf.Variable(tf.constant(-2, shape=(num_patches,), dtype=tf.float32), trainable=True)

    def low_rank(self, L):
        L_pre = L  # (nb, nt, nx*ny)
        S, U, V = tf.linalg.svd(L_pre)
        # s is a tensor of singular values. shape is [..., P]. (nb, nt)
        # u is a tensor of left singular vectors. shape is [..., M, P]. (nb, nt, nt)
        # v is a tensor of right singular vectors. shape is [..., N, P]. (nb, nx*ny, nt)

        # update the threshold
        # s[..., 0] is the largest value
        thres = tf.sigmoid(self.thres_coef) * S[:, 0]  # shape=(80,)
        thres = tf.expand_dims(thres, -1)  # shape=(80, 1)

        # Only keep singular values greater than thres
        S = tf.nn.relu(S - thres) + thres * tf.nn.sigmoid(S - thres)
        S = tf.linalg.diag(S)  # (nb, nt, nt)
        S = tf.dtypes.cast(S, tf.complex64)
        V_conj = tf.transpose(V, perm=[0, 2, 1])  # (nb, nt, nx*ny)
        V_conj = tf.math.conj(V_conj)
        # U: (nb, nt, nt), S: (nb, nt, nt)
        US = tf.linalg.matmul(U, S)  # (nb, nt, nt)
        L = tf.linalg.matmul(US, V_conj)  # (nb, nt, nx*ny)
        return L

    def call(self, inputs):
        # L0: zero-filled image
        # L1~Ln: previous reconstructed images
        image = inputs

        # compress coil dimension
        if len(inputs.shape) == 5:
            image = tf.squeeze(image, axis=-1)
        nb, nt, nx, ny = image.shape
        if nb is None:
            nb = 1
        L_pre = tf.reshape(image, shape=(nb, nt, nx*ny))
        L = self.low_rank(L_pre)

        L = tf.reshape(L, shape=(nb, nt, nx, ny))
        if len(inputs.shape) == 5:
            L = tf.expand_dims(L, axis=-1)  # (nb, nt, nx, ny, 1)

        return L
      
