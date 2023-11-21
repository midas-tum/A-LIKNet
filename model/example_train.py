import os
import time
import utils
import tensorflow as tf
from A_LIKNet_model import A_LIKNet

# specify GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def create_random_real_tensor(shape):
    return tf.random.normal(shape=shape)

def create_random_complex_tensor(shape):
    return tf.complex(tf.random.normal(shape=shape), tf.random.normal(shape=shape))

def create_inputs(batch_size, nt, nx, ny, nc):
    masked_img = create_random_complex_tensor(shape=(batch_size, nt, nx, ny, 1))
    masked_kspace = create_random_complex_tensor(shape=(batch_size, nt, nx, ny, nc))
    mask = create_random_real_tensor(shape=(1, nt, 1, ny, 1))
    smaps = create_random_complex_tensor(shape=(1, 1, nx, ny, nc))

    kspace_label = create_random_complex_tensor(shape=(1, nt, nx, ny, nc))
    image_label = create_random_complex_tensor(shape=(batch_size, nt, nx, ny, 1))
    return [masked_img, masked_kspace, mask, smaps], [kspace_label, image_label]


if __name__ == '__main__':
    model = A_LIKNet(num_iter=8)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer, loss=utils.get_loss('loss_complex_mae_all_dim'), metrics=utils.get_metrics(),
                  run_eagerly=True)

    # initialize model to print model summary
    inputs, targets = create_inputs(batch_size=1, nt=25, nx=192, ny=156, nc=15)
    start = time.time()
    outputs = model.predict(inputs)
    end = time.time()
    print(end - start)
    print(model.summary())
  
