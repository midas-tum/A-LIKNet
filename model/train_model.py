
import os
import time
import datetime
import callbacks
import utils.utils
import tensorflow as tf
from A_LIKNet_model import A_LIKNet
from dataset import CINE2DDataset


def main(train_min_R, train_max_R, val_min_R, val_max_R):
    # dataset
    ds_train = CINE2DDataset(train_min_R, train_max_R, mode='train', shuffle=True)
    ds_val = CINE2DDataset(val_min_R, val_max_R, mode='val', shuffle=False)

    model = A_LIKNet(num_iter=8)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer, loss=utils.get_loss('loss_complex_mae_all_dim'), metrics=utils.get_metrics(),
                  run_eagerly=True)

    # initialize model to print model summary
    inputs, targets = ds_train.__getitem__(0)

    start = time.time()
    outputs = model.predict(inputs)
    end = time.time()
    print(end - start)
    print(model.summary())

    fold = 1
    exp_dir = './experiments/fold__%d__/R__%d__%d' % (fold, train_min_R, train_max_R)
    log_dir = os.path.join(exp_dir, model.name + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    history = model.fit(ds_train, epochs=200, validation_data=ds_val, max_queue_size=1,
                        callbacks=callbacks.get_callbacks(ds_val, model, log_dir))


if __name__ == '__main__':
    main(train_min_R=2, train_max_R=24, val_min_R=12, val_max_R=12)
  
