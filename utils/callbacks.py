
import tensorflow as tf
import tensorflow.keras.backend as K
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle


def get_callbacks(validation_generator, model, logdir):
    return get_image_callbacks(validation_generator, model, logdir) + \
           get_model_callbacks(model, logdir) + \
           get_plotting_callbacks(logdir)


def get_image_callbacks(validation_generator, model, logdir):
    # Reshape the image for the Summary API.
    inputs, outputs = validation_generator.__getitem__(0)
    if isinstance(inputs, list):
        noisy = inputs[0]  # validation noisy xy
    else:
        noisy = inputs
    if isinstance(outputs, list):
        target = outputs[1]  # validation target xy
    else:
        target = outputs

    frames, M, N = target.shape[1:-1]  # (25, 176, 132)

    def log_images(epoch, logs):
        prediction = model.predict(inputs)[1]

        # Creates a file writer for the log directory.
        file_writer = tf.summary.create_file_writer(logdir)

        # Using the file writer, log the reshaped image.
        def process(x):
            # return tf.image.flip_left_right(tf.image.flip_up_down(x))
            return x / K.max(x)

        with file_writer.as_default():
            tf.summary.image("Validation predict xy", process(tf.abs(prediction[:, frames // 2])), step=epoch)
            tf.summary.image("Validation noisy xy", process(tf.abs(noisy[:, frames // 2])), step=epoch)
            tf.summary.image("Validation target xy", process(tf.abs(target[:, frames // 2])), step=epoch)
            tf.summary.image("Validation predict xt", process(tf.abs(prediction[:, :, M // 2])), step=epoch)
            tf.summary.image("Validation noisy xt", process(tf.abs(noisy[:, :, M // 2])), step=epoch)
            tf.summary.image("Validation target xt", process(tf.abs(target[:, :, M // 2])), step=epoch)
            tf.summary.image("Validation predict yt", process(tf.abs(prediction[:, :, :, N // 2])), step=epoch)
            tf.summary.image("Validation noisy yt", process(tf.abs(noisy[:, :, :, N // 2])), step=epoch)
            tf.summary.image("Validation target yt", process(tf.abs(target[:, :, :, N // 2])), step=epoch)

    class GCCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()

    gc_callback = GCCallback()
    img_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_images)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)

    return [img_callback, gc_callback, tensorboard_callback]


def get_model_callbacks(model, logdir):
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=logdir + '/weights{epoch:03d}.tf',
        verbose=10,
        save_weights_only=True,
        save_freq='epoch')

    def optimizer_checkpoint_callback(epoch, logs=None):
        opt_weights = model.optimizer.get_weights()
        with open(f'{logdir}/optimizer.pkl', 'wb') as f:
            pickle.dump(opt_weights, f)

    opt_cp_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=optimizer_checkpoint_callback)

    return [cp_callback, opt_cp_callback]


class LossCallback(tf.keras.callbacks.Callback):
    def __init__(self, logdir):
        super().__init__()
        self.logdir = logdir
        self.csv_path = f'{self.logdir}/loss.csv'
        self.keys = ['loss', 'val_loss',
                     'crop_loss_rmse', 'val_crop_loss_rmse',
                     'crop_loss_abs_mse', 'val_crop_loss_abs_mse',
                     'crop_loss_abs_mae', 'val_crop_loss_abs_mae',
                     ]

    def on_train_begin(self, logs=None):
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
        else:
            self.df = pd.DataFrame(columns=['epoch'] + self.keys)

    def on_epoch_end(self, epoch, logs):
        # create the loss dict and update dataframe
        update_dict = {'epoch': epoch}
        for key in self.keys:
            update_dict[key] = logs.get(key)
        self.df = self.df.append(update_dict, ignore_index=True)

        # save csv
        self.df.to_csv(self.csv_path, index=False)

        # Plot train & val loss
        plt.figure()
        x = np.arange(0, len(self.df))
        plt.plot(x, self.df['loss'], label="train_loss")
        plt.plot(x, self.df['val_loss'], label="val_loss")
        plt.title(f"Training/Validation Loss Epoch {epoch}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'{self.logdir}/loss.png')
        plt.close()


def get_plotting_callbacks(logdir):
    loss_callback = LossCallback(logdir)
    return [loss_callback]
  
