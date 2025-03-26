
"""
dataset.py

Dataset loader for the in-vivo Cine MR dataset used in the A-LIKNet project.

⚠️ Note: The dataset used in this project is not publicly available due to institutional restrictions.
If you're trying to reproduce our results, please refer to the original paper or contact the authors.
You may also consider testing on publicly available datasets (e.g., OCMR, CMRxRecon).
"""


import tensorflow as tf
import pandas as pd
import numpy as np
import random
import merlintf


def FFT2c(image):  # image: (z, t, x, y, c)
    image = np.transpose(image, axes=[0, 1, 4, 2, 3])  # (z, t, c, x, y)
    axes = [3, 4]
    scale = np.sqrt(np.prod(image.shape[-2:]).astype(np.float64))
    kspace = merlintf.complex_scale(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=axes),
                                                                axes=axes), axes=axes), 1/scale)
    return np.transpose(kspace, axes=[0, 1, 3, 4, 2])


def IFFT2c(kspace):
    kspace = np.transpose(kspace, axes=[0, 1, 4, 2, 3])  # (z, t, c, x, y)
    axes = [3, 4]
    scale = np.sqrt(np.prod(kspace.shape[-2:]).astype(np.float64))
    image = merlintf.complex_scale(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=axes),
                                                                axes=axes), axes=axes), scale)
    return np.transpose(image, axes=[0, 1, 3, 4, 2])


class CINE2DDataset(tf.keras.utils.Sequence):
    """CINE data training set."""

    def __init__(self, min_R, max_R, mode='train', transform=None, shuffle=True):
        self.transform = transform
        self.mode = mode
        self.batch_size = 1
        self.min_R = min_R
        self.max_R = max_R

        if self.mode == 'train':
            data_set = pd.read_csv('CINE2D_train.csv')
        elif self.mode == 'val' or self.mode == 'test':
            data_set = pd.read_csv('CINE2D_val.csv')

        self.data_set = []
        self.shuffle = shuffle

        for i in range(len(data_set)):
            subj = data_set.iloc[i]
            fname = subj.filename
            nPE = subj.nPE
            num_slices = subj.SLICE_DIM

            # specify the slices
            minsl = 0
            maxsl = num_slices - 1
            assert minsl <= maxsl

            attrs = {'nPE': nPE, 'metadata': subj.to_dict()}
            self.data_set += [(fname, minsl, maxsl, attrs)]

    def __len__(self):
        return len(self.data_set)

    def on_epoch_end(self):
        """Updates indeces after each epoch"""
        self.indeces = np.arange(len(self.data_set))
        if self.shuffle == True:
            np.random.shuffle(self.indeces)

    def __getitem__(self, idx):
        fname, minsl, maxsl, attrs = self.data_set[idx]
        fname = fname.split('.')[0]

        # according to batchsize, random choose slices
        print('selecting slices as one batch...')
        slice_range = np.arange(minsl, maxsl + 1)
        slice_prob = np.ones_like(slice_range, dtype=float)
        slice_prob /= slice_prob.sum()
        slidx = list(np.sort(np.random.choice(
            slice_range,
            min(self.batch_size, maxsl + 1 - minsl),
            p=slice_prob,
            replace=False,
        )))

        # load normalized image: norm_img: (z, t, x, y, coil=1)
        norm_imgc = np.load('norm_img_%s.txt.npy' % fname)
        batch_imgc = norm_imgc[slidx]  # (batchsize=1, t, x, y, coil=1)

        # load coil-compressed averaged smap, (z, t=1, x, y, c)
        cc_smap = np.load('cc_smap_15_%s.txt.npy' % fname)
        batch_smaps = cc_smap[slidx]  # (batchsize=1, t=1, x, y, c)

        # load pre-generated VISTA sampling mask:
        p = batch_imgc.shape[3]
        R = random.randint(self.min_R, self.max_R)
        sd = random.randint(1, 20)
        mask = np.loadtxt("mask_VISTA_%dx%d_acc%d_%d.txt" % (p, 25, R, sd), dtype=int, delimiter=",")  # (y, t=25)
        mask = np.expand_dims(np.transpose(mask), (0, 2, 4))  # (1, t, 1, y, 1)

        # create batch k-space
        imgccoil = batch_imgc * batch_smaps  # (1, t, x, y, c)
        coilkspace = FFT2c(imgccoil)

        # apply mask
        masked_kspace = mask * coilkspace  # (1, t, x, y, c)
        masked_coilimg = IFFT2c(masked_kspace) 
        masked_img = np.expand_dims(np.sum(masked_coilimg * np.conj(batch_smaps), -1), axis=-1)

        masked_img = tf.cast(masked_img, tf.complex64)
        masked_kspace = tf.cast(masked_kspace, tf.complex64)
        mask = mask.astype(np.float64)
        kspace_label = tf.cast(coilkspace, tf.complex64)
        batch_imgc = tf.cast(batch_imgc, tf.complex64)
        input_smaps = tf.cast(batch_smaps, tf.complex64)

        print('inputs shape:', masked_img.shape, masked_kspace.shape, mask.shape, input_smaps.shape,
              kspace_label.shape, batch_imgc.shape)

        return [masked_img, masked_kspace, mask, input_smaps], [kspace_label, batch_imgc]
      
