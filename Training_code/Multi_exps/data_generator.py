# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:25:26 2018

@author: fcalvet
under GPLv3
"""
import matplotlib

matplotlib.use('Agg')

import numpy as np  # to manipulate the arrays
import keras  # to use the Sequence class

from matplotlib import pyplot as plt  # to create figures examples
import os  # to save figures

from image_augmentation import *


import random as rd


class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    Based on keras.utils.Sequence for efficient and safe multiprocessing
    idea from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    Needs on initialization:
        list_IDs: a list of ID which will be supplied to the ReadFunction to obtain
        params: a dictionnary of parameters, explanation supplied in the README
        batch_size: number of IDs used to generate a batch
        shuffle: if set to True, the list will be shuffled every time an epoch ends
        plotgenerator: number of times a part of a batch will be saved to disk as examples
    """

    def __init__(self, data, GT, params, paddingGT=0, batch_size=1, shuffle=True, plotgenerator=0):
        'Initialization'
        self.batch_size = batch_size
        self.X = data
        self.Y = GT
        self.paddingGT = paddingGT
        self.shuffle = shuffle
        self.plotgenerator = plotgenerator
        self.params = params
        self.plotedgenerator = 0  # counts the number of images saved
        if self.Y is not None:
            assert self.Y[0].shape[0] == self.X.shape[0], "ERROR: Data list and GT list must have the same length"
        self.list_len = self.X.shape[0]
        print 'Using datagenerator with {} basic images...'.format(self.list_len)
        if self.params["augmentation"]['augmentation_choices'][0] == True:
            print 'with basic transformations'
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        nb = int(np.floor(self.list_len / self.batch_size))
        assert nb != 0, 'Batch size too large, number of batches per epoch is zero'
        return nb

    def __getitem__(self, index):
        """
        Generate one batch of data by:
            generating a list of indexes which corresponds to a list of ID,
            use prepare_batch to prepare the batch
        """
        # Extract a batch sized sub-lists
        X_batch = self.X[self.indexes[index * self.batch_size:(index + 1) * self.batch_size]]
        if self.Y is not None:
            Y_batch_0 = self.Y[0][self.indexes[index * self.batch_size:(index + 1) * self.batch_size]]
            Y_batch_1 = self.Y[1][self.indexes[index * self.batch_size:(index + 1) * self.batch_size]]
            Y_batch = [Y_batch_0 , Y_batch_1]
        else:
            Y_batch = None
        X, Y = self.prepare_batch(X_batch, Y_batch)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.list_len)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.index = 0

    def prepare_batch(self, X, Y):
        """
        Prepare a bacth of data:
            creating a list of images and masks after having preprocessed (and possibly augmented them)
            saving a few examples to disk if required
        """
        X_augm = np.zeros(X.shape, dtype=float)
        Y_augm_0 = np.zeros(Y[0].shape, dtype=float)
        Y_augm_1 = np.zeros(Y[1].shape, dtype=float)

        Y_augm = [Y_augm_0 , Y_augm_1]

        for i in range(X.shape[0]):
            if Y is not None:
                X_augm[i], Y_augm[0][i] , Y_augm[1][i] = self.imaugment(X[i], Y[0][i] , Y[1][i])
            else:
                X_augm[i], a = self.imaugment(X[i], None, None)

        # self.save_images(X_augm, Y_augm, X, Y)

        # if self.paddingGT is not 0:
        #     Y_augm = Y_augm[:, self.paddingGT:-self.paddingGT, self.paddingGT:-self.paddingGT, self.paddingGT:-self.paddingGT, :]
        #
        # if self.Y is None:
        #     return X_augm, X_augm  # return X, X    return X, None
        else:
            return X_augm, Y_augm

    def imaugment(self, X, Y0 , Y1):
        """
        Preprocess the tuple (image,mask) and then apply if selected:
            augmentation techniques adapted from Keras ImageDataGenerator
            elastic deformation
        """
        if self.params["augmentation"]["augmentation_choices"][0] == True:
            X, Y0 , Y1 = randomTransform(X, Y0 , Y1 , self.params['augmentation']["random_transform"], self.params['augmentation']['save_folder'])
        return X, Y0 , Y1

    def save_images(self, X, Y, X_origin, Y_origin):
        """
        Save a png to disk (params["savefolder"]) to illustrate the data been generated
        predict: if set to True allows the saving of predicted images, remember to set Y and list_IDs as None
        the to_predict function can be used to reset the counter of saved images, this allows if shuffle is False to have the same order between saved generated samples and the predicted ones
        """

        if self.plotgenerator > self.plotedgenerator:
            '''
            Save augmented images for 3D (will save 10 slices from a single volume)
            '''
            if self.plotedgenerator == 0:
                os.makedirs(os.path.join(self.params["savefolder"], 'datagen_sample'))

            print("\nSaving datagen_sample...")

            # select random index
            index = rd.randrange(0, X.shape[0])

            # print original images and the result of augmentation
            Xto_print = X[index]
            X_origin_to_print = X_origin[index]

            steps = np.linspace(0, Xto_print.shape[2] - 1, num=10, dtype=np.int)

            channels = Xto_print.shape[3]

            plt.figure(figsize=(5 * channels + 2, 11), dpi=200)
            plt.suptitle(u'plot n°' + repr(self.plotedgenerator), fontsize=5)
            # print(X[1])
            for i in range(10):
                for j in range(channels):
                    im = X_origin_to_print[:, :, steps[i], j]
                    ax = plt.subplot(5, 4 * channels, (2 * channels) * i + j * 2 + 1)
                    plt.imshow(np.squeeze(im), cmap='gray')  # , vmin=0, vmax=1)
                    plt.axis(u'off')
                    pltname = u'original slice ' + str(steps[i])
                    fz = 5  # Works best after saving
                    ax.set_title(pltname, fontsize=fz)

                    im = Xto_print[:, :, steps[i], j]
                    ax = plt.subplot(5, 4 * channels, (2 * channels) * i + j * 2 + 2)
                    plt.imshow(np.squeeze(im), cmap='gray')  # , vmin=0, vmax=1)
                    plt.axis(u'off')
                    pltname = u'augm slice ' + str(steps[i])
                    fz = 5  # Works best after saving
                    ax.set_title(pltname, fontsize=fz)
            plt.savefig(
                os.path.join(self.params["savefolder"], 'datagen_sample', str(self.plotedgenerator) + '_im.png'))
            plt.close()

            if self.Y is not None:
                Yto_print = Y[index]
                Y_origin_to_print = Y_origin[index]
                channels = Yto_print.shape[3]

                plt.figure(figsize=(5 * channels + 2, 11), dpi=200)
                plt.suptitle(u'plot GT n°' + repr(self.plotedgenerator), fontsize=5)

                for i in range(10):
                    for j in range(channels):
                        im = Y_origin_to_print[:, :, steps[i], j]
                        ax = plt.subplot(5, 4 * channels, (2 * channels) * i + j + 1)
                        plt.imshow(np.squeeze(im), cmap='gray')  # , vmin=0, vmax=1)
                        plt.axis(u'off')
                        pltname = u"original slice " + str(steps[i])
                        fz = 5  # Works best after saving
                        ax.set_title(pltname, fontsize=fz)

                        im = Yto_print[:, :, steps[i], j]
                        ax = plt.subplot(5, 4 * channels, (2 * channels) * i + j + 2)
                        plt.imshow(np.squeeze(im), cmap='gray')  # , vmin=0, vmax=1)
                        plt.axis(u'off')
                        pltname = u"augm slice " + str(steps[i])
                        fz = 5  # Works best after saving
                        ax.set_title(pltname, fontsize=fz)
                plt.savefig(
                    os.path.join(self.params["savefolder"], 'datagen_sample', str(self.plotedgenerator) + '_GT.png'))
                plt.close()
            self.plotedgenerator += 1

    def to_predict(self):
        self.plotedgenerator = 0

    def flow(self, train_set_x, train_set_y, batch_size=1, shuffle=True):
        return self

    def next(self):
        self.index += 1
        return self.__getitem__(self.index)

    def __next__(self):
        self.index += 1
        return self.__getitem__(self.index)
