import keras
from keras import backend as K
import numpy as np
import tensorflow as tf

def dice_loss(y_true, y_pred):
    return 1. - dice(y_true, y_pred) + mean_squared_error_custom(y_true , y_pred )

def dice_loss_total_variation(y_true, y_pred):
    return 1. - dice(y_true, y_pred) + total_variation_loss(y_pred)


def total_variation_loss(x):
    IMAGE_HEIGHT = 1024
    IMAGE_WIDTH = 256
    print(IMAGE_WIDTH)
    print(IMAGE_HEIGHT)
    a = K.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, 1:, :IMAGE_WIDTH-1, :])
    b = K.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, :IMAGE_HEIGHT-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

def dice(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def mean_squared_error_custom(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
