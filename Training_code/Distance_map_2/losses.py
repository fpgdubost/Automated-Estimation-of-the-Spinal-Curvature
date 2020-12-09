import keras
from keras import backend as K
import numpy as np
import tensorflow as tf

def dice_loss(y_true, y_pred):
    return 1. - dice(y_true, y_pred) + mean_squared_error_custom(y_true , y_pred )

def dice(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    # y_true += 1
    # y_pred += 1
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def mean_squared_error_custom(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
