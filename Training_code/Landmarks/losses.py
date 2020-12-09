import numpy as np
from keras import backend as K

def gaussian(x, mu, sig):
    return K.exp(-K.square(x - mu) / (2 * K.square(sig)))

def gaussian_loss(Y_true,Y_pred):
    diff = K.square(Y_true-Y_pred)
    shape = K.int_shape(error)
    shape[1]/=2
    error = K.zeros(shape)
    for i in range(shape[1]):
        error[:,i] = diff[:,2*i] + diff[:,2*i+1]
    error = K.sqrt(error)
    gaussian_error = 1 - gaussian(np.array(error),0,0.004)
    return K.sum(gaussian_error,axis=-1)/(float(shape[1]))
