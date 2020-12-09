import keras.backend as K
import numpy as np

def custom_inverse_normal(shape, dtype = float):

    return K.variable(np.ones(shape) - np.random.normal(loc = 0, scale=1, size=shape), name=name)
