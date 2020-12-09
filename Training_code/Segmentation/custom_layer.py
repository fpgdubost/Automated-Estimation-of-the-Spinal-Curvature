from keras import backend as K
from keras.layers import Layer
from custom_initializer import *



class WholeLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(WholeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # initializer = RandomNormal(mean = 0.2, stddev = 0.05, )
        initializer = custom_inverse_normal(shape = (input_shape[0], input_shape[1]) )

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer= initializer,
                                      trainable= False)
        super(WholeLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
