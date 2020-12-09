from keras import Input, Model
from keras.layers import Dense, Flatten, Reshape, Softmax, Lambda, add, Conv2D, UpSampling2D, merge, MaxPooling2D, Concatenate,Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.layers import concatenate
from keras.losses import mean_squared_error

from keras.optimizers import Adam
from metrics import *

def smallRegNet2(loss , conv_activation , dense_activation, image_shape, out_size):
    print('Create model ...')
    nfeat = 16
    dropout_rate = 0.2

    input = Input(shape=(None,None, 1))

    conv1 = Conv2D(nfeat,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(input)
    conv1 = Dropout(dropout_rate)(conv1)

    conv2 = Conv2D(nfeat,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(conv1)
    conv2 = Dropout(dropout_rate)(conv2)
    #conv2 = BatchNormalization()(conv2)

    skip1 = concatenate([input , conv2],axis=-1)
    pool1 = MaxPooling2D(pool_size =(4,4) , padding='same')(skip1)

    conv3 = Conv2D(nfeat*2,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(pool1)
    conv3 = Dropout(dropout_rate)(conv3)
    #conv3 = BatchNormalization()(conv3)

    conv6 = Conv2D(nfeat*2,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(conv3)
    conv6 = Dropout(dropout_rate)(conv6)
    #conv6 = BatchNormalization()(conv6)

    skip2 = concatenate([pool1,conv6],axis=-1)

    pool1 = MaxPooling2D(pool_size =(4,4) , padding='same')(skip2)

    conv3 = Conv2D(nfeat*4,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(pool1)
    conv3 = Dropout(dropout_rate)(conv3)
    #conv3 = BatchNormalization()(conv3)

    conv6 = Conv2D(nfeat*4,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(conv3)
    conv6 = Dropout(dropout_rate)(conv6)
    #conv6 = BatchNormalization()(conv6)

    skip2 = concatenate([pool1,conv6],axis=-1)

    pool1 = MaxPooling2D(pool_size =(4,4) , padding='same')(skip2)

    conv3 = Conv2D(nfeat*8,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(pool1)
    conv3 = Dropout(dropout_rate)(conv3)
    #conv3 = BatchNormalization()(conv3)

    conv6 = Conv2D(nfeat*8,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(conv3)
    conv6 = Dropout(dropout_rate)(conv6)
    #conv6 = BatchNormalization()(conv6)

    skip2 = concatenate([pool1,conv6],axis=-1)

    pool1 = MaxPooling2D(pool_size =(4,4) , padding='same')(skip2)

    conv3 = Conv2D(nfeat*16,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(pool1)
    conv3 = Dropout(dropout_rate)(conv3)
    #conv3 = BatchNormalization()(conv3)

    conv6 = Conv2D(nfeat*16,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(conv3)
    conv6 = Dropout(dropout_rate)(conv6)
    #conv6 = BatchNormalization()(conv6)

    skip2 = concatenate([pool1,conv6],axis=-1)

    pool1 = MaxPooling2D(pool_size =(4,4) , padding='same')(skip2)

    conv3 = Conv2D(nfeat*32, (3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(pool1)
    conv3 = Dropout(dropout_rate)(conv3)
    #conv3 = BatchNormalization()(conv3)

    conv6 = Conv2D(nfeat*32,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(conv3)
    conv6 = Dropout(dropout_rate)(conv6)
    #conv6 = BatchNormalization()(conv6)

    skip2 = concatenate([pool1,conv6],axis=-1)

    gap = GlobalAveragePooling2D()(skip2)
    gap = Dropout(dropout_rate)(gap)
    gap = BatchNormalization()(gap)

    output = Dense(3, kernel_initializer='normal',activation=dense_activation)(gap) #softmax

    model = Model(input, output)

    model.summary()

    # Todo : define metrics
    if loss=='mean_squared_error':
        model.compile(optimizer='adadelta',loss=loss ,metrics=[mean_pred])

    return model



def smallRegNet(loss , conv_activation , dense_activation):
    print('Create model ...')
    nfeat = 16
    dropout_rate = 0.2

    input = Input(shape=(None,None, 1))

    conv1 = Conv2D(nfeat,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(input)
    conv1 = Dropout(dropout_rate)(conv1)
    conv1 = Conv2D(nfeat,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv2D(nfeat,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(conv1)
    conv2 = Conv2D(nfeat,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(conv2)
    conv2 = Dropout(dropout_rate)(conv2)
    #conv2 = BatchNormalization()(conv2)

    skip1 = concatenate([input , conv2])
    pool1 = MaxPooling2D(pool_size =(2,2) , padding='same')(conv2)

    conv3 = Conv2D(nfeat*2,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(pool1)
    conv3 = Conv2D(nfeat*2,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(conv3)
    conv3 = Dropout(dropout_rate)(conv3)
    #conv3 = BatchNormalization()(conv3)

    conv6 = Conv2D(nfeat*2,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(conv3)
    conv6 = Conv2D(nfeat*2,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(conv6)
    conv6 = Dropout(dropout_rate)(conv6)
    #conv6 = BatchNormalization()(conv6)

    skip2 = concatenate([pool1,conv6])

    gap = GlobalAveragePooling2D()(conv6)
    gap = Dropout(dropout_rate)(gap)
    gap = BatchNormalization()(gap)

    output = Dense(3, kernel_initializer='normal',activation=dense_activation)(gap) #softmax

    model = Model(input, output)

    model.summary()

    # Todo : define metrics
    if loss=='mean_squared_error':
        model.compile(optimizer='adadelta',loss=loss ,metrics=[mean_pred])

    return model


def get_gpunet_bn(loss):

    print('... create model')
    nfeat = 16
    dropout_rate = 0.1
    filter = (5,3)

    input = Input(shape=(None,None,1))

    conv1 = Conv2D(nfeat, filter, activation = 'relu', kernel_initializer='normal', padding='same')(input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(nfeat, filter, activation = 'relu', kernel_initializer='normal', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    skip1 = concatenate([input,conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(skip1)

    conv2 = Conv2D(nfeat*2, filter, activation = 'relu', kernel_initializer='normal', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(nfeat*2, filter, activation = 'relu', kernel_initializer='normal', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    skip2 = concatenate([pool1,conv2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(skip2)

    conv3 = Conv2D(nfeat*4, filter, activation = 'relu', kernel_initializer='normal', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(nfeat*4, filter, activation = 'relu', kernel_initializer='normal', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    skip3 = concatenate([pool2,conv3])
    pool3 = MaxPooling2D(pool_size=(2, 2))(skip3)

    conv5 = BatchNormalization()(pool3)

    gap = GlobalAveragePooling2D()(conv5)
    gap = BatchNormalization()(gap)

    out1 = Dense(3, kernel_initializer='normal', activation='relu')(gap)

    model = Model(input=input, output= out1)
    model.summary()

    if loss=='mean_squared_error':
        model.compile(optimizer='adadelta',loss=loss ,metrics=[SMAPE])
    elif loss=='mean_absolute_percentage_error':
        model.compile(optimizer='adadelta',loss=loss ,metrics=[SMAPE])
    elif loss=='mean_absolute_error':
        model.compile(optimizer='adadelta',loss=loss ,metrics=[SMAPE])
    else:
        print('error during compile model, please select a valid loss')

    return model
