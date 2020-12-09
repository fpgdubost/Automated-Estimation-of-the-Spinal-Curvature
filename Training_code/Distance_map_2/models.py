from keras import Input, Model
from keras.layers import Dense, Flatten, Reshape, Softmax, Lambda, add, Conv2D, UpSampling2D, merge, MaxPooling2D, Concatenate,Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.layers import concatenate
from keras.losses import mean_squared_error
from losses import dice_loss

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
    # gap = BatchNormalization()(gap)

    output = Dense(out_size, kernel_initializer='normal',activation=dense_activation)(gap) #softmax

    model = Model(input, output)

    model.summary()

    # Todo : define metrics
    if loss=='mean_squared_error':
        model.compile(optimizer='adadelta',loss=loss ,metrics=[mean_pred])

    return model


# Todo : probleme sur les pooling qui acceptent pas les tenseurs des concatenate
def smallRegNet(loss , conv_activation , dense_activation, image_shape, out_size):
    print('Create model ...')
    nfeat = 16
    dropout_rate = 0.2

    input = Input(shape=(None,None, 1))

    conv1 = Conv2D(nfeat,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same', input_shape = (image_shape[1],image_shape[2], 1))(input)
    conv1 = Dropout(dropout_rate)(conv1)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv2D(nfeat,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(conv1)
    conv2 = Dropout(dropout_rate)(conv2)
    #conv2 = BatchNormalization()(conv2)

    skip1 = Concatenate([input , conv2])
    pool1 = MaxPooling2D(pool_size =(2,2) , padding='same')(conv2)

    conv3 = Conv2D(nfeat*2,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(pool1)
    conv3 = Dropout(dropout_rate)(conv3)
    #conv3 = BatchNormalization()(conv3)

    conv6 = Conv2D(nfeat*2,(3,3),activation = conv_activation, kernel_initializer='normal',padding='same')(conv3)
    conv6 = Dropout(dropout_rate)(conv6)
    #conv6 = BatchNormalization()(conv6)

    skip2 = Concatenate([pool1,conv6])

    gap = GlobalAveragePooling2D()(conv6)
    gap = Dropout(dropout_rate)(gap)
    gap = BatchNormalization()(gap)

    output = Dense(out_size, kernel_initializer='normal',activation=dense_activation)(gap) #softmax

    model = Model(input, output)

    model.summary()

    # Todo : define metrics
    if loss=='mean_squared_error':
        model.compile(optimizer='adadelta',loss=loss ,metrics=[mean_pred])

    return model


def get_gpunet_bn_2(loss):

    print('... create model')
    nfeat = 16
    dropout_rate = 0.2

    input = Input(shape=(None,None,1))
    conv1 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(input)
    conv1 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv1)
    conv1 = Dropout(dropout_rate)(conv1)
    conv1 = BatchNormalization()(conv1)
    skip1 = concatenate([input,conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(skip1)


    conv2 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(pool1)
    conv2 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv2)
    conv2 = Dropout(dropout_rate)(conv2)
    conv2 = BatchNormalization()(conv2)
    skip2 = concatenate([pool1,conv2])



    conv6 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(skip2)
    conv6 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv6)
    conv6 = Dropout(dropout_rate)(conv6)
#    conv6 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1],axis=-1)
    conv7 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(up7)
    conv7 = Conv2D(1, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv7)


    model = Model(input=input, output= conv7)
    model.summary()

    if loss=='dice_loss' :
        model.compile(optimizer='adadelta',loss=dice_loss, metrics=[mean_pred])
    elif loss=='mean_squared_error':
        model.compile(optimizer='adadelta',loss=loss, metrics=[mean_pred])
    else:
        print('error during compile model, please select a valid loss')

    return model


def get_gpunet_bn(loss):

    print('... create model')
    dropout_rate = 0.2
    nfeat = 16

    input = Input(shape=(None,None,1))
    conv1 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(input)
    conv1 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    # conv1 = concatenate([input,conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(pool1)
    conv2 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    # conv2 = concatenate([pool1,conv2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(nfeat*4, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(pool2)
    conv3 = Conv2D(nfeat*4, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(nfeat*8, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(pool3)
    conv4 = Conv2D(nfeat*8, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)

    ######################
    conv5 = Conv2D(nfeat*4, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv4)
    conv5 = Conv2D(nfeat*4, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)


    up6 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3],axis=-1)
    conv6 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(up6)
    conv6 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv2],axis=-1)
    conv7 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(up7)
    conv7 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv7)
    # skip6 = concatenate([up6, conv6])
#    conv6 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv6)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv1],axis=-1)
    conv8 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(up8)
    conv8 = Conv2D(1, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv8)

    # gap = GlobalAveragePooling2D()(conv7)
    # gap = BatchNormalization()(gap)

    ## out1 = Dense(136, kernel_initializer='normal', activation=dense_activation)(gap)

    model = Model(input=input, output= conv8)
    model.summary()

    if loss=='dice_loss' :
        model.compile(optimizer='adadelta',loss=dice_loss, metrics=[mean_pred])
    elif loss=='mean_squared_error':
        model.compile(optimizer='adadelta',loss=loss, metrics=[mean_pred])
    elif loss=='dice_loss_total_variation':
        model.compile(optimizer='adadelta',loss=dice_loss_total_variation, metrics=[mean_pred])
    else:
        print('error during compile model, please select a valid loss')

    return model
