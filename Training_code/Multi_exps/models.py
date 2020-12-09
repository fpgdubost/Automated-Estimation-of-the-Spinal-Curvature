from keras import Input, Model
from keras.layers import Dense, Flatten, Reshape, Softmax, Lambda, add, Conv2D, UpSampling2D, merge, MaxPooling2D, Concatenate,Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.layers import concatenate
from keras.losses import mean_squared_error
from losses import *

from keras.optimizers import Adam
from metrics import *

def smallRegNet2(loss , conv_activation , dense_activation, image_shape, out_size):
    print('Create model ...')
    nfeat = 16
    dropout_rate = 0.2

    input = Input(shape=(None,None, 3))

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

    output = Dense(out_size, kernel_initializer='normal',activation=dense_activation)(gap) #softmax

    model = Model(input, output)

    model.summary()

    # Todo : define metrics
    if loss=='mean_squared_error':
        model.compile(optimizer='adadelta',loss=loss ,metrics=[mean_pred])

    return model


# Todo : probleme sur les pooling qui acceptent pas les tenseurs des concatenate
def get_gpunet_bn():

    print('... create model')
    dropout_rate = 0.2
    nfeat = 16

    ################# FIRST NETWORK ##############################

    x = Input(shape=(None,None,1))
    conv1_1 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(x)
    conv1_1 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv1_1)
    conv1_1 = BatchNormalization()(conv1_1)
    # conv1 = concatenate([input,conv1])
    pool1_1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)

    conv1_2 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(pool1_1)
    conv1_2 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv1_2)
    conv1_2 = BatchNormalization()(conv1_2)
    # conv2 = concatenate([pool1,conv2])
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    conv1_3 = Conv2D(nfeat*4, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(pool1_2)
    conv1_3 = Conv2D(nfeat*4, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv1_3)
    conv1_3 = BatchNormalization()(conv1_3)
    pool1_3 = MaxPooling2D(pool_size=(2, 2))(conv1_3)

    conv1_4 = Conv2D(nfeat*4, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(pool1_3)
    conv1_4 = Conv2D(nfeat*4, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv1_4)
    conv1_4 = BatchNormalization()(conv1_4)
    pool1_4 = MaxPooling2D(pool_size=(2, 2))(conv1_4)

    conv1_5 = Conv2D(nfeat*4, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(pool1_4)
    conv1_5 = Conv2D(nfeat*4, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv1_5)
    conv1_5 = BatchNormalization()(conv1_5)
    pool1_5 = MaxPooling2D(pool_size=(2, 2))(conv1_5)

    conv1_6 = Conv2D(nfeat*4, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(pool1_5)
    conv1_6 = Conv2D(nfeat*4, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv1_6)
    conv1_6 = BatchNormalization()(conv1_6)


    ######################
    conv1_7 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv1_6)
    conv1_7 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv1_7)
    conv1_7 = BatchNormalization()(conv1_7)

    up1_8 = concatenate([UpSampling2D(size=(2, 2))(conv1_6), conv1_5],axis=-1)
    conv1_8 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(up1_8)
    conv1_8 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv1_8)

    up1_9 = concatenate([UpSampling2D(size=(2, 2))(conv1_8), conv1_4],axis=-1)
    conv1_9 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(up1_9)
    conv1_9 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv1_9)

    up1_10 = concatenate([UpSampling2D(size=(2, 2))(conv1_9), conv1_3],axis=-1)
    conv1_10 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(up1_10)
    conv1_10 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv1_10)

    up1_11 = concatenate([UpSampling2D(size=(2, 2))(conv1_10), conv1_2],axis=-1)
    conv1_11 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(up1_11)
    conv1_11 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv1_11)
    # skip6 = concatenate([up6, conv6])
    #    conv6 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv6)

    up1_12 = concatenate([UpSampling2D(size=(2, 2))(conv1_11), conv1_1],axis=-1)
    conv1_12 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(up1_12)
    y1_pred = Conv2D(1, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same', name='y1_pred')(conv1_12)


    ################# SECOND NETWORK ##############################


    conv2_1 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(y1_pred)
    conv2_1 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv2_1)
    conv2_1 = BatchNormalization()(conv2_1)
    # conv1 = concatenate([input,conv1])
    pool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)

    conv2_2 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(pool2_1)
    conv2_2 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv2_2)
    conv2_2 = BatchNormalization()(conv2_2)
    # conv2 = concatenate([pool1,conv2])
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

    conv2_3 = Conv2D(nfeat*4, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(pool2_2)
    conv2_3 = Conv2D(nfeat*4, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv2_3)
    conv2_3 = BatchNormalization()(conv2_3)
    pool2_3 = MaxPooling2D(pool_size=(2, 2))(conv2_3)

    conv2_4 = Conv2D(nfeat*8, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(pool2_3)
    conv2_4 = Conv2D(nfeat*8, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv2_4)
    conv2_4 = BatchNormalization()(conv2_4)
    # pool2_4 = MaxPooling2D(pool_size=(2, 2))(conv2_4)
    #
    # conv2_5 = Conv2D(nfeat*8, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(pool2_4)
    # conv2_5 = Conv2D(nfeat*8, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv2_5)
    # conv2_5 = BatchNormalization()(conv2_5)

    ######################
    # conv2_6 = Conv2D(nfeat*4, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv2_5)
    # conv2_6 = Conv2D(nfeat*4, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv2_6)
    # conv2_6 = BatchNormalization()(conv2_6)
    #
    # up2_7 = concatenate([UpSampling2D(size=(2, 2))(conv2_5), conv2_4],axis=-1)
    conv2_7 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv2_4)
    conv2_7 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv2_7)
    conv2_7 = BatchNormalization()(conv2_7)

    up2_8 = concatenate([UpSampling2D(size=(2, 2))(conv2_4), conv2_3],axis=-1)
    conv2_8 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(up2_8)
    conv2_8 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv2_8)

    up2_9 = concatenate([UpSampling2D(size=(2, 2))(conv2_8), conv2_2],axis=-1)
    conv2_9 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(up2_9)
    conv2_9 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv2_9)
    # skip6 = concatenate([up6, conv6])
    #    conv6 = Conv2D(nfeat*2, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(conv6)

    up2_9 = concatenate([UpSampling2D(size=(2, 2))(conv2_9), conv2_1],axis=-1)
    conv2_9 = Conv2D(nfeat, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same')(up2_9)
    y2_pred = Conv2D(1, (3, 3), activation = 'relu', kernel_initializer='normal', padding='same', name='y2_pred')(conv2_9)

    model = Model(input = x, output = [ y1_pred , y2_pred ])

    model.summary()

    # model.add_loss(loss_multiOut)

    print 'compile multiout model...'

    model.compile(optimizer='adadelta', loss={'y1_pred': loss_multiOut1 , 'y2_pred':loss_multiOut2} ,metrics=[mean_pred])

    return model
