from utils import *
import numpy as np
from PIL import Image
from scipy import ndimage
from PIL import ImageFilter
from PIL.ImageFilter import (
    BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
    EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
    )

def randomTransform(X,Y, param, save_path):
    """
        params : image X , labels Y , a dictionnary that contains the transformation that you want to apply,  save path for images generated by the datagen
        function : apply randomly different transformations on the image and labels
        return : new image and landmarks, with transformation
    """

    if param['horizontal_switch'] and random.randint(0,1):
        # left go on right, right on left
        X = flipAxis(X, 1)
        Y = flipAxis(Y, 1)

    if param['width_shift_range'] and random.randint(0,1):
        X, Y = widthShiftRange(X, Y, 0)

    if param['height_shift_range'] and random.randint(0,1):
         X, Y = heightShiftRange(X, Y, 0)

    # if param['rotate'] and random.randint(0,1):
    #      X,Y = rotateImage(X,Y)

    ### uncomment the following 4 lines to save images ###
    # img = X
    # img = Image.fromarray(img.astype('uint8'))
    # img_name = getNextImgName(save_path)
    # img.save(save_name)

    return X,Y


def getNextImgName(path):
    """
        param : the path were images generated by the datagen are saved
        function : compute the name of a new image, according to those which already exist
        return : the name of the new image
    """
    list_img_names_with_extension = os.listdir(path)
    next_img_num = 1
    num = 1
    new_name = 'img_1.png'
    for name_with_extension in list_img_names_with_extension:
        name = name_with_extension.split(".")[0]
        num = int(name.split("_")[1])
        if num >= next_img_num:
            next_img_num = num +1
    new_name = 'img_'+str(next_img_num)+'.png'
    return new_name

def flipAxis(X, axis):
    """
        param : An image X, the axis around wich you want to flip X
        function : flip X around the axis
        return : the flipped image
    """
    X = X.swapaxes(axis,0)
    X = X[::-1, ...]
    X = X.swapaxes(0, axis)
    return X


def widthShiftRange(X, Y, convolution_size):
    """
        param : image X , labels Y , size of the convolution filter
        function : if X has column with only zero on its left and right borders, the function has a probability to move the image and labels on left or right
                   It takes into account a convolution_size by keeping at least convolution/2 columns with zeros on left and right
        return : new image and labels
    """
    border_size = convolution_size/2
    nb_black_columns = 0
    # move left
    if random.randint(0,1):
        #first we count the number of columns that contains only zeros on left
        left_pixels = X[:,border_size,0]
        while np.sum(left_pixels) == 0.0:
            nb_black_columns += 1
            left_pixels = X[:,border_size+nb_black_columns,0]
        # we move the image and labels up by a random integer between zero and number of black columns
        nb_move = random.randint(0,nb_black_columns)
        X = np.roll(X, -(nb_move*3))
        Y = np.roll(Y, -nb_move)
        # we replace left columns with zeros
        # X[:,len(X[0])-nb_move:] = 0.0
        X[:,len(X[0])-nb_move:,:3] = 0.0
        Y[:,len(Y[0])-nb_move:] = 0.0
    #move right
    else:
        # same but on right
        right_pixels = X[:,(len(X[0]))-1-border_size,0]
        while np.sum(right_pixels) == 0.0:
            nb_black_columns += 1
            right_pixels = X[:,len(X[0])-1-nb_black_columns-border_size,0]
        nb_move = random.randint(0,nb_black_columns)
        X = np.roll(X,nb_move*3)
        Y = np.roll(Y,nb_move)
        # X[:,0,:(nb_move*3)] = 0.0
        X[:,:nb_move,:3] = 0.0
        Y[:,:nb_move] = 0.0
    return X, Y

def heightShiftRange(X,Y, convolution_size):
    """
        param : image X , labels Y , size of the convolution filter
        function : if X has column with only zero on its top and bottom borders, the function has a probability to move the image and labels up or down
                   It takes into account a convolution_size by keeping at least convolution/2 rows with zeros on top and bottom
        return : new image and labels
    """
    border_size = convolution_size/2
    nb_black_rows = 0
    #move up
    if random.randint(0,1):
        #first we count the number of rows that contains only zeros
        up_pixels = X[border_size,:,0]
        while np.sum(up_pixels) == 0.0:
            nb_black_rows += 1
            up_pixels = X[border_size+nb_black_rows,:,0]
        # we move the image and labels up by a random integer between zero and number of black rows
        nb_move = random.randint(0,nb_black_rows)
        X = np.roll(X, -nb_move, axis = 0)
        Y = np.roll(Y, -nb_move, axis = 0)
        # we replace bottom rows with zeros
        # X[len(X)-nb_move:,:,0] = 0.0
        if nb_move != 0:
            X[len(X)-nb_move,:,:] = 0.0
            Y[len(Y)-nb_move:,:] = 0.0
    # move down
    else:
        # same but move down
        bottom_pixels = X[len(X)-1-border_size,:,0]
        while np.sum(bottom_pixels) == 0.0:
            nb_black_rows += 1
            bottom_pixels = X[len(X)-1-nb_black_rows-border_size,:,0]
        nb_move = random.randint(0,nb_black_rows)
        X = np.roll(X,nb_move, axis = 0)
        Y = np.roll(Y,nb_move, axis = 0)
        X[:nb_move,:,:] = 0.0
        Y[:nb_move,:] = 0.0
    return X, Y


def rotateImage(X, Y):
    """
        param : image X , labels Y
        function : Generate a random angle between -10 and 10 degrees, and rotate the image and its labels
        return : new image and labels
    """
    angle_degree = random.randint(-10,10)
    X = ndimage.rotate(X,180,reshape=False)
    Y = ndimage.rotate(Y,180,reshape=False)
    return X, Y


# X = np.array([[[5.2,0.0,0.0],[5.7,3.2,5.3],[5.6,7.3,1.1]],[[7.3,11.0,7.0],[2.0,5.0,6.0],[7.0,4.0,5.0]],[[0.0,18.0,7.0],[6.0,5.0,1.0],[3.0,4.0,7.0]]])
# Y = np.array([[[0],[0],[0],[0]],[[0],[0],[0],[0]],[[0],[0],[0],[0]]])
# print(X)
# X, Y =rotateImage(X,Y)
# print(X)


def getGTexcels(path_csv_test_set):
    test_df = pd.read_csv(path_csv_test_set+"/Sets/Test/test.csv")
    gt = pd.read_csv('../test_benj.csv')
    benj_df = pd.DataFrame(columns = ['Name','gt1','gt2','gt3'])

    for index, row in test_df.iterrows():
        for index2, row2 in gt.iterrows():
            if row['Name'] == row2['Name']:
                benj_df.loc[index,"Name"]=row['Name']
                benj_df.loc[index,"gt1"]=row2['0']
                benj_df.loc[index,"gt2"]=row2['1']
                benj_df.loc[index,"gt3"]=row2['2']
    benj_df.to_csv(r""+path_csv_test_set+"/gt.csv",index = None, header = True)


getGTexcels("../../../Results/Multi_exp/37_network=GpunetBn_loss=custom_loss_datagen=spinal2D_inputShape=reducedImages-1024_256_labelsShape=distanceMapWhite-12-1024_256")
