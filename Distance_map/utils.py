import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import h5py
import os
from keras import models
from keras.models import model_from_json
from keras.models import load_model
from metrics import *
from PIL import Image


def loadImage(path, filename, contrasted = False):
    if (contrasted):
        imgpil = Image.open(path + filename)
        img = np.array(imgpil, dtype = 'float')
        imgpil_copy_dark = Image.open(path + filename)
        copy_img_dark = np.array(imgpil_copy_dark, dtype = 'float')
        imgpil_copy_light = Image.open(path + filename)
        copy_img_light = np.array(imgpil_copy_light, dtype = 'float')

        temporary_tab = np.zeros(img.shape, dtype = 'float')

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                copy_img_light[i][j] = copy_img_light[i][j] + 1
                temporary_tab[i][j] = np.log(copy_img_light[i][j])

        max = np.amax(temporary_tab)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                temporary_tab[i][j] = temporary_tab[i][j] / max
                copy_img_light[i][j] = temporary_tab[i][j] * 255

        max = np.amax(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                copy_img_dark[i][j] = copy_img_dark[i][j] / max
                copy_img_dark[i][j] = copy_img_dark[i][j] * 9
                temporary_tab[i][j] = np.exp(copy_img_dark[i][j])

        max = np.amax(temporary_tab)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                temporary_tab[i][j] = temporary_tab[i][j] / max
                copy_img_dark[i][j] = temporary_tab[i][j] * 255

        img = img/float(np.amax(img))
        copy_img_dark = copy_img_dark/float(np.amax(copy_img_dark))
        copy_img_light = copy_img_light/float(np.amax(copy_img_light))
        img_merge = np.zeros((img.shape[0], img.shape[1], 3))

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img_merge[i][j] = img[i][j]
                img_merge[i][j][1] = copy_img_dark[i][j]
                img_merge[i][j][2] = copy_img_light[i][j]

        return(np.array([img_merge]))

    else :
        imgpil = Image.open(path + filename)
        img = np.array(imgpil, dtype = 'float')
        img = img/float(np.amax(img))
        img = img.reshape(len(img), len(img[0]) , 1)
        return (np.array([img]))

def loadDistanceMap(path,dataframe):
    images=[]

    list_name = dataframe.values
    for name in list_name:
        name = name[0]
        name = name.split('.')[0] + ".png"
        images.append(mpimg.imread(path + name))
    return images

def plotDistanceMap(original_distance_map,predicted_distance_map, image, save_path):
    fig = plt.figure(figsize=(16,9))
    grid = plt.GridSpec(1, 3)
    ax_original = fig.add_subplot(grid[0,0])
    ax_predicted = fig.add_subplot(grid[0,2])
    ax_image = fig.add_subplot(grid[0,1])
    ax_original.imshow(original_distance_map,cmap='gray')
    ax_predicted.imshow(predicted_distance_map,cmap='gray')
    ax_image.imshow(image, cmap='gray')
    fig.savefig(save_path)
    plt.close()

def saveDistanceMap(distance_map, path, name):
    name = name.split('.')[0] + '.png'
    img = Image.fromarray(distance_map.astype('uint8'))
    img.save(path + name)

def loadModels(path):
    print('Loading models ...')
    models = []
    list_models_names = os.listdir(path)

    for model_name in list_models_names:
        json_file = open(os.path.join(path, model_name, 'model.json') , 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(os.path.join(path, model_name, 'best_weights.hdf5'))

        # model = load_model(os.path.join(path, model_name, 'best_weights.hdf5'))
        models.append(loaded_model)
    print(models)

    return models

"""
Function which allows to add null values in an image to modify its size
:param image: image whose size is wanted to be modified
:param above: number of lines to add above the image (in first line)
:param right: number of columns to add to the image on the right (in last column)
:param below: number of lines to add below the image (in last line)
:param left: number of columns to add to the image on the left (in first column)
:return image: modified image
"""
def addNullPixels(image, above, right, below, left):
    length = len(image)
    width = len(image[0])
    line_zeros = np.zeros((1,width), dtype=int)

    for i in range(above):
        image = np.append(line_zeros, image, axis=0)

    length += above
    column_zeros = np.zeros((length,1), dtype=int)

    for i in range(right):
        image = np.append(image, column_zeros, axis=1)

    width += right
    line_zeros = np.zeros((1,width), dtype=int)

    for i in range(below):
        image = np.append(image, line_zeros, axis=0)

    length += below
    column_zeros = np.zeros((length,1), dtype=int)

    for i in range(left):
        image = np.append(column_zeros, image, axis=1)

    return(image)

"""
Function which allows to delete pixels from an image to modify its size
:param image: image whose size is wanted to be modified
:param above: number of lines to remove above the image (in first line)
:param right: number of columns to remove to the image on the right (in last column)
:param below: number of lines to remove below the image (in last line)
:param left: number of columns to remove to the image on the left (in first column)
:return image: modified image
"""
def deletePixels(image, above, right, below, left):
    for i in range(above):
        image = np.delete(image, 0, axis=0)

    for i in range(right):
        image = np.delete(image, len(image[0])-1, axis=1)

    for i in range(below):
        image = np.delete(image, len(image)-1, axis=0)

    for i in range(left):
        image = np.delete(image, 0, axis=1)
    return(image)

def reformateImage(img, line_threshold, column_threshold, original_landmarks):
    nb_columns = len(img[0])
    nb_lines = len(img)

    line_threshold = 2776
    column_threshold = 954

    diff_lines = line_threshold - nb_lines
    if (diff_lines > 0):
        if (diff_lines%2 != 0):
            diff_below = (diff_lines - 1)/2
            diff_above = diff_below + 1
            img = addNullPixels(img, diff_above, 0, diff_below, 0)
        else :
            diff_below = diff_lines/2
            diff_above = diff_below
            img = addNullPixels(img, diff_above, 0, diff_below, 0)

    diff_columns = column_threshold - nb_columns
    if (diff_columns >= 0):
        if (diff_columns%2 != 0):
            diff_left = (diff_columns - 1)/2
            diff_right = diff_left + 1
            img = addNullPixels(img, 0, diff_right, 0, diff_left)
        else :
            diff_left = diff_columns/2
            diff_right = diff_left
            img = addNullPixels(img, 0, diff_right, 0, diff_left)

    else :
        diff_columns = -1 * diff_columns
        if (diff_columns%2 != 0):
            diff_left = (diff_columns - 1)/2
            diff_right = diff_left + 1
            img = deletePixels(img, 0, diff_right, 0, diff_left)
        else :
            diff_left = diff_columns/2
            diff_right = diff_left
            img = deletePixels(img, 0, diff_right, 0, diff_left)
        diff_left = -1 * diff_left
        diff_right = -1 * diff_right

    new_landmarks = resizeLabels([nb_columns, nb_lines], [diff_left, diff_right, diff_above, diff_below], original_landmarks)

    return(img, new_landmarks)

def resizeLabels(initial_image_size , cuts , landmarks_origin):
    list_x = []
    list_y = []

    for i in range(len(landmarks_origin)/2):
        list_x.append(landmarks_origin[2*i])
        list_y.append(landmarks_origin[2*i+1])
    list_x = [int(x*initial_image_size[0]) for x in list_x]
    list_y = [int(y*initial_image_size[1]) for y in list_y]

    if (cuts[0] + cuts[1] < 0):
        for i in range(len(list_x)):
            if list_x[i] > initial_image_size[0] + cuts[1]:
                return "Error label x > 1"
            elif list_x[i] < abs(cuts[0]):
                return "Error label x < 0"

    if (cuts[2] + cuts[3] < 0):
        for i in range(len(list_y)):
            if list_y[i] > initial_image_size[1] + cuts[3]:
                return "Error label y > 1"
            elif list_y[i] < abs(cuts[2]):
                return "Error label y < 0"
    # resize
    list_x = [x + cuts[0] for x in list_x]
    list_y = [y + cuts[2] for y in list_y]
    #change the shape of image
    new_image_size = [initial_image_size[0]+cuts[0]+cuts[1] , initial_image_size[1]+cuts[2]+cuts[3]]
    #resize
    list_x = [float(x)/float(new_image_size[0]) for x in list_x]
    list_y = [float(y)/float(new_image_size[1]) for y in list_y]

    new_landmarks = np.zeros(len(list_x) + len(list_y))

    for i in range(len(list_x)):
        new_landmarks[2*i]=list_x[i]
        new_landmarks[2*i+1]=list_y[i]
    return (new_landmarks)

def getValueFromJson(json_path,key):
    with open(json_path, 'r') as fp:
        data = json.load(fp)
        return data[key]
