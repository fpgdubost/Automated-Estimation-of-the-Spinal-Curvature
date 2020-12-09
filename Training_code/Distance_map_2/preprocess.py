import pandas as pd
import matplotlib
from PIL import Image
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from distance_map import *
from utils import *
import os
import os
from distance_map import *
from utils import *
from scipy import ndimage

"""
Function which allows to check if all images are the same size
:param images: list of images whose we want to check size
:return bool: True if all images are the same size
:return nb_lines_images: list which contains each line number for each image
:return nb_columns_images: list which contains each column number for each image
"""
def checkSizeImages(images):

    nb_lines_images = []
    nb_columns_images = []

    for img in images:
        nb_lines_images.append(len(img))
        nb_columns_images.append(len(img[0]))

    bool = True
    nb_images = len(nb_lines_images)
    nb_lines_check = nb_lines_images[0]
    nb_columns_check = nb_columns_images[0]
    counter = 0

    while (bool and counter < nb_images-1 and nb_lines_check == nb_lines_images[counter] and nb_columns_check == nb_columns_images[counter]):
        counter += 1
        if (nb_lines_check != nb_lines_images[counter] or nb_columns_check != nb_columns_images[counter]):
            bool = False

    return(bool, nb_lines_images, nb_columns_images)

"""
Function which allows to plot two histograms where you can see the used sizes in data set
:param nb_lines_images: list which contains each line number for each image
:param nb_columns_images: list which contains each column number for each image
:return: save two histograms
"""
def plotHistogramSizeImages(nb_lines_images, nb_columns_images):
    nb_images = len(nb_lines_images)
    plt.hist(nb_lines_images, bins = nb_images)
    plt.xlim(0, 500)
    plt.ylim(0, 30)
    plt.savefig("Histogram_lines_images")
    plt.close()
    plt.hist(nb_columns_images, bins = nb_images)
    plt.xlim(0, 200)
    plt.ylim(0, 30)
    plt.savefig("Histogram_columns_images")

    return(0)

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

def lookMaxSizeImages(name_images, images, save_path, nb_lines_images, nb_columns_images, line_threshold, column_threshold):
    save_path = os.path.join(save_path, 'maxSizesImages')
    os.makedirs(save_path)
    for count in range(len(nb_lines_images)):
        if (nb_lines_images[count] >= line_threshold and nb_columns_images[count] >= column_threshold):
            name_image = name_images[count] + '.png'
            path_image = os.path.join(save_path, name_image)
            image = Image.fromarray(images[count]) # Transformation du tableau en image PIL
            image.save(path_image)

        elif (nb_lines_images[count] >= line_threshold):
            name_image = name_images[count] + '.png'
            path_image = os.path.join(save_path, name_image)
            image = Image.fromarray(images[count]) # Transformation du tableau en image PIL
            image.save(path_image)

        elif (nb_columns_images[count] >= column_threshold):
            name_image = name_images[count] + '.png'
            path_image = os.path.join(save_path, name_image)
            image = Image.fromarray(images[count]) # Transformation du tableau en image PIL
            image.save(path_image)
    return(0)


def maxWidthImage(path):
    max_nb_columns = 0
    max_nb_lines = 0
    for image in os.listdir(path):
        img = mpimg.imread(os.path.join(path,image))
        nb_columns = len(img[0])
        nb_lines = len(img)
        if (nb_columns > max_nb_columns):
            max_nb_columns = nb_columns
            maxSizeImage = img
            nameImage = image
        if (nb_lines > max_nb_lines):
            max_nb_lines = nb_lines
    print("The maximum number of lines is :")
    print(max_nb_lines)
    print("The maximum number of columns is :")
    print(max_nb_columns)
    return(nameImage, maxSizeImage)

def reformateMaxWidthImage(name_image, path):
    image = mpimg.imread(os.path.join('maxSizesImages', name_image))
    delete = len(image[0])/6
    image = deletePixels(image, 0, delete, 0, delete)
    save_path = os.path.join(path, 'reshapedImages')
    os.makedirs(save_path)
    path_image = os.path.join(save_path, name_image)
    image = Image.fromarray(image) # Transformation du tableau en image PIL
    image.save(path_image)
    print(" The new number of columns is :")
    print(len(image[0]))
    return (image)

def reformateDataSet(line_threshold, column_threshold, source_path, save_new_images_path, labels_path, save_new_labels_path):
    index = 0
    dataframe_labels = pd.read_csv(labels_path)
    init_table = np.zeros([dataframe_labels.shape[0], dataframe_labels.shape[1]])
    dataframe_new_labels = pd.DataFrame(init_table)
    length_df = dataframe_new_labels.shape[1]

    diff_above = 0
    diff_below = 0
    diff_left = 0
    diff_right = 0

    for image in os.listdir(source_path):
        print(image)
        print("Index :")
        print(index)
        imgpil = Image.open(os.path.join(source_path,image))
        img = np.array(imgpil)
        nb_columns = len(img[0])
        nb_lines = len(img)

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

        else :
            diff_lines = -1 * diff_lines
            if (diff_lines%2 != 0):
                diff_above = (diff_lines - 1)/2
                diff_below = diff_above + 1

                img = deletePixels(img, diff_above, 0, diff_below, 0)

            else :
                diff_above = diff_lines/2
                diff_below = diff_above

                img = deletePixels(img, diff_above, 0, diff_below, 0)

            diff_above = -1 * diff_above
            diff_below = -1 * diff_below

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

                if (image == "sunhl-1th-06-Jan-2017-178 B AP.jpg"):
                    diff_right += diff_left
                    diff_left = 0

                elif (image == "sunhl-1th-05-Jan-2017-167 A AP.jpg"):
                    diff_right += diff_left
                    diff_left = 0

                elif (image == "sunhl-1th-09-Jan-2017-204 A AP.jpg"):
                    diff_right += diff_left
                    diff_left = 0

                elif (image == "sunhl-1th-09-Jan-2017-207 B AP.jpg"):
                    diff_right += diff_left
                    diff_left = 0

                elif (image == "sunhl-1th-20-Jul-2016-2 B AP.jpg"):
                    diff_left = 18
                    diff_right = 9

                elif (image == "sunhl-1th-21-Jul-2016-15 E AP.jpg"):
                    diff_left += diff_right
                    diff_right = 0

                elif (image == "sunhl-1th-26-Jul-2016-86 A AP.jpg"):
                    diff_right += diff_left
                    diff_left = 0

                img = deletePixels(img, 0, diff_right, 0, diff_left)

            else :
                diff_left = diff_columns/2
                diff_right = diff_left

                # CAS PARTICULIER COLONNE VERTEBRALE TRES A GAUCHE
                if (image == "sunhl-1th-06-Jan-2017-178 B AP.jpg"):
                    diff_right += diff_left
                    diff_left = 0

                elif (image == "sunhl-1th-05-Jan-2017-167 A AP.jpg"):
                    diff_right += diff_left
                    diff_left = 0

                elif (image == "sunhl-1th-09-Jan-2017-204 A AP.jpg"):
                    diff_right += diff_left
                    diff_left = 0

                elif (image == "sunhl-1th-09-Jan-2017-207 B AP.jpg"):
                    diff_right += diff_left
                    diff_left = 0

                elif (image == "sunhl-1th-20-Jul-2016-2 B AP.jpg"):
                    diff_left = 18
                    diff_right = 9

                elif (image == "sunhl-1th-21-Jul-2016-15 E AP.jpg"):
                    diff_left += diff_right
                    diff_right = 0

                elif (image == "sunhl-1th-26-Jul-2016-86 A AP.jpg"):
                    diff_right += diff_left
                    diff_left = 0

                img = deletePixels(img, 0, diff_right, 0, diff_left)

            diff_left = -1 * diff_left
            diff_right = -1 * diff_right

        result_function = resizeLabels([nb_columns, nb_lines], [diff_left, diff_right, diff_above, diff_below], dataframe_labels.iloc[index])

        if (type(result_function) == str):
            print(result_function)
            print("Error during resizing : one or more labels have been lost")
            print(image)
            return(1)

        for count in range(length_df):
            dataframe_new_labels.iloc[index].iloc[count] = result_function.iloc[count].iloc[0]
        index = index + 1
        path_image = os.path.join(save_new_images_path, image)
        img = Image.fromarray(img.astype('uint8'))
        img.save(path_image)
    dataframe_new_labels.to_csv(save_new_labels_path, index = None, header = True)
    return(0)

def resizeLabels(initial_image_size , cuts , df):
    list_x = []
    list_y = []
    print("resize_labels")
    for i in range(len(df)/2):
        list_x.append(df.iloc[i])
        list_y.append(df.iloc[i+len(df)/2])
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

    init_table = np.zeros(len(list_x) + len(list_y))
    df = pd.DataFrame(init_table)
    for i in range(len(list_x)):
        df.iloc[i]=list_x[i]
        df.iloc[i+len(list_x)]=list_y[i]
    return (df)

def sortSizeImages(nb_lines, nb_columns):
    length = len(nb_lines)
    for i in range(length) :
        for j in range(0, length-i-1) :
            if (nb_lines[j] > nb_lines[j+1]):
                nb_lines[j], nb_lines[j+1] = nb_lines[j+1], nb_lines[j]
            if (nb_columns[j] > nb_columns[j+1]):
                nb_columns[j], nb_columns[j+1] = nb_columns[j+1], nb_columns[j]
    print("La liste des nombres de lignes est :")
    print(nb_lines)
    print("La liste des nombres de colonnes est :")
    print(nb_columns)
    return(nb_lines, nb_columns)

def resizeProportionnalDataSet(height_min, width_min, source_path, save_new_images_path):

    index = 0

    for image in os.listdir(source_path):
        print(image)
        print("Index")
        print(index)
        image_path = os.path.join(source_path, image)
        img = Image.open(image_path)
        initial_width_image, initial_height_image = img.size
        height_quotient = float(initial_height_image) / float(height_min)
        width_quotient = float(initial_width_image) / float(width_min)

        if (height_quotient >= width_quotient):
            img = img.resize((width_min, int(initial_height_image/width_quotient)), Image.ANTIALIAS)
        else :
            img = img.resize((int(initial_width_image/height_quotient), height_min), Image.ANTIALIAS)

        index = index + 1
        path_image = os.path.join(save_new_images_path, image)
        img.save(path_image)
    return(0)

def createDistanceMap():
    path_image_name = "../../../DATA/labels/training_distance_map/filenames.csv"
    path_image = "../../../DATA/data/rIAPRD/"
    path_landmarks = "../../../DATA/labels/training_distance_map/middle_final_landmarks.csv"
    dataframe_filename = (pd.read_csv(path_image_name)).values
    dataframe_landmarks = (pd.read_csv(path_landmarks)).values
    number_spin = len(dataframe_filename)
    for spin_index in range(number_spin):
        print 'Spin index : {}/{}'.format(spin_index+1,number_spin)
        image_name = dataframe_filename[spin_index][0]
        image = plt.imread(path_image + image_name)
        width = image.shape[1]
        height = image.shape[0]
        distance_map = np.zeros((height,width),dtype=int)
        number_vertebra = len(dataframe_landmarks[spin_index])/2
        len_area = 6
        middle_landmarks = dataframe_landmarks[spin_index]
        middle_landmarks = sortMiddleLandmarks(middle_landmarks)
        list_to_color = [[] for i in range(len_area)]
        for vertebra_index in range(number_vertebra-1):
            y_index = vertebra_index+number_vertebra
            first_point = [int(middle_landmarks[y_index]*height),int(middle_landmarks[vertebra_index]*width)]
            second_point = [int(middle_landmarks[y_index+1]*height),int(middle_landmarks[vertebra_index+1]*width)]
            point_to_color = compute_distance(first_point,second_point,len_area)
            for i in range(len_area):
                list_to_color[i] += point_to_color[i]
        for i in range(len_area):
            for point in list_to_color[i]:
                #distance_map[point[0]][point[1]] = 255/(len_area-i)
                distance_map[point[0]][point[1]] = 255
        img = Image.fromarray(distance_map.astype('uint8'))
        path_save = '../../../DATA/labels/training_distance_map/distance_map_white_6/'
        img.save(path_save + image_name)

def contrastedImages(path_images, size_images, save_path_light, save_path_dark):
    index = 0
    temporary_tab = np.zeros(size_images, dtype = 'float')

    for image in os.listdir(path_images):
        print(image)
        print("Index :")
        print(index)
        imgpil = Image.open(os.path.join(path_images,image))
        img = np.array(imgpil, dtype = 'float')
        imgpil_copy = Image.open(os.path.join(path_images,image))
        copy_img = np.array(imgpil_copy, dtype = 'float')

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                copy_img[i][j] = copy_img[i][j] + 1
                temporary_tab[i][j] = np.log(copy_img[i][j])

        max = np.amax(temporary_tab)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                temporary_tab[i][j] = temporary_tab[i][j] / max
                copy_img[i][j] = temporary_tab[i][j] * 255

        path_image = os.path.join(save_path_light, image)
        copy_img = Image.fromarray(copy_img.astype('uint8'))
        copy_img.save(path_image)

        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         img[i][j] = img[i][j]/255
        #         temporary_tab[i][j] = np.exp(img[i][j])
        #
        # max = np.amax(temporary_tab)
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         temporary_tab[i][j] = (temporary_tab[i][j]-1) / max
        #         img[i][j] = temporary_tab[i][j] * 255
        img = ndimage.sobel(img)

        path_image = os.path.join(save_path_dark, image)
        img = Image.fromarray(img.astype('uint8'))
        img.save(path_image)

        index = index + 1
    return(0)
