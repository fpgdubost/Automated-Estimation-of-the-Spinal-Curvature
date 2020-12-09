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
from scipy import ndimage
from scipy.interpolate import CubicSpline

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
    print(line_zeros.shape)
    print(image.shape)
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

def reformateAutomaticDataSet(line_threshold, column_threshold, source_path, save_new_images_path, labels_path, save_new_labels_path):
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

        list_x = []
        list_y = []

        print("load labels")
        df = dataframe_labels.iloc[index]
        for i in range(len(df) / 2):
            list_x.append(df.iloc[i])
            list_y.append(df.iloc[i+len(df)/2])
        list_x = [int(x * nb_columns) for x in list_x]
        list_y = [int(y * nb_lines) for y in list_y]

        list_x = np.array(list_x)
        list_y = np.array(list_y)
        max_x = np.amax(list_x)
        max_y = np.amax(list_y)
        min_x = np.amin(list_x)
        min_y = np.amin(list_y)
        free_place_x = nb_columns - (max_x - min_x + 1)
        free_place_y = nb_lines - (max_y - min_y + 1)

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
            if (diff_lines > free_place_y):
                print("Heigth image impossible to resize")
                return(1)

            elif (diff_lines%2 != 0):
                diff_above = 0
                diff_below = 1
                while ((diff_above + diff_below) < diff_lines):
                    if (diff_above + diff_below)%2 != 0:
                        if (diff_below < nb_lines - max_y) :
                            diff_below += 1
                        else :
                            diff_above += 1

                    else :
                        if (diff_above < min_y):
                            diff_above += 1
                        else :
                            diff_below += 1

                img = deletePixels(img, diff_above, 0, diff_below, 0)

            else :
                diff_above = 0
                diff_below = diff_above

                while ((diff_above + diff_below) < diff_lines):

                    if (diff_above + diff_below)%2 != 0:
                        if (diff_below < nb_lines - max_y) :
                            diff_below += 1
                        else :
                            diff_above += 1

                    else :
                        if (diff_above < min_y):
                            diff_above += 1
                        else :
                            diff_below += 1

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
            if (diff_columns > free_place_x):
                print("Width image impossible to resize")
                return(1)

            elif (diff_columns%2 != 0):
                diff_left = 0
                diff_right = 1

                while ((diff_left + diff_right) < diff_columns):
                    if (diff_left + diff_right)%2 != 0:
                        if (diff_right < nb_columns - max_x) :
                            diff_right += 1
                        else :
                            diff_left += 1

                    else :
                        if (diff_left < min_x):
                            diff_left += 1
                        else :
                            diff_right += 1

                img = deletePixels(img, 0, diff_right, 0, diff_left)

            else :
                diff_left = 0
                diff_right = diff_left

                while ((diff_left + diff_right) < diff_columns):
                    if (diff_left + diff_right)%2 != 0:
                        if (diff_right < nb_columns - max_x) :
                            diff_right += 1
                        else :
                            diff_left += 1

                    else :
                        if (diff_left < min_x):
                            diff_left += 1
                        else :
                            diff_right += 1

                img = deletePixels(img, 0, diff_right, 0, diff_left)

            diff_left = -1 * diff_left
            diff_right = -1 * diff_right

        # resize
        list_x = [x + diff_left for x in list_x]
        list_y = [y + diff_above for y in list_y]
        #change the shape of image
        new_image_size = [nb_columns + diff_left + diff_right , nb_lines + diff_above + diff_below]
        #resize
        list_x = [float(x)/float(new_image_size[0]) for x in list_x]
        list_y = [float(y)/float(new_image_size[1]) for y in list_y]

        init_table = np.zeros(len(list_x) + len(list_y))
        new_df = pd.DataFrame(init_table)
        for i in range(len(list_x)):
            new_df.iloc[i]=list_x[i]
            new_df.iloc[i+len(list_x)]=list_y[i]

        for count in range(length_df):
            dataframe_new_labels.iloc[index].iloc[count] = new_df.iloc[count].iloc[0]
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
        name = image.split('.')[0] + ".png"
        path_image = os.path.join(save_new_images_path, name)
        img.save(path_image)
    return(0)

def createDistanceMap():
    path_image_name = "../../../DATA/labels/training_distance_map/filenames.csv"
    length = "1024_256"
    len_area = 6
    path_image = "../../../DATA/data/reduced_images_" + length +"/"
    path_landmarks = "../../../DATA/labels/training_distance_map/middle_final_landmarks_"+ length + ".csv"
    path_save = '../../../DATA/labels/training_distance_map/distance_map_white_'+str(len_area)+"_"+length + "/"
    dataframe_filename = (pd.read_csv(path_image_name)).values
    dataframe_landmarks = (pd.read_csv(path_landmarks)).values
    number_spin = len(dataframe_filename)
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    for spin_index in range(number_spin):
        print 'Spin index : {}/{}'.format(spin_index+1,number_spin)
        image_name = dataframe_filename[spin_index][0]
        image = plt.imread(path_image + image_name)
        image_name = image_name.split(".")[0] + ".png"
        width = image.shape[1]
        height = image.shape[0]
        distance_map = np.zeros((height,width),dtype=int)
        number_vertebra = len(dataframe_landmarks[spin_index])/2
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
        img.save(path_save + image_name)

def contrastedImages(path_images, size_images, save_path_light, save_path_dark):
    index = 0

    for image in os.listdir(path_images):
        print(image)
        print("Index :")
        print(index)
        imgpil = Image.open(os.path.join(path_images,image))
        img = np.array(imgpil, dtype = 'float')
        temporary_tab = np.zeros(img.shape, dtype = 'float')
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

        max = np.amax(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i][j] = img[i][j] / max
                img[i][j] = img[i][j] * 9
                temporary_tab[i][j] = np.exp(img[i][j])

        max = np.amax(temporary_tab)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                temporary_tab[i][j] = temporary_tab[i][j] / max
                img[i][j] = temporary_tab[i][j] * 255
        # img = ndimage.sobel(img)

        path_image = os.path.join(save_path_dark, image)
        img = Image.fromarray(img.astype('uint8'))
        img.save(path_image)

        index = index + 1
    return(0)

# distanceMapToCurve('../../../DATA/labels/training_angles/distance_map/')
# createDistanceMap()
def reformateTestSet(line_threshold, column_threshold, source_path, save_new_images_path):
    index = 0

    diff_above = 0
    diff_below = 0
    diff_left = 0
    diff_right = 0

    for image in os.listdir(source_path):
        print(image)
        print("Index :")
        print(index)

        imgpil = Image.open(os.path.join(source_path,image)).convert("L")
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

                img = deletePixels(img, 0, diff_right, 0, diff_left)

            else :
                diff_left = diff_columns/2
                diff_right = diff_left

                img = deletePixels(img, 0, diff_right, 0, diff_left)

        index = index + 1
        path_image = os.path.join(save_new_images_path, image)
        img = Image.fromarray(img.astype('uint8'))
        img.save(path_image)
    return(0)

def rogne(source_path, save_new_images_path):
    index = 0

    diff_above = 0
    diff_below = 0
    diff_left = 0
    diff_right = 0

    for image in os.listdir(source_path):
        print(image)
        print("Index :")
        print(index)

        imgpil = Image.open(os.path.join(source_path,image)).convert("L")
        img = np.array(imgpil)
        nb_columns = len(img[0])
        nb_lines = len(img)

        diff_right = nb_columns / 5
        diff_left = diff_right

        # diff_above = nb_lines / 10
        # diff_below = nb_lines / 8
        diff_above = 0
        diff_below = 0

        img = deletePixels(img, diff_above, diff_right, diff_below, diff_left)

        index = index + 1
        path_image = os.path.join(save_new_images_path, image)
        img = Image.fromarray(img.astype('uint8'))
        img.save(path_image)
    return(0)

def plotLandmarksOnImage(image,Landmarks):
    """
        param image: Image to plot on
        param X_landmarks: Array with the value of X
        param Y_landmarks: Array with the value of Y
    """
    X_landmarks,Y_landmarks = [],[]
    number_vertebra = len(Landmarks)/2
    for i in range(number_vertebra):
        X_landmarks.append(Landmarks[2*i])
        Y_landmarks.append(Landmarks[2*i+1])
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(image,cmap='gray')
    #Plot landmarks on the image
    ax.plot((np.array(X_landmarks))*image.shape[1],(np.array(Y_landmarks))*image.shape[0],'x',label='Landmarks',color='r')
    plt.savefig("test_landmarks")

# createDistanceMap()
# resizeProportionnalDataSet(346, 118, "../../../DATA/data/test", "../../../DATA/data/preprocess/test_proportional_reduced_images_346_118")
# reformateAutomaticDataSet(512, 128, "../../../DATA/data/preprocess/test_proportional_reduced_images_346_118", "../../../DATA/data/preprocess/test_reduced_images_512_128", "../../../DATA/labels/test/landmarks.csv", "../../../DATA/labels/test/test_final_landmarks.csv")
# resizeProportionnalDataSet(692, 236, "../../../DATA/data/test", "../../../DATA/data/preprocess/test_proportional_reduced_images_692_236")
# reformateAutomaticDataSet(1024, 256, "../../../DATA/data/preprocess/test_proportional_reduced_images_692_236", "../../../DATA/data/preprocess/test_reduced_images_1024_256", "../../../DATA/labels/test/landmarks.csv", "../../../DATA/labels/test/test_final_landmarks_1024_256.csv")
# resizeProportionnalDataSet(1384, 472, "../../../DATA/data/test", "../../../DATA/data/preprocess/test_proportional_reduced_images_1384_472")
# reformateAutomaticDataSet(2048, 512, "../../../DATA/data/preprocess/test_proportional_reduced_images_1384_472", "../../../DATA/data/preprocess/test_reduced_images_2048_512", "../../../DATA/labels/test/landmarks.csv", "../../../DATA/labels/test/test_final_landmarks_2048_512.csv")
# labels = loadPointsLandmarks("../../../DATA/labels/test/test_final_landmarks_512_128.csv", "../../../DATA/labels/test/filenames.csv")
# labels = loadPointsLandmarks("../../../DATA/labels/test/landmarks.csv", "../../../DATA/labels/test/filenames.csv")
# images = loadImages("../../../DATA/labels/test/filenames.csv", "../../../DATA/data/preprocess/test_reduced_images_512_128/", 11)
# images = loadImages("../../../DATA/labels/test/filenames.csv", "../../../DATA/data/test/", 11)
# img = Image.open("../../../DATA/data/test/sunhl-1th-01-Mar-2017-310 C AP.jpg")
# img = images[8].reshape(images[8].shape[0], images[8].shape[1])
# img = np.array(img, dtype='float')
# plotLandmarksOnImage(img,labels[8])

# resizeProportionnalDataSet(499, 75, "../../../DATA/data/preprocess/rogne_images", "../../../DATA/data/preprocess/true_test_499_75")
# reformateTestSet(512, 128, "../../../DATA/data/preprocess/true_test_499_75", "../../../DATA/data/preprocess/reduced_images_test_challenge_512_128")
# rogne("../../../DATA/data/test_sans_coccyx", "../../../DATA/data/preprocess/rogne_images_sans_coccyx")

# resizeProportionnalDataSet(692, 236, "../../../DATA/data/preprocess/preprocess_test/rogne_images_sans_coccyx", "../../../DATA/data/preprocess/preprocess_test/true_test_692_236")
# reformateTestSet(1024, 256, "../../../DATA/data/preprocess/preprocess_test/true_test_692_236", "../../../DATA/data/preprocess/preprocess_test/reduced_images_sans_coccyx_test_challenge_1024_256")

# resizeProportionnalDataSet(346, 118, "../../../DATA/data/preprocess/preprocess_test/rogne_images_sans_coccyx", "../../../DATA/data/preprocess/preprocess_test/true_test_346_118")
# reformateTestSet(512, 128, "../../../DATA/data/preprocess/preprocess_test/true_test_346_118", "../../../DATA/data/preprocess/preprocess_test/sans_coccyx_512_128")

# resizeProportionnalDataSet(500, 100, "../../../DATA/data/preprocess/preprocess_test/images_relou", "../../../DATA/data/preprocess/preprocess_test/true_images_relou")
# reformateTestSet(512, 128, "../../../DATA/data/preprocess/preprocess_test/true_images_relou", "../../../DATA/data/preprocess/preprocess_test/images_relou_sans_coccyx")

# resizeProportionnalDataSet(474, 224, "../../../DATA/data/preprocess/preprocess_test/image_relou_extrem", "../../../DATA/data/preprocess/preprocess_test/true_images_relou")
# reformateTestSet(1024, 256, "../../../DATA/data/preprocess/preprocess_test/true_images_relou", "../../../DATA/data/preprocess/preprocess_test/images_relou_sans_coccyx")

# resizeProportionnalDataSet(346, 118, "../../../DATA/data/training", "../../../DATA/data/preprocess/proportional_reduced_images")
# reformateAutomaticDataSet(512, 128, "../../../DATA/data/preprocess/proportional_reduced_images", "../../../DATA/data/preprocess/reduced_images", "../../../DATA/labels/training/landmarks.csv", "../../../DATA/labels/training/final_landmarks.csv")

# resizeProportionnalDataSet(822, 254, "../../../DATA/data/preprocess/preprocess_test/58", "../../../DATA/data/preprocess/preprocess_test/58_proportional")
# reformateAutomaticDataSet(1024, 256, "../../../DATA/data/preprocess/proportional_reduced_images_692_236", "../../../DATA/data/preprocess/training_1024_256", "../../../DATA/labels/training/landmarks.csv", "../../../DATA/labels/training/final_landmarks_692_256.csv")

# reformateTestSet(1024, 256, "../../../DATA/data/preprocess/preprocess_test/58_resize", "../../../DATA/data/preprocess/preprocess_test/58_1024_256")
