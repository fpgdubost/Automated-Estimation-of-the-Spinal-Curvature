import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
import os

def generateTrainingAndValidSetsCSV(percent_training, exp_name):
    """
        params : percent of files that you want to use as training files, the name of the experience
        function : read the filenames that are used for the experience and create 2 set : valid and training
                   it creates 2 csv containing the filenames for both sets
        return : zero
    """
    list_filenames_df = pd.read_csv("../../../DATA/labels/training_distance_map_2/filenames.csv")
    training_files_df = pd.DataFrame(columns = ["Index", 'Name'])
    validation_files_df = pd.DataFrame(columns = ["Index", 'Name'])

    os.makedirs(os.path.join('../../../Results/Distance_map_2_prediction/',exp_name,'Sets'))

    t = 0
    v = 0
    for index, row in list_filenames_df.iterrows():
        if random.uniform(0,1) < percent_training:
            training_files_df.loc[t,"Index"]=index
            training_files_df.loc[t,"Name"]=row['Name']
            t += 1
        else:
            validation_files_df.loc[v,'Index']=index
            validation_files_df.loc[v,'Name']=row['Name']
            v += 1

    training_files_df.to_csv(r'../../../Results/Distance_map_2_prediction/'+exp_name+'/Sets/training_files.csv',index = None, header = True)
    validation_files_df.to_csv(r'../../../Results/Distance_map_2_prediction/'+exp_name+'/Sets/validation_files.csv',index = None, header = True)

"""
Function which allows to load data from the csv file sent by organizers, here data are the training or test images
:param path_csv: path to csv file
:param images_path: path where all images are stored
:param nb_images: number of images that we want to load, the default value is the whole data set
:return images: list of images extracted from csv file
"""
def loadImagesModified(path_csv, path_images, nb_images = None):
    images = []
    name_images = []
    dataframe_names_images = pd.read_csv(path_csv)

    if (nb_images == None):
        for index_names in range(len(dataframe_names_images)):
            img = mpimg.imread(path_images + dataframe_names_images.iloc[index_names].iloc[0])
            name_images.append(dataframe_names_images.iloc[index_names].iloc[0])
            images.append(img)
    else :
        if (nb_images > len(dataframe_names_images)):
            print("Not enough images in the database, loading of all images")
            loadImages(path_csv, path_images)

        else :
            counter = 0

            while(counter < nb_images):
                img = mpimg.imread(path_images + dataframe_names_images.iloc[counter].iloc[0])
                images.append(img)
                name_images.append(dataframe_names_images.iloc[counter].iloc[0])
                counter += 1

    return(name_images, images)

"""
Function which allows to load data from a csv file, created by Benjamin with function generateTrainingAndValidSetsCSV, here data are the training or test images
:param save_images_path: path where all images are stored
:param index_dataframe_path: path to csv file which contains the index and the names of the images to load
:param nb_images: number of images that we want to load, the default value is the whole data set
:return images: list of images extracted from csv file
"""
def loadImagesLabels(index_dataframe_path, images_path, nb_images = None):
    images = []
    dataframe_numbers_lines = pd.read_csv(index_dataframe_path)

    if (nb_images == None):
        for index, row in dataframe_numbers_lines.iterrows():
            img = mpimg.imread(images_path+ row['Name'])
            img = np.array(img, dtype = 'float')
            img = img.reshape(img.shape[0], img.shape[1], 1)
            images.append(img)

    else :
        if (nb_images > len(dataframe_numbers_lines)):
            print("Not enough images in the database, loading of all images")
            loadImages(index_dataframe_path, images_path)

        else :
            counter = 0

            while(counter < nb_images):
                img = mpimg.imread(images_path + dataframe_numbers_lines.iloc[counter].iloc[0])
                img = np.array(img, dtype = 'float')
                img = img.reshape(img.shape[0], img.shape[1], 1)
                images.append(img)
                images.append(img)
                counter += 1

    images = np.array(images,dtype='float')
    return(images)


def loadImages(index_dataframe_path, images_from_first_network_path,  nb_images = None):
    images = []
    dataframe_numbers_lines = pd.read_csv(index_dataframe_path)

    if (nb_images == None):
        for index, row in dataframe_numbers_lines.iterrows():
            img = mpimg.imread(images_from_first_network_path+ row['Name'])
            img = np.array(img, dtype = 'float')
            img = img.reshape(img.shape[0], img.shape[1], 1)
            images.append(img)

    else :
        if (nb_images > len(dataframe_numbers_lines)):
            print("Not enough images in the database, loading of all images")
            loadImages(index_dataframe_path, images_path)

        else :
            counter = 0

            while(counter < nb_images):
                img = mpimg.imread(images_path + dataframe_numbers_lines.iloc[counter].iloc[0])
                img = np.array(img, dtype = 'float')
                img = img.reshape(img.shape[0], img.shape[1], 1)
                images.append(img)
                images.append(img)
                counter += 1

    images = np.array(images,dtype='float')
    return(images)


def loadDistanceMap(index_dataframe_path, images_path, nb_images = None):
    dataframe_numbers_lines = pd.read_csv(index_dataframe_path)
    images = []
    if (nb_images == None):
        for index, row in dataframe_numbers_lines.iterrows():
            name = row['Name']
            name = name.split('.')[0] + '.png'
            img = mpimg.imread(images_path+ name)
            img = np.array(img, dtype = 'float')
            img = img.reshape(img.shape[0], img.shape[1], 1)
            images.append(img)

    else :
        if (nb_images > len(dataframe_numbers_lines)):
            print("Not enough images in the database, loading of all images")
            loadImages(index_dataframe_path, images_path)

        else :
            counter = 0

            while(counter < nb_images):
                name = dataframe_numbers_lines.iloc[counter].iloc[0]
                name = name.split('.')[0] + '.png'
                img = mpimg.imread(images_path + name)
                img = np.array(img, dtype = 'float')
                img = img.reshape(img.shape[0], img.shape[1], 1)
                images.append(img)
                counter += 1

    images = np.array(images, dtype = 'float')
    return(images)
    
def loadPointsLandmarks(path, index_dataframe_path):
    labels = []

    dataframe_numbers_lines = pd.read_csv(index_dataframe_path)
    df = pd.read_csv(path)
    length = len(df.iloc[0])/2
    for index, row in dataframe_numbers_lines.iterrows():
        label = []
        for i in range(length):
            label.append(df.iloc[index].iloc[i])
            label.append(df.iloc[index].iloc[i + length])
        labels.append(label)
    return (labels)

def sortMiddleLandmarks(middle_landmarks):

    number_vertebra = len(middle_landmarks)/2
    for i in range(number_vertebra,1,-1):
        for j in range(i-1):
            if middle_landmarks[j+number_vertebra]<middle_landmarks[j+1+number_vertebra]:
                new_y = middle_landmarks[j+number_vertebra]
                middle_landmarks[j+number_vertebra]= middle_landmarks[j+1+number_vertebra]
                middle_landmarks[j+1+number_vertebra] = new_y
                new_x = middle_landmarks[j]
                middle_landmarks[j] = middle_landmarks[j+1]
                middle_landmarks[j+1] = new_x
    return middle_landmarks

def plotLandmarksOnImage(image,Landmarks,save=False,save_path=""):
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
    if (save):
        plt.savefig(save_path)
    plt.close()


def normalize(set):
    """
        param : a set of images
        function : normalize each image by dividing each of its pixels by the max value
        return : the normalized set
    """
    for i in range(len(set)):
        set[i] /= float(np.amax(set[i]))
    return set

def loadPointsLandmarks(path, index_dataframe_path):
    labels = []

    dataframe_numbers_lines = pd.read_csv(index_dataframe_path)
    df = pd.read_csv(path)
    length = len(df.iloc[0])/2
    for index, row in dataframe_numbers_lines.iterrows():
        img_labels = []
        for i in range(length):
            point = []
            point.append(df.iloc[index].iloc[i])
            point.append(df.iloc[index].iloc[i + length])
            img_labels.append(point)
        labels.append(img_labels)
    for k in range(len(labels[0])):
        print(labels[0][k])

    list_tilted_vertebraes =[]
    for j in range(len(labels)):

        pente_min = 0
        pente_max = 0
        most_tilted_vertebraes = [[0,0,0,0],[0,0,0,0]]
        for i in range(len(labels[j])/4):
            x1 , y1 = labels[j][4*i][0] , labels[j][4*i][1]
            x2 , y2 = labels[j][4*i+1][0] , labels[j][4*i+1][1]
            x3 , y3 = labels[j][4*i+2][0] , labels[j][4*i+2][1]
            x4 , y4 = labels[j][4*i+3][0] , labels[j][4*i+3][1]

            x_mean_left , y_mean_left = (x1+x3)/2.0 , (y1+y3)/2.0

            x_mean_right , y_mean_right = (x2+x4)/2.0 , (y2+y4)/2.0

            pente = (y_mean_right - y_mean_left) / float((x_mean_right - x_mean_left))


            if pente > pente_max:
                pente_max = pente
                most_tilted_vertebraes[0] = [x3, y3 , x4 , y4]
            elif pente < pente_min:
                pente_min = pente
                most_tilted_vertebraes[1] = [x3, y3 , x4 , y4]
        list_tilted_vertebraes.append(most_tilted_vertebraes)

    # list_x = []
    # list_y = []
    # list_x_tilted = []
    # list_y_tilted = []
    # for i in range(len(labels[0])):
    #     list_x.append(labels[0][i][0])
    #     list_y.append(labels[0][i][1])
    #
    # print list_tilted_vertebraes[0]
    # list_x_tilted.append(list_tilted_vertebraes[0][0][0])
    # list_x_tilted.append(list_tilted_vertebraes[0][0][2])
    # list_x_tilted.append(list_tilted_vertebraes[0][1][0])
    # list_x_tilted.append(list_tilted_vertebraes[0][1][2])
    #
    # list_y_tilted.append(list_tilted_vertebraes[0][0][1])
    # list_y_tilted.append(list_tilted_vertebraes[0][0][3])
    # list_y_tilted.append(list_tilted_vertebraes[0][1][1])
    # list_y_tilted.append(list_tilted_vertebraes[0][1][3])
    #
    # fig = plt.figure()
    # plt.scatter(list_x,list_y)
    # plt.scatter(list_x_tilted,list_y_tilted, color='red')
    #
    # # plt.scatter(x1,y1,color='green')
    # # plt.scatter(x2,y2,color='black')
    # # plt.scatter(x3,y3,color='cyan')
    # # plt.scatter(x4,y4,color='magenta')
    #
    # plt.axis((0,1,0,1))
    # fig.savefig('fig.png')
    # plt.close()

    return 0
