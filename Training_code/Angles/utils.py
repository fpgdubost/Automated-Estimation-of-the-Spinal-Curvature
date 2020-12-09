import pandas as pd
import numpy as np
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


"""
Function which allows to load data from a csv file, here data are angles formed by the spine
:param labels_path: path where all angles are stored
:param index_dataframe_path: path to csv file, created by Benjamin with function generateTrainingAndValidSetsCSV, which contains the index and the names of the images to load
:return labels: list of true angles extracted from csv file
"""

def loadAngles(filenames_path,filenames_validation_path,labels_path):
    filenames_df = (pd.read_csv(filenames_path)).values
    validation_files_df = (pd.read_csv(filenames_validation_path)).values
    angles_df = (pd.read_csv(labels_path)).values
    angles = []

    for filename in validation_files_df:
        row_number = np.where(filenames_df == filename)[0][0]
        angles.append(((angles_df[row_number]))/90)
    angles = np.array(angles)
    return angles

"""
Function which allows to load data from a csv file, here data are angles formed by the spine
:param labels_path: path where all angles are stored
:param index_dataframe_path: path to csv file, created by Benjamin with function generateTrainingAndValidSetsCSV, which contains the index and the names of the images to load
:return labels: list of true angles extracted from csv file
"""

def loadAllAngles(filenames_path,filenames_validation_path,labels_path):
    filenames_df = (pd.read_csv(filenames_path)).values
    validation_files_df = (pd.read_csv(filenames_validation_path)).values
    angles_df = (pd.read_csv(labels_path)).values
    angles = []

    for filename in validation_files_df:
        row_number = np.where(filenames_df == filename)[0][0]
        angles.append(angles_df[row_number])
    angles = np.array(angles)
    return angles
"""
Function which allows to load data from a csv file, created by Benjamin with function generateTrainingAndValidSetsCSV, here data are the training or test images
:param save_images_path: path where all images are stored
:param index_dataframe_path: path to csv file which contains the index and the names of the images to load
:param nb_images: number of images that we want to load, the default value is the whole data set
:return images: list of images extracted from csv file    print(dataframe_labels[[column_number]])
"""
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

def generateTrainingAndValidSetsCSV(percent_training, exp_name):
    """
        params : percent of files that you want to use as training files, the name of the experience
        function : read the filenames that are used for the experience and create 2 set : valid and training
                   it creates 2 csv containing the filenames for both sets
        return : zero
    """
    list_filenames_df = pd.read_csv("../../../DATA/labels/training_angles/filenames.csv")
    training_files_df = pd.DataFrame(columns = ["Index", 'Name'])
    validation_files_df = pd.DataFrame(columns = ["Index", 'Name'])

    os.makedirs(os.path.join('../../../Results/Angles_prediction',exp_name,'Sets'))

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

    training_files_df.to_csv(r'../../../Results/Angles_prediction/'+exp_name+'/Sets/training_files.csv',index = None, header = True)
    validation_files_df.to_csv(r'../../../Results/Angles_prediction/'+exp_name+'/Sets/validation_files.csv',index = None, header = True)

def normalize(set):
    """
        param : a set of images
        function : normalize each image by dividing each of its pixels by the max value
        return : the normalized set
    """
    for i in range(len(set)):
        set[i] /= float(np.amax(set[i]))
    return set
