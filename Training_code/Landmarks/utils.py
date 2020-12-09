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
    list_filenames_df = pd.read_csv("../../../DATA/labels/training/filenames.csv")
    training_files_df = pd.DataFrame(columns = ["Index", 'Name'])
    validation_files_df = pd.DataFrame(columns = ["Index", 'Name'])
    os.makedirs(os.path.join('../../../Results',exp_name,'Sets'))
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
    training_files_df.to_csv(r'../../../Results/'+exp_name+'/Sets/training_files.csv',index = None, header = True)
    validation_files_df.to_csv(r'../../../Results/'+exp_name+'/Sets/validation_files.csv',index = None, header = True)

    return 0

"""
Function which allows to load data from the csv file sent by organizers, here data are the training or test images
:param path_csv: path to csv file
:param images_path: path where all images are stored
:param nb_images: number of images that we want to load, the default value is the whole data set
:return images: list of images extracted from csv file
"""
def loadImagesSimplified(path_csv, path_images, nb_images = None):
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
:param images_path: path where all images are stored
:param index_dataframe_path: path to csv file which contains the index and the names of the images to load
:param nb_images: number of images that we want to load, the default value is the whole data set
:return images: list of images extracted from csv file
"""
def loadImages(images_path, index_dataframe_path, nb_images = None):
    images = []
    dataframe_numbers_lines = pd.read_csv(index_dataframe_path)

    if (nb_images == None):
        for index, row in dataframe_numbers_lines.iterrows():
            img = mpimg.imread("../../../DATA/data/training/" + row['Name'])
            images.append(img)
    else :
        if (nb_images > len(dataframe_numbers_lines)):
            print("Not enough images in the database, loading of all images")
            loadImagesModified(images_path, index_dataframe_path)

        else :
            counter = 0

            while(counter < nb_images):
                img = mpimg.imread(images_path + "/" + dataframe_numbers_lines[counter][1])
                images.append(img)
                counter += 1

    return(images)


"""
COMMENTAIRE A REECRIRE, FONCTION NON A JOUR
Function which allows to load data from a csv file, here data are landmarks of the points forming spine
:param labels_path: path where all angles are stored
:param index_dataframe_path: path to csv file, created by Benjamin with function generateTrainingAndValidSetsCSV, which contains the index and the names of the images to load
:return labels: list of true angles extracted from csv file
"""
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
