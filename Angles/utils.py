import pandas as pd
import numpy as np
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras import models
from keras.models import model_from_json

"""
Function which allows to return the max angles of the labels angles.csv
:param labels_path: path where all angles are stored
"""
def maxAngles(labels_path):
    dataframe_labels = (pd.read_csv(labels_path)).values
    max_angle = np.amax(dataframe_labels)
    return(max_angle)

def loadImage(path,filename):
    return mpimg.imread(path + filename)

def loadDistanceMap(path,dataframe):
    images=[]

    list_name = dataframe.values
    for name in list_name:
        name = name[0]
        name = name.split('.')[0] + '.png'
        images.append(mpimg.imread(path + name))
    return images

def loadModels(path):
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
    return models


def writeResultsCSV(save_result_path, results):
    results_df = pd.DataFrame( results , columns = ['0', '1','2'])
    results_df.to_csv(r'Final_results/angles.csv',index = None, header = True)

def writeResults1AngleCSV(save_result_path, results):
    results_df = pd.DataFrame( results , columns = ['0'])
    results_df.to_csv(r'Final_results/angles.csv',index = None, header = True)


def loadAngles(labels_path, index_dataframe_path):

    GT_angles = pd.DataFrame(columns = ["0", '1','2'])

    labels = []
    dataframe_labels = (pd.read_csv(labels_path))
    dataframe_index = (pd.read_csv(index_dataframe_path)).values

    t = 0
    for index in dataframe_index:
        GT_angles.loc[t,'0'] = dataframe_labels.iloc[index[0]].iloc[0]
        GT_angles.loc[t,'1'] = dataframe_labels.iloc[index[0]].iloc[1]
        GT_angles.loc[t,'2'] = dataframe_labels.iloc[index[0]].iloc[2]
        t += 1
    GT_angles.to_csv(r'GT_angles.csv',index = None, header = True)
    return(0)
