import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
from distance_map import *

def plotDistanceMap(spin_index):
    path_image_name = "../../../DATA/labels/training_distance_map_3/filenames.csv"
    path_image = "../../../DATA/data/training_distance_map_2/"
    path_landmarks = "../../../DATA/labels/training_distance_map/middle_landmarks.csv"
    dataframe_filename = (pd.read_csv(path_image_name)).values
    dataframe_landmarks = (pd.read_csv(path_landmarks)).values
    image_name = dataframe_filename[spin_index][0]
    image = plt.imread(path_image + image_name)
    width = image.shape[1]
    height = image.shape[0]
    distance_map = np.zeros((height,width),dtype=int)
    number_vertebra = len(dataframe_landmarks[spin_index])/2
    len_area = 4
    middle_landmarks = dataframe_landmarks[spin_index]
    middle_landmarks = sortMiddleLandmarks(middle_landmarks)
    for vertebra_index in range(number_vertebra-1):
        y_index = vertebra_index+number_vertebra
        first_point = [int(middle_landmarks[y_index]*height),int(middle_landmarks[vertebra_index]*width)]
        second_point = [int(middle_landmarks[y_index+1]*height),int(middle_landmarks[vertebra_index+1]*width)]
        list_to_color = compute_distance(first_point,second_point,len_area)
        for i in range(len_area):
            for point in list_to_color[i]:
                distance_map[point[0]][point[1]] = i+1
    fig = plt.figure(figsize=(16,9))
    grid = plt.GridSpec(1, 2)
    ax_image = fig.add_subplot(grid[0,0])
    ax_distance_map = fig.add_subplot(grid[0,1])
    ax_image.imshow(image,cmap='gray')
    ax_distance_map.imshow(distance_map,cmap='gray')
    plt.savefig('../../../DATA/data/distance_map/' + image_name)
    plt.close()

### main ###
#
# if __name__ == '__main__':
#
#      for spin_index in range(0,476):
#          print 'Spin index : {}/476'.format(spin_index+1)
#          plotDistanceMap(spin_index)
