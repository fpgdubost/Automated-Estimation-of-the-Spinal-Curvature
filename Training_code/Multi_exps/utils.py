import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import os
from keras.models import model_from_json
from keras.models import load_model
from curveSmooth import *
from scipy.interpolate import CubicSpline

def createTenSets(filenames_path,training_path,test_path):
    filenames = pd.read_csv(filenames_path).values
    np.random.shuffle(filenames)
    split_sets = np.array_split(filenames, 10)

    ## on choisit aleatoirement un set de test
    test_set_id = np.random.randint(10)
    test_set = split_sets[test_set_id]

    for i in range(len(split_sets)):
        if i != test_set_id:
            set_filenames = pd.DataFrame(split_sets[i] , columns=['Name'])
            set_filenames.to_csv(training_path+'set_' + str(i)+'.csv',index = None, header = True)

        else:
            set_filenames = pd.DataFrame(split_sets[i] , columns=['Name'])
            set_filenames.to_csv(test_path + 'test.csv',index = None, header = True)


"""
Function which allows to load data from a csv file, created by Benjamin with function generateTrainingAndValidSetsCSV, here data are the training or test images
:param save_images_path: path where all images are stored
:param index_dataframe_path: path to csv file which contains the index and the names of the images to load
:param nb_images: number of images that we want to load, the default value is the whole data set
:return images: list of images extracted from csv file
"""
def loadImagesTraining(index_dataframe_path, images_path,augmented_path, nb_images = None):
    dataframe_numbers_lines = pd.read_csv(index_dataframe_path)
    augmented_list = (pd.read_csv(augmented_path)).values
    images = []
    if (nb_images == None):
        for index, row in dataframe_numbers_lines.iterrows():
            name = row['Name']
            number_images = 1
            if np.any(augmented_list==name):
                number_images = 4
            name = name.split('.')[0] + '.png'
            for i in range(number_images):
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
                img = mpimg.imread(images_path + dataframe_numbers_lines.iloc[counter].iloc[0])
                img = np.array(img, dtype = 'float')
                img = img.reshape(img.shape[0], img.shape[1], 1)
                images.append(img)
                counter += 1

    return(images)

def loadImagesValidation(index_dataframe_path, images_path, nb_images = None):
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
            loadImagesValidation(index_dataframe_path, images_path)

        else :
            counter = 0

            while(counter < nb_images):
                img = mpimg.imread(images_path + dataframe_numbers_lines.iloc[counter].iloc[0])
                img = np.array(img, dtype = 'float')
                img = img.reshape(img.shape[0], img.shape[1], 1)
                images.append(img)
                counter += 1

    return(images)

def loadDistanceMapTraining(index_dataframe_path, images_path,augmented_path, nb_images = None):
    dataframe_numbers_lines = pd.read_csv(index_dataframe_path)
    images = []
    augmented_list = (pd.read_csv(augmented_path)).values
    if (nb_images == None):
        for index, row in dataframe_numbers_lines.iterrows():
            name = row['Name']
            number_images = 1
            if np.any(augmented_list==name):
                # nombre de fois pour le boost
                number_images = 4
            name = name.split('.')[0] + '.png'
            for i in range(number_images):
                img = mpimg.imread(images_path+ name)
                img = np.array(img, dtype = 'float')
                img = img.reshape(img.shape[0], img.shape[1], 1)
                images.append(img)

    else :
        if (nb_images > len(dataframe_numbers_lines)):
            print("Not enough images in the database, loading of all images")
            loadImagesTraining(index_dataframe_path, images_path,augmented_path)

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

    return(images)

def loadDistanceMapValidation(index_dataframe_path, images_path, nb_images = None):
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

    return(images)

def loadSets(set_path,images_path,distance_map_path_1,distance_map_path_2,augmented_validation_path,augmented_training_path):
    set_list = os.listdir(set_path)
    validation_list = random.choice(set_list)
    set_list.remove(validation_list)
    validation_data = loadImagesValidation(set_path+validation_list,images_path)
    validation_labels_1 = loadDistanceMapValidation(set_path + validation_list,distance_map_path_1)
    validation_labels_2 = loadDistanceMapValidation(set_path + validation_list,distance_map_path_2)
    training_data = []
    training_labels_1 = []
    training_labels_2 = []
    for training_path in set_list:
        training_data += loadImagesTraining(set_path + training_path,images_path,augmented_training_path)
        training_labels_1 += loadDistanceMapTraining(set_path + training_path,distance_map_path_1,augmented_training_path)
        training_labels_2 += loadDistanceMapTraining(set_path + training_path,distance_map_path_2,augmented_training_path)
    validation_data = np.array(validation_data, dtype = 'float')
    validation_labels_1 = np.array(validation_labels_1, dtype = 'float')
    validation_labels_2 = np.array(validation_labels_2, dtype = 'float')
    training_data = np.array(training_data, dtype = 'float')
    training_labels_1 = np.array(training_labels_1, dtype = 'float')
    training_labels_2 = np.array(training_labels_2, dtype = 'float')
    return training_data,training_labels_1,training_labels_2,validation_data,validation_labels_1,validation_labels_2


def normalize(set):
    """
        param : a set of images
        function : normalize each image by dividing each of its pixels by the max value
        return : the normalized set
    """
    for i in range(len(set)):
        set[i] /= float(np.amax(set[i]))
    return set

def loadModels(path):
    models = []

    json_file = open(os.path.join(path, 'model.json') , 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(path, 'best_weights.hdf5'))

    models.append(loaded_model)

    return models

def saveDistanceMap(distance_map,path,name):
    if not os.path.exists(path):
        os.mkdir(path)
    name = name.split('.')[0] + '.png'
    img = Image.fromarray(distance_map.astype('uint8'))
    img.save(path + name)

def loadDistanceMap(path,dataframe):
    images=[]

    list_name = dataframe.values
    for name in list_name:
        name = name[0]
        name = name.split('.')[0] + ".png"
        images.append(mpimg.imread(path + name))
    return np.array(images)

def distanceMapToCurveValidation(distance_map_path,filenames_path,filenames_validation_path,angles_path,result_path,R2Threshold,tmax=None):
    """
    Function which compute the three angles for all distance map find in the directory given in argument
    Use this function with the validation set
    :param distance_map_path: Path to the directory with all distance map
    :param R2Threshold: keeps smoothing the curve as long as the correlation between the result curve and the original data is lower than this point
    """
    index = 1
    min_white = 0
    angles_csv = []
    if tmax==None:
        file_path = result_path + '/Smooth_min_white='+str(min_white)+'_R2Threshold='+ str(R2Threshold)
    else:
        file_path = result_path + '/Smooth_min_white='+str(min_white)+'_tmax='+ str(tmax)
    global_mae = [0,0,0]
    global_mse = [0,0,0]
    global_smape = 0
    angles = loadAllAngles(filenames_path,filenames_validation_path,angles_path)
    distance_maps = loadImages(filenames_validation_path,distance_map_path)
    filenames = (pd.read_csv(filenames_validation_path)).values
    number_column = len(distance_maps)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    for k in range(number_column):
        print '{}/{}'.format(index,len(distance_maps))
        print filenames[k][0]
        true_angle = angles[k]
        image = distance_maps[k]
        image = image.reshape(image.shape[0],image.shape[1])
        width = image.shape[1]
        height = image.shape[0]
        X,Y = [],[]
        for i in range(height):
            spin_interval = np.where(image[i]>min_white)
            if spin_interval[0].size:
                width_min = spin_interval[0][0]
                width_max = spin_interval[0][-1]
                width_mean = (width_max + width_min)/2
                if width_mean:
                    X.append(width_mean)
                    Y.append(i)
        if X!=[] and Y!=[]:
            mae,mse,smape,angles_list = angleByInterpolationValidation(R2Threshold,tmax,width,X,Y,true_angle,file_path+'/'+ filenames[k][0],height)
            global_mae += mae
            global_mse += mse
            global_smape += smape
            name = (filenames[k][0]).split(".")[0]
            angles_csv.append([name] + angles_list)
        index+=1
    f = open(file_path +  "/error.txt","w+")
    f.write("mae = " + str(np.array(global_mae)/float(index)) + "\n")
    f.write("mse = " + str(np.array(global_mse)/float(index)) + "\n")
    f.write("smape = " + str(global_smape*200/float(index)))
    f.close()
    if len(angles_csv) != 0:
        angles_csv = np.array(angles_csv)
        columns = ['name','an1','an2','an3']
        new_landmarks_df = pd.DataFrame(np.array(angles_csv))
        # Create the new csv
        new_landmarks_df.to_csv(file_path+'/'+'angles.csv',index = None,header=columns)

def angleByInterpolationValidation(R2Threshold,tmax,width,X,Y,true_angle,save_path,height):
    fig = plt.figure(figsize=(16,9))
    grid = plt.GridSpec(1, 3)
    ax_original = fig.add_subplot(grid[0,0])
    ax_smooth = fig.add_subplot(grid[0,1])
    ax_tang = fig.add_subplot(grid[0,2])
    ax_original.plot(X,height-np.array(Y))
    ax_original.set_xlim([0,width])
    ax_original.title.set_text('Original Curve')
    Y,X = heatSmoothing(np.array(Y),np.array(X),R2Threshold=R2Threshold,t_final=tmax)
    cs = CubicSpline(Y, X)
    ax_smooth.plot(cs(Y),height-Y)
    ax_smooth.set_xlim([0,width])
    ax_smooth.title.set_text('Smooth Curve')
    tangent = cs(np.linspace(Y[0],Y[-1],18,endpoint=True),1)
    ax_tang.plot(cs(Y,1),height-Y)
    ax_tang.title.set_text('Tangent Curve')
    angle_list = [0,0,0]
    max = np.amax(tangent)
    min = np.amin(tangent)
    max_index = np.where(tangent == max)[0][0]
    min_index = np.where(tangent == min)[0][0]
    delta = 0
    angle_list[0] = np.abs(np.arctan((max-min)/(1+max*min)))*180/np.pi
    if max_index < min_index:
        if max_index ==0:
            angle_list[1]=0.0
            sub_max = np.amax(tangent[min_index:])
            angle_list[2] =delta +  np.abs(np.arctan((sub_max-min)/(1+sub_max*min)))*180/np.pi
        elif min_index == len(tangent)-1:
            angle_list[2]=0.0
            sub_min = np.amin(tangent[:max_index])
            angle_list[1] = delta + np.abs(np.arctan((max-sub_min)/(1+max*sub_min)))*180/np.pi
        else:
            sub_min = np.amin(tangent[:max_index])
            sub_max = np.amax(tangent[min_index:])
            angle_list[1] = delta + np.abs(np.arctan((max-sub_min)/(1+max*sub_min)))*180/np.pi
            angle_list[2] = delta + np.abs(np.arctan((sub_max-min)/(1+sub_max*min)))*180/np.pi
    else:
        if min_index == 0:
            angle_list[1]==0.0
            sub_min = np.amin(tangent[max_index:])
            angle_list[2] = delta + np.abs(np.arctan((max-sub_min)/(1+max*sub_min)))*180/np.pi
        elif max_index == len(tangent)-1:
            angle_list[2]=0.0
            sub_max = np.amax(tangent[:min_index])
            angle_list[1] = delta + np.abs(np.arctan((sub_max-min)/(1+sub_max*min)))*180/np.pi
        else:
            sub_max = np.amax(tangent[:min_index])
            sub_min = np.amin(tangent[max_index:])
            angle_list[1] = delta + np.abs(np.arctan((sub_max-min)/(1+sub_max*min)))*180/np.pi
            angle_list[2] = delta + np.abs(np.arctan((max-sub_min)/(1+max*sub_min)))*180/np.pi
    text = 'true angle = ' + str(true_angle[0]) +', ' +str(true_angle[1]) +', ' +str(true_angle[2]) + '\nangle = ' + str(angle_list[0]) +', ' +str(angle_list[1]) +', ' +str(angle_list[2])
    props=dict(facecolor='none', edgecolor='black', pad=6.0)
    plt.text(0.02,0.95, text , fontsize=8, transform=plt.gcf().transFigure,bbox = props)
    fig.savefig(save_path)
    plt.close()
    mse = np.square(angle_list-true_angle)
    mae = np.abs(angle_list-true_angle)
    smape = sum(mae)/sum(angle_list+true_angle)
    return mae,mse,smape,angle_list

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

def loadImages(index_dataframe_path, images_path,  nb_images = None):
    images = []
    dataframe_numbers_lines = pd.read_csv(index_dataframe_path)

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
                img = mpimg.imread(images_path + dataframe_numbers_lines.iloc[counter].iloc[0])
                img = np.array(img, dtype = 'float')
                img = img.reshape(img.shape[0], img.shape[1], 1)
                images.append(img)
                images.append(img)
                counter += 1

    images = np.array(images,dtype='float')
    return(images)

def removeNoise(source_path,save_path,height_filter,width_filter):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for image in os.listdir(source_path):
        print(image)
        imgpil = Image.open(os.path.join(source_path, image))
        img = np.array(imgpil, dtype = "float")
        height_image = img.shape[0] - 1
        width_image = img.shape[1] - 1

        img = (img > 20)*img

        i = 0
        while i < img.shape[0]-1:
            j = 0
            while j < img.shape[1]-1:
                if (sum(img[max(0,i-height_filter), max(0,j-width_filter):min(width_image, j+width_filter)]) + sum(img[min(height_image, i+height_filter), max(0,j-width_filter):min(width_image, j+width_filter)]) == 0):
                    if (sum(img[max(0,i-height_filter):min(height_image, i+height_filter), max(0,j-width_filter)]) == 0 and sum(img[max(0,i-height_filter):min(height_image, i+height_filter), min(width_image, j+width_filter)]) == 0):
                        img[max(0,i-height_filter):min(height_image, i+height_filter), max(0,j-width_filter):min(width_image, j+width_filter)] = np.zeros((min(height_image, i+height_filter) - max(0,i-height_filter), min(width_image, j+width_filter) - max(0,j-width_filter)))
                        j += width_filter / 2
                    else:
                        j +=1
                else:
                    j += 1
            i += 1
        img = (img > 0)*255

        imgpil = Image.fromarray(img.astype('uint8'))
        imgpil.save(os.path.join(save_path, image))
