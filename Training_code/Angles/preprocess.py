import pandas as pd
import matplotlib
from PIL import Image
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import os
from curveSmooth import *
from scipy.interpolate import CubicSpline

def resizeProportionnalDataSet(height_min, width_min, source_path, save_new_images_path):

    index = 0
    for image in os.listdir(source_path):
        # print(image)
        # print("Index")
        # print(index)
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

def distanceMapToCurveValidation(distance_map_path,R2Threshold,tmax=None):
    """
    Function which compute the three angles for all distance map find in the directory given in argument
    Use this function with the validation set
    :param distance_map_path: Path to the directory with all distance map
    :param R2Threshold: keeps smoothing the curve as long as the correlation between the result curve and the original data is lower than this point
    """
    index = 1
    filenames_path = '../../../DATA/labels/training_angles/filenames.csv'
    filenames_validation_path = '../../../DATA/labels/training_angles/filenames_validation.csv'
    angles_path = '../../../DATA/labels/training_angles/angles.csv'
    min_white = 20
    if tmax==None:
        file_path = '../../../DATA/plot/Smooth_min_white='+str(min_white)+'_R2Threshold='+ str(R2Threshold)
    else:
        file_path = '../../../DATA/plot/Smooth_min_white='+str(min_white)+'_tmax='+ str(tmax)
    global_error = [0,0,0]
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
        error,smape = angleByInterpolationValidation(R2Threshold,tmax,width,X,Y,true_angle,file_path+'/'+ filenames[k][0])
        global_error += error
        global_smape += smape
        index+=1
    f = open(file_path +  "/error.txt","w+")
    f.write(str(global_error/float(index)) + "\n")
    f.write(str(global_smape*100/float(index)))
    f.close()


def distanceMapToCurveTest(distance_map_path, name, R2Threshold,tmax=None):
    index = 1
    filenames_validation_path = '../../../Results/Multi_exp/36_network=GpunetBn_loss=custom_loss_datagen=spinal2D_inputShape=reducedImages-1024_256_labelsShape=distanceMapWhite-12-1024_256/Sets/Test/test.csv'
    min_white = 20
    if tmax==None:
        file_path = '../../../DATA/plot/Final_results_Smooth_min_white='+str(min_white)+'_R2Threshold='+ str(R2Threshold) + name
    else:
        file_path = '../../../DATA/plot/Final_results_Smooth_min_white='+str(min_white)+'_tmax='+ str(tmax) + name
    global_error = [0,0,0]
    global_smape = 0
    distance_maps = loadDistanceMap(filenames_validation_path,distance_map_path)
    filenames = (pd.read_csv(filenames_validation_path)).values
    number_column = len(distance_maps)
    angles = []
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    for k in range(number_column):
        print '{}/{}'.format(index,len(distance_maps))
        print filenames[k][0]
        image = distance_maps[k]*255
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
        angle_list = angleByInterpolationTest(R2Threshold,tmax,width,X,Y,file_path+'/'+ filenames[k][0])
        index+=1
        name = (filenames[k][0]).split(".")[0]
        angles.append([name] + angle_list)
    angles = np.array(angles)
    columns = ['name','an1','an2','an3']
    new_landmarks_df = pd.DataFrame(np.array(angles))
    # Create the new csv
    new_landmarks_df.to_csv(file_path+'/'+'angles.csv',index = None,header=columns)

def angleByInterpolationValidation(R2Threshold,tmax,width,X,Y,true_angle,save_path):
    fig = plt.figure(figsize=(16,9))
    grid = plt.GridSpec(1, 3)
    ax_original = fig.add_subplot(grid[0,0])
    ax_smooth = fig.add_subplot(grid[0,1])
    ax_tang = fig.add_subplot(grid[0,2])
    ax_original.plot(X,Y)
    ax_original.set_xlim([0,width])
    ax_original.title.set_text('Original Curve')
    Y,X = heatSmoothing(np.array(Y),np.array(X),R2Threshold=R2Threshold,t_final=tmax)
    cs = CubicSpline(Y, X)
    ax_smooth.plot(cs(Y),Y)
    ax_smooth.set_xlim([0,width])
    ax_smooth.title.set_text('Smooth Curve')
    tangent = cs(np.linspace(Y[0],Y[-1],18,endpoint=True),1)
    ax_tang.plot(cs(Y,1),Y)
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
    error = np.abs(angle_list-true_angle)
    smape = sum(error)/sum(angle_list+true_angle)
    return error,smape


def angleByInterpolationTest(R2Threshold,tmax,width,X,Y,save_path):
    fig = plt.figure(figsize=(16,9))
    grid = plt.GridSpec(1, 3)
    ax_original = fig.add_subplot(grid[0,0])
    ax_smooth = fig.add_subplot(grid[0,1])
    ax_tang = fig.add_subplot(grid[0,2])
    ax_original.plot(X,Y)
    ax_original.set_xlim([0,width])
    ax_original.title.set_text('Original Curve')
    Y,X = heatSmoothing(np.array(Y),np.array(X),R2Threshold=R2Threshold,t_final=tmax)
    cs = CubicSpline(Y, X)
    ax_smooth.plot(cs(Y),Y)
    ax_smooth.set_xlim([0,width])
    ax_smooth.title.set_text('Smooth Curve')
    # number_point = len(Y)
    # distance = np.sum(np.sqrt(np.square(Y[:number_point-1]-Y[1:]) + np.square(X[:number_point-1]-X[1:])))
    # distance_point = 0
    # tangent = [cs(np.array(Y[0]),1)]
    # for i in range(number_point-1):
    #     distance_point += np.sqrt(np.square(Y[i]-Y[i+1]) + np.square(X[i]-X[i+1]))
    #     if distance_point > len(tangent)*distance/17.0:
    #         tangent_point = cs(np.array(Y[i-2:i+3]),1)
    #         tangent.append(np.sum(tangent_point)/5.0)
    # if len(tangent)<18:
    #     tangent.append(cs(np.array(Y[-1]),1))
    # tangent = np.array(tangent)
    tangent = cs(np.linspace(Y[0],Y[-1],18,endpoint=True),1)
    ax_tang.plot(cs(Y,1),Y)
    ax_tang.title.set_text('Tangent Curve')
    angle_list = [0,0,0]
    max = np.amax(tangent)
    min = np.amin(tangent)
    max_index = np.where(tangent == max)[0][0]
    min_index = np.where(tangent == min)[0][0]
    angle_list[0] = np.abs(np.arctan((max-min)/(1+max*min)))*180/np.pi
    if max_index < min_index:
        if max_index ==0:
            angle_list[1]=0.0
            sub_max = np.amax(tangent[min_index:])
            angle_list[2] = np.abs(np.arctan((sub_max-min)/(1+sub_max*min)))*180/np.pi
        elif min_index == len(tangent)-1:
            angle_list[2]=0.0
            sub_min = np.amin(tangent[:max_index])
            angle_list[1] = np.abs(np.arctan((max-sub_min)/(1+max*sub_min)))*180/np.pi
        else:
            sub_min = np.amin(tangent[:max_index])
            sub_max = np.amax(tangent[min_index:])
            angle_list[1] = np.abs(np.arctan((max-sub_min)/(1+max*sub_min)))*180/np.pi
            angle_list[2] = np.abs(np.arctan((sub_max-min)/(1+sub_max*min)))*180/np.pi
    else:
        if min_index == 0:
            angle_list[1]==0.0
            sub_min = np.amin(tangent[max_index:])
            angle_list[2] = np.abs(np.arctan((max-sub_min)/(1+max*sub_min)))*180/np.pi
        elif max_index == len(tangent)-1:
            angle_list[2]=0.0
            sub_max = np.amax(tangent[:min_index])
            angle_list[1] = np.abs(np.arctan((sub_max-min)/(1+sub_max*min)))*180/np.pi
        else:
            sub_max = np.amax(tangent[:min_index])
            sub_min = np.amin(tangent[max_index:])
            angle_list[1] = np.abs(np.arctan((sub_max-min)/(1+sub_max*min)))*180/np.pi
            angle_list[2] = np.abs(np.arctan((max-sub_min)/(1+max*sub_min)))*180/np.pi
    text = 'angle = ' + str(angle_list[0]) +', ' +str(angle_list[1]) +', ' +str(angle_list[2])
    props=dict(facecolor='none', edgecolor='black', pad=6.0)
    plt.text(0.02,0.95, text , fontsize=8, transform=plt.gcf().transFigure,bbox = props)
    fig.savefig(save_path)
    plt.close()
    return angle_list

def meanAngles(angle_csv_path_1,angle_csv_path_2,save_path):
    angles_values_1_df = pd.read_csv(angle_csv_path_1)
    angles_values_2_df = pd.read_csv(angle_csv_path_2)
    angles_values_2_df['an1'] = (angles_values_2_df['an1'] + angles_values_1_df['an1'])/2.0
    angles_values_2_df['an2'] = (angles_values_2_df['an2'] + angles_values_1_df['an2'])/2.0
    angles_values_2_df['an3'] = (angles_values_2_df['an3'] + angles_values_1_df['an3'])/2.0
    columns = ['name','an1','an2','an3']
    # Create the new csv
    angles_values_2_df.to_csv(save_path+'/'+'angles.csv',index = None,header=columns)

def removeNoise(source_path,save_path):
    height_filter = 35
    width_filter = 10
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for image in os.listdir(source_path):
        print(image)
        imgpil = Image.open(os.path.join(source_path, image))
        img = np.array(imgpil, dtype = "float")
        height_image = img.shape[0] - 1
        width_image = img.shape[1] - 1

        step = float(height_image)/512.0
        height_filter = int(35 * step)
        width_filter = int(10 * step)

        img = (img > 25)*img

        # i = height_filter
        # while i < img.shape[0]-height_filter:
        #     j = width_filter
        #     while j < img.shape[1]-width_filter:
        #         if (sum(img[i-height_filter : i+height_filter , j-width_filter]) + sum(img[i-height_filter : i+height_filter , j+width_filter]) == 0):
        #             if (sum(img[i-height_filter, j-width_filter: j+width_filter]) == 0 and sum(img[ i+height_filter, j-width_filter: j+width_filter]) == 0):
        #                 img[i-height_filter : i+height_filter, j-width_filter : j+width_filter] = np.zeros((2*height_filter, 2*width_filter))
        #                 j += width_filter / 2
        #             else:
        #                 j +=1
        #         else:
        #             j += 1
        #     i += 1

        img[:,0] = 0
        img[:,width_image] = 0
        img[0,:] = 0
        img[height_image,:] = 0

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

        # for i in range(0,img.shape[0]):
        #     for j in range(0,img.shape[1]):
        #         if (sum(img[max(0,i-height_filter), max(0,j-width_filter):min(width_image, j+width_filter)]) + sum(img[min(height_image, i+height_filter), max(0,j-width_filter):min(width_image, j+width_filter)]) == 0):
        #             if (sum(img[max(0,i-height_filter):min(height_image, i+height_filter), max(0,j-width_filter)]) == 0 and sum(img[max(0,i-height_filter):min(height_image, i+height_filter), min(width_image, j+width_filter)]) == 0):
        #                 img[max(0,i-height_filter):min(height_image, i+height_filter), max(0,j-width_filter):min(width_image, j+width_filter)] = np.zeros((min(height_image, i+height_filter) - max(0,i-height_filter), min(width_image, j+width_filter) - max(0,j-width_filter)))

        img = (img > 0)*255

        imgpil = Image.fromarray(img.astype('uint8'))
        imgpil.save(os.path.join(save_path, image))

# distanceMapToCurveValidation('../../../DATA/labels/training_angles/distance_map/',R2Threshold=0.90,tmax=13250)
# removeNoise("../../Final_code/Segmentation/14_8","../../../DATA/labels/training_angles/1024_256/final_improved_2")
# removeNoise("../../Final_code/Segmentation/14_12","../../../DATA/labels/training_angles/1024_256/final_improved_3")
# removeNoise("../../Final_code/Segmentation/16_2","../../../DATA/labels/training_angles/1024_256/final_improved_4")
# distanceMapToCurveTest('../../Final_code/Segmentation/AXEL/36_35-54-58-60-64_improved/', "_36_35-54-58-60-64", R2Threshold=0.90,tmax=65000)
# distanceMapToCurveTest('../../Final_code/Segmentation/AXEL/36_46-48-50-52-53_improved/', "_36_46-48-50-52-53", R2Threshold=0.90,tmax=65000)
distanceMapToCurveTest("../../Final_code/Segmentation/BENJ/output/36_1_3_26_31_48/", "BENJ_36_1_3_26_31_48", R2Threshold=0.90,tmax=65000)
# distanceMapToCurveTest('../../../SpinEva_2019/Final_code/Segmentation/final_improved_37/', "_37_best_models", R2Threshold=0.90,tmax=65000)
# distanceMapToCurveTest('../../../SpinEva_2019/Final_code/Segmentation/final_improved_37_2-8-12/', "_37_2-8-12", R2Threshold=0.90,tmax=65000)
# distanceMapToCurveTest('../../../SpinEva_2019/Final_code/Segmentation/final_improved_37_14-15-24/', "_37_14-15-24", R2Threshold=0.90,tmax=65000)
# distanceMapToCurveTest('../../../SpinEva_2019/Final_code/Segmentation/final_improved_37_19-20-25/', "_37_19-20-25", R2Threshold=0.90,tmax=65000)
# distanceMapToCurveTest('../../../SpinEva_2019/Final_code/Segmentation/final_improved_37_2-20-24/', "_37_2-20-24", R2Threshold=0.90,tmax=65000)
# distanceMapToCurveTest('../../../SpinEva_2019/Final_code/Segmentation/final_improved_37_2-8-15-20-24-25/', "_37_2-8-15-20-24-25", R2Threshold=0.90,tmax=65000)
# distanceMapToCurveTest('../../../SpinEva_2019/Final_code/Segmentation/final_improved_36_10-29-32/', "_36_10-29-32", R2Threshold=0.90,tmax=65000)
# distanceMapToCurveTest('../../../SpinEva_2019/Final_code/Segmentation/final_improved_36_1-3-26-31-48/', "_36_1-3-26-31-48", R2Threshold=0.90,tmax=65000)
# distanceMapToCurveTest('../../../SpinEva_2019/Final_code/Segmentation/final_improved_36_3-10-26-29-31/', "_36_3-10-26-29-31", R2Threshold=0.90,tmax=65000)
# distanceMapToCurveTest('../../../SpinEva_2019/Final_code/Segmentation/final_improved_36_10-26-29-31/', "_36_10-26-29-31", R2Threshold=0.90,tmax=65000)
# distanceMapToCurveTest('../../../DATA/labels/training_angles/1024_256/final_improved_crop_1/', "_crop_1_new_angles_2", R2Threshold=0.90,tmax=65000)
# distanceMapToCurveTest('../../../DATA/labels/training_angles/1024_256/final_improved_2/', "crop_2", R2Threshold=0.90,tmax=65000)
# distanceMapToCurveTest('../../../DATA/labels/training_angles/1024_256/final_improved_3/', "crop_3", R2Threshold=0.90,tmax=65000)
# distanceMapToCurveTest('../../../DATA/labels/training_angles/1024_256/final_improved_4/', "crop_4", R2Threshold=0.90,tmax=65000)
# meanAngles("../../../DATA/plot/Final_results_Smooth_min_white=20_tmax=65000/angles.csv","../../../DATA/plot/Final_results_Smooth_min_white=20_tmax=32500_18_nocrop/angles.csv","../../../DATA/plot")
