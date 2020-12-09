import matplotlib
matplotlib.use('Agg')
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from curveSmooth import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.gridspec as gridspec

def landmarksToVertebra(spin_index,csv_path):
    """
        param spin_index: Index of the spin in the landmarks.csv file
        param csv_path: Path of the csv to read
        return: None
    """
    X,Y = [],[]
    i = 0
    df = pd.read_csv(csv_path)
    length_landmark = len(df.iloc[spin_index])/2

    while i < length_landmark:
        #Create the centroid of each upper and lower part of the vertebra
        X.append((df.iloc[spin_index].iloc[i] + df.iloc[spin_index].iloc[i+1])/2)
        Y.append((df.iloc[spin_index].iloc[i+length_landmark] + df.iloc[spin_index].iloc[i+1+length_landmark])/2)
        i+=2
    for j in range(len(X)/2):
        plt.plot([X[2*j],X[2*j+1]],[Y[2*j],Y[2*j+1]])
        #Plot each vertebra by joining each centroid
    plt.show()

def landmarksRegression(spin_index,degree,scores,dataframe):
    """
        param spin_index: Index of the spin in the landmarks.csv file
        param degree: Degree of the polynomial use for the regression
        param scores: dictionnary to save the score of each spin
        param dataframe: dataframe with the data
        return X_label : Array with all the x of the landmarks
        return Y_label : Array with all the y of the landmarks
    """
    #Index of the vertebra
    vertebra_index = 0

    #Number of landmarks
    length_landmark = len(dataframe.iloc[spin_index - 1])/2
    X_plot = np.arange(0,1,0.001)
    plt.rcParams['axes.grid'] = True
    fig = plt.figure(figsize=(16,9))
    grid = plt.GridSpec(2, 2)
    ax_full = fig.add_subplot(grid[0,:])
    ax_true = fig.add_subplot(grid[1,0])
    ax_projected = fig.add_subplot(grid[1,1])
    coef_text = ""
    title = 'Regression of the spin number ' + str(spin_index) +' with a ' + str(degree) + ' degree polynomial'
    img_name = 'regression_spin_' + str(spin_index) + "_polynomial_degree_" + str(degree) + ".png"
    X_label_array,Y_label,X_label = [],[],[]

    #Dictionnary to save data
    spin = {}
    spin["bottom_left_corner"]={}
    spin["bottom_left_corner"]["X"]=[]
    spin["bottom_left_corner"]["Y"]=[]
    spin["bottom_left_corner"]["X_projection"]=[]
    spin["bottom_left_corner"]["Y_projection"]=[]
    spin["bottom_left_corner"]["index"]=0
    spin["bottom_left_corner"]["poly"]=PolynomialFeatures(degree)
    spin["bottom_left_corner"]["name"]="bottom_left_corner"
    spin["bottom_left_corner"]["color"]="b"

    spin["bottom_right_corner"]={}
    spin["bottom_right_corner"]["X"]=[]
    spin["bottom_right_corner"]["Y"]=[]
    spin["bottom_right_corner"]["X_projection"]=[]
    spin["bottom_right_corner"]["Y_projection"]=[]
    spin["bottom_right_corner"]["index"]=1
    spin["bottom_right_corner"]["poly"]=PolynomialFeatures(degree)
    spin["bottom_right_corner"]["name"]="bottom_right_corner"
    spin["bottom_right_corner"]["color"]="g"

    spin["upper_left_corner"]={}
    spin["upper_left_corner"]["X"]=[]
    spin["upper_left_corner"]["Y"]=[]
    spin["upper_left_corner"]["X_projection"]=[]
    spin["upper_left_corner"]["Y_projection"]=[]
    spin["upper_left_corner"]["index"]=2
    spin["upper_left_corner"]["poly"]=PolynomialFeatures(degree)
    spin["upper_left_corner"]["name"]="upper_left_corner"
    spin["upper_left_corner"]["color"]="r"

    spin["upper_right_corner"]={}
    spin["upper_right_corner"]["X"]=[]
    spin["upper_right_corner"]["Y"]=[]
    spin["upper_right_corner"]["X_projection"]=[]
    spin["upper_right_corner"]["Y_projection"]=[]
    spin["upper_right_corner"]["index"]=3
    spin["upper_right_corner"]["poly"]=PolynomialFeatures(degree)
    spin["upper_right_corner"]["name"]="upper_right_corner"
    spin["upper_right_corner"]["color"]="y"

    while vertebra_index < length_landmark:
        #Get each corner of the vertebra in different list
        for key in spin:
            #Save data in dictionnary
            spin[key]["X"].append(df.iloc[spin_index - 1].iloc[vertebra_index+spin[key]["index"]])
            spin[key]["Y"].append(1 - df.iloc[spin_index - 1 ].iloc[vertebra_index+spin[key]["index"]+length_landmark])
        vertebra_index+=4

    for key in spin:

        #Transform the X list to fit the polynomial degree, take all columns except the first one
        #Shape is [X,power(X,2),power(X,3),...,power(X,d)] with d=degree
        #It has to be a function so we swap X and Y
        X_transform = spin[key]["poly"].fit_transform(np.array(spin[key]["Y"]).reshape(len(spin[key]["Y"]),1))[:,1:]
        X_plot_transform = spin[key]["poly"].fit_transform(X_plot.reshape(len(X_plot),1))[:,1:]
        #Transform Y to have the good shape
        Y_transform = np.array(spin[key]["X"]).reshape(len(spin[key]["X"]),1)
        #Do the linear regression with X and Y
        reg = LinearRegression().fit(X_transform,Y_transform)
        #Get the score of the regression
        score = reg.score(X_transform,Y_transform)
        coef_text += 'Validation score for {} : {} \n'.format(spin[key]["name"], score)
        scores[key]["score"]+=score
        #Projection of each dot to match the curve
        spin[key]["X_projection"],spin[key]["Y_projection"] = projection(reg,spin[key]["poly"],spin[key]["X"],spin[key]["Y"])
        #Plot everything
        ax_full.plot(spin[key]["X"],spin[key]["Y"],'d',label = spin[key]["name"] + " true",color=spin[key]["color"])
        ax_full.plot(spin[key]["X_projection"],spin[key]["Y_projection"],'o',label = spin[key]["name"] + " projection",color=spin[key]["color"])
        ax_full.plot(reg.predict(X_plot_transform),X_plot_transform[:,0],label = spin[key]["name"],color=spin[key]["color"])
        ax_full.set_xlim([0,1])
        ax_full.set_ylim([0,1.02])
        ax_true.plot(spin[key]["X"],spin[key]["Y"],'d',color=spin[key]["color"])
        ax_projected.plot(spin[key]["X_projection"],spin[key]["Y_projection"],'o',color=spin[key]["color"])

    for j in range(len(spin["bottom_left_corner"]["X_projection"])):
        #Put the value in the right order
        X_label_array += [spin["bottom_left_corner"]["X_projection"][j],spin["bottom_right_corner"]["X_projection"][j],spin["upper_left_corner"]["X_projection"][j],spin["upper_right_corner"]["X_projection"][j]]
        Y_label += [spin["bottom_left_corner"]["Y_projection"][j],spin["bottom_right_corner"]["Y_projection"][j], spin["upper_left_corner"]["Y_projection"][j],spin["upper_right_corner"]["Y_projection"][j]]

    for j in range(len(X_label_array)):
        X_label.append(X_label_array[j][0])

    ax_full.title.set_text(title)
    ax_true.title.set_text('True landmarks')
    ax_projected.title.set_text('Projected landmarks')
    handles, labels = ax_full.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',ncol=4)
    props=dict(facecolor='none', edgecolor='black', pad=6.0)
    plt.text(0.02,0.9, coef_text, fontsize=9, transform=plt.gcf().transFigure,bbox = props)
    plt.savefig("../../../DATA/plot/degree_" +str(degree) +"/"+ img_name)
    plt.close()
    return X_label,Y_label

def projection(reg,poly,X,Y):
    """
        param reg: Linear regression to match
        param poly: poly to match
        param X: Initial X to project. It's a numpy array
        param Y: Initial Y to project. It's a numpy array
        return X_projection: Numpy array with the projected value in X
        return Y_projection: Numpy array with the projected value in Y
    """
    Y_range = np.linspace(min(Y),max(Y),1000)
    Y_transform = poly.fit_transform(Y_range.reshape(len(Y_range),1))[:,1:]
    X_range = reg.predict(Y_transform)
    X_projection,Y_projection = [],[]
    for i in range(len(Y)):
        min_distance = float("inf")
        min_index = -1
        for j in range(len(Y_range)):
            #For each dot on the initial array we calculate the distance between each regression dot
            #We take the min
            diff_y = Y_range[j]-Y[i]
            diff_x = X_range[j]-X[i]
            distance = diff_y**2 + diff_x**2
            if distance<min_distance:
                min_distance = distance
                min_index = j
        X_projection.append(X_range[min_index])
        Y_projection.append(Y_range[min_index])
    return X_projection,Y_projection

def landmarksSmoothing(degree):
    """
        param degree: Degree of the polynomial function
        return: None
    """

    name = "landmarks_regression_and_projection_polynomial_degree_" + str(degree) + ".csv"
    new_landmarks = []
    df = pd.read_csv("../../../DATA/labels/training/landmarks.csv")
    number_spin = df.shape[0]
    number_columns = df.shape[1]
    columns = []

    #Create a dictionnary to save the score for each spin
    scores = {}
    scores["bottom_left_corner"]={}
    scores["bottom_left_corner"]["name"]="bottom_left_corner"
    scores["bottom_left_corner"]["score"]=0

    scores["bottom_right_corner"]={}
    scores["bottom_right_corner"]["name"]="bottom_right_corner"
    scores["bottom_right_corner"]["score"]=0

    scores["upper_left_corner"]={}
    scores["upper_left_corner"]["name"]="upper_left_corner"
    scores["upper_left_corner"]["score"]=0

    scores["upper_right_corner"]={}
    scores["upper_right_corner"]["name"]="upper_right_corner"
    scores["upper_right_corner"]["score"]=0

    for spin_index in range(number_spin):
        #Do the regression for each spin
        print 'Spin index : {}/{}'.format(spin_index + 1,number_spin)
        X_label,Y_label = landmarksRegression(spin_index + 1,degree,scores,dataframe)
        #Get the new X and Y
        new_landmarks.append(X_label+Y_label)
    for j in range(number_columns):
        columns.append(str(j))
    f = open("../../../DATA/plot/degree_" +str(degree) +"/" + "meansScore_degree" + str(degree)+ ".txt","w+")
    for key in scores:
        # Calculate the mean of the score for each corner
        text = 'Validation score for {} : {} \n'.format(scores[key]["name"], scores[key]["score"]/float(number_spin))
        f.write(text)
    f.close()
    new_landmarks_df = pd.DataFrame(np.array(new_landmarks))
    #Create the new csv with the new landmarks
    new_landmarks_df.to_csv("../../../DATA/labels/training/" + name,index = None,header=columns)

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def gaussianCombination(small_degree,high_degree):
    """
        param small_degree: smallest degree of polynomial function used to execute the combination
        param high_degree : higher degree of polynomial function used to execute the combination
    """
    path = "../../../DATA/labels/training/landmarks_regression_and_projection_polynomial_degree_"
    name = "landmarks_smoother.csv"
    #Get the dataframe of each polynomial function
    df_small_degree = pd.read_csv(path + str(small_degree) + ".csv")
    df_high_degree = pd.read_csv(path + str(high_degree) + ".csv")
    number_spin = df_small_degree.shape[0]
    number_columns = df_small_degree.shape[1]
    columns,new_landmarks = [],[]

    for spin_index in range(number_spin):
        #For each spin calculate the combination
        print 'Spin index : {}/{}'.format(spin_index + 1,number_spin)
        X_label,Y_label = spinGaussianCombination(df_small_degree,df_high_degree,spin_index+1,small_degree,high_degree)
        new_landmarks.append(X_label+Y_label)

    for j in range(number_columns):
        columns.append(str(j))

    new_landmarks_df = pd.DataFrame(np.array(new_landmarks))
    #Create the new csv
    new_landmarks_df.to_csv("../../../DATA/labels/training/" + name,index = None,header=columns)

def spinGaussianCombination(df_small_degree,df_high_degree,spin_index,small_degree,high_degree):
    """
        param df_small_degree: Dataframe containing the value of the small degree polynomial
        param df_high_degree: Dataframe containing the value of the high degree polynomial
        param spin_index: Index of the spin in the landmarks.csv file
        param small_degree: smallest degree of polynomial function used to execute the combination
        param high_degree : higher degree of polynomial function used to execute the combination
        return X_label: New array with the new x value
        return Y_label: New array with the new y value
    """
    mu = 0.5
    sig = 0.125
    vertebra_index = 0
    plt.rcParams['axes.grid'] = True
    fig = plt.figure(figsize=(16,9))
    grid = plt.GridSpec(1, 3)
    ax_small_degree = fig.add_subplot(grid[0,0])
    ax_high_degree = fig.add_subplot(grid[0,1])
    ax_gaussian = fig.add_subplot(grid[0,2])
    title = "Spinal curve"
    img_name = 'gaussian_combination_spin_' + str(spin_index) + "_degree_" + str(small_degree) +"_and_degree_" + str(high_degree)+ ".png"

    #Number of landmarks
    length_landmark = len(df_small_degree.iloc[spin_index - 1])/2

    X_label,Y_label = [],[]

    #Dictionnary to save data
    spin = {}
    spin["bottom_left_corner"]={}
    spin["bottom_left_corner"]["X_small_degree"]=[]
    spin["bottom_left_corner"]["Y_small_degree"]=[]
    spin["bottom_left_corner"]["X_high_degree"]=[]
    spin["bottom_left_corner"]["Y_high_degree"]=[]
    spin["bottom_left_corner"]["X_gaussian_combination"]=[]
    spin["bottom_left_corner"]["Y_gaussian_combination"]=[]
    spin["bottom_left_corner"]["index"]=0
    spin["bottom_left_corner"]["name"]="bottom_left_corner"
    spin["bottom_left_corner"]["color"]="b"

    spin["bottom_right_corner"]={}
    spin["bottom_right_corner"]["X_small_degree"]=[]
    spin["bottom_right_corner"]["Y_small_degree"]=[]
    spin["bottom_right_corner"]["X_high_degree"]=[]
    spin["bottom_right_corner"]["Y_high_degree"]=[]
    spin["bottom_right_corner"]["X_gaussian_combination"]=[]
    spin["bottom_right_corner"]["Y_gaussian_combination"]=[]
    spin["bottom_right_corner"]["index"]=1
    spin["bottom_right_corner"]["name"]="bottom_right_corner"
    spin["bottom_right_corner"]["color"]="g"

    spin["upper_left_corner"]={}
    spin["upper_left_corner"]["X_small_degree"]=[]
    spin["upper_left_corner"]["Y_small_degree"]=[]
    spin["upper_left_corner"]["X_high_degree"]=[]
    spin["upper_left_corner"]["Y_high_degree"]=[]
    spin["upper_left_corner"]["X_gaussian_combination"]=[]
    spin["upper_left_corner"]["Y_gaussian_combination"]=[]
    spin["upper_left_corner"]["index"]=2
    spin["upper_left_corner"]["name"]="upper_left_corner"
    spin["upper_left_corner"]["color"]="r"

    spin["upper_right_corner"]={}
    spin["upper_right_corner"]["X_small_degree"]=[]
    spin["upper_right_corner"]["Y_small_degree"]=[]
    spin["upper_right_corner"]["X_high_degree"]=[]
    spin["upper_right_corner"]["Y_high_degree"]=[]
    spin["upper_right_corner"]["X_gaussian_combination"]=[]
    spin["upper_right_corner"]["Y_gaussian_combination"]=[]
    spin["upper_right_corner"]["index"]=3
    spin["upper_right_corner"]["name"]="upper_right_corner"
    spin["upper_right_corner"]["color"]="y"

    while vertebra_index < length_landmark:
        #Get each corner of the vertebra in different list
        for key in spin:
            #Save data in dictionnary
            spin[key]["X_small_degree"].append(df_small_degree.iloc[spin_index - 1].iloc[vertebra_index+spin[key]["index"]])
            spin[key]["Y_small_degree"].append(df_small_degree.iloc[spin_index - 1 ].iloc[vertebra_index+spin[key]["index"]+length_landmark])
            spin[key]["X_high_degree"].append(df_high_degree.iloc[spin_index - 1].iloc[vertebra_index+spin[key]["index"]])
            spin[key]["Y_high_degree"].append(df_high_degree.iloc[spin_index - 1 ].iloc[vertebra_index+spin[key]["index"]+length_landmark])
        vertebra_index+=4

    vertebra_index = 0
    while vertebra_index < len(spin[key]["X_small_degree"]):
        #Get each corner of the vertebra in different list
        for key in spin:
            #Gaussian combination for each dot
            #For Y we take the middle of the two polynomial function
            spin[key]["Y_gaussian_combination"].append((spin[key]["Y_small_degree"][vertebra_index] + spin[key]["Y_high_degree"][vertebra_index])/2)
            #For X we take (1-Gaussian(Y))*Small_degree(X) + Gaussian(Y)*high_degree(X)
            spin[key]["X_gaussian_combination"].append((1 - gaussian(spin[key]["Y_gaussian_combination"][vertebra_index],mu,sig))*spin[key]["X_small_degree"][vertebra_index] + gaussian(spin[key]["Y_gaussian_combination"][vertebra_index],mu,sig)*spin[key]["X_high_degree"][vertebra_index])
        vertebra_index+=1

    for j in range(len(spin["bottom_left_corner"]["X_gaussian_combination"])):
        #Put the value in the right order
        X_label += [spin["bottom_left_corner"]["X_gaussian_combination"][j],spin["bottom_right_corner"]["X_gaussian_combination"][j],spin["upper_left_corner"]["X_gaussian_combination"][j],spin["upper_right_corner"]["X_gaussian_combination"][j]]
        Y_label += [spin["bottom_left_corner"]["Y_gaussian_combination"][j],spin["bottom_right_corner"]["Y_gaussian_combination"][j], spin["upper_left_corner"]["Y_gaussian_combination"][j],spin["upper_right_corner"]["Y_gaussian_combination"][j]]

    for key in spin:
        #Plot everything
        ax_small_degree.plot(spin[key]["X_small_degree"],spin[key]["Y_small_degree"],'d',label = spin[key]["name"] + " degree" + str(small_degree),color=spin[key]["color"])
        ax_small_degree.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03),ncol=1,prop={'size': 8})
        ax_small_degree.title.set_text(title + " with degree " + str(small_degree) + " regression")
        ax_high_degree.plot(spin[key]["X_high_degree"],spin[key]["Y_high_degree"],'o',label = spin[key]["name"] + " degree" + str(high_degree),color=spin[key]["color"])
        ax_high_degree.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03),ncol=1,prop={'size': 8})
        ax_high_degree.title.set_text(title + " with degree " + str(high_degree) + " regression")
        ax_gaussian.plot(spin[key]["X_gaussian_combination"],spin[key]["Y_gaussian_combination"],'+',label = spin[key]["name"] + " gaussian combination",color=spin[key]["color"])
        ax_gaussian.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03),ncol=1,prop={'size': 8})
        ax_gaussian.title.set_text(title + " with gaussian combination of regressions")
    plt.savefig("../../../DATA/plot/gaussian_combination/"+ img_name)
    plt.close()

    #Return the new X and Y
    return X_label,Y_label

def plotLandmarksOnOriginalandReshape(spin_index):
    """
        param spin_index: Index of the spin in the csv file
        return: None
    """
    path_landmarks_reshape = "../../../DATA/labels/training/centroid_landmarks.csv"
    path_landmarks_original_reshape = "../../../DATA/labels/training/landmarks_initial.csv"
    path_image_name = "../../../DATA/labels/training/filenames.csv"
    path_image_original = "../../../DATA/data/training_origin/"
    path_image_reshape = "../../../DATA/data/training/"
    df_landmarks = pd.read_csv(path_landmarks_reshape)
    df_landmarks_original = pd.read_csv(path_landmarks_original_reshape)
    df_image = pd.read_csv(path_image_name)
    img_name = df_image.iloc[spin_index-1].iloc[0]
    img_original = plt.imread(path_image_original + str(img_name))
    img_reshape = plt.imread(path_image_reshape + str(img_name))
    length_landmark_original = len(df_landmarks_original.iloc[spin_index-1])/2
    length_landmark_reshape = len(df_landmarks.iloc[spin_index-1])/2
    X_reshape = df_landmarks.values[spin_index-1][:length_landmark_reshape]*img_reshape.shape[1]
    Y_reshape = (df_landmarks.values[spin_index-1][length_landmark_reshape:])*img_reshape.shape[0]
    X_original = df_landmarks_original.values[spin_index-1][:length_landmark_original]*img_original.shape[1]
    Y_original = (df_landmarks_original.values[spin_index-1][length_landmark_original:])*img_original.shape[0]
    #Plot everything
    fig = plt.figure(figsize=(16,9))
    grid = plt.GridSpec(1, 2)
    ax_original = fig.add_subplot(grid[0,0])
    ax_reshape = fig.add_subplot(grid[0,1])
    ax_original.imshow(img_original,cmap='gray')
    #Plot landmarks on the image
    ax_original.plot(X_original,Y_original,'x',label='Original landmarks',color='b')
    ax_original.legend(loc='upper center', bbox_to_anchor=(0.5, -0.04),ncol=1)
    ax_original.title.set_text("Original Landmarks on Spin")
    ax_reshape.imshow(img_reshape,cmap='gray')
    #Plot landmarks on the image
    ax_reshape.plot(X_reshape, Y_reshape, '+',label='Reshape landmarks',color='r')
    ax_reshape.legend(loc='upper center', bbox_to_anchor=(0.5, -0.04),ncol=1)
    ax_reshape.title.set_text("Reshape Landmarks on Spin")
    plot_name = "landmarks_on_spin_original_and_regression_" + str(img_name) + ".png"
    plt.savefig("../../../DATA/plot/proportionnal_plot/"+ plot_name)
    plt.close()
    #plt.show()


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
    plt.show()

def landmarksToMiddle(path_landmarks,path_save_landmarks):
    df_landmarks = pd.read_csv(path_landmarks)
    spin_middle_list = []
    df_landmarks = df_landmarks.values
    number_spin = len(df_landmarks)
    number_middle = len(df_landmarks[0])/2
    for spin_index in range(number_spin):
        print 'Spin index {}/{}'.format(spin_index+1,number_spin)
        middle_list = np.zeros(number_middle)
        for middle_index in range(number_middle):
            middle_list[middle_index]=0.5*(df_landmarks[spin_index][2*middle_index]+df_landmarks[spin_index][2*middle_index+1])
        spin_middle_list.append(middle_list)
    columns = np.arange(number_middle)

    new_landmarks_df = pd.DataFrame(np.array(spin_middle_list))
    #Create the new csv
    new_landmarks_df.to_csv(path_save_landmarks,index = None,header=columns)

def landmarksToCentroid(path_landmarks,path_save_landmarks):
    df_landmarks = pd.read_csv(path_landmarks)
    df_landmarks = df_landmarks.values
    spin_centroid_list = []
    number_spin = len(df_landmarks)
    number_centroid = len(df_landmarks[0])/4
    for spin_index in range(number_spin):
        print 'Spin index {}/{}'.format(spin_index+1,number_spin)
        centroid_list = np.zeros(number_centroid)
        for centroid_index in range(number_centroid):
            centroid_list[centroid_index]=0.25*(df_landmarks[spin_index][4*centroid_index]+df_landmarks[spin_index][4*centroid_index+1]+df_landmarks[spin_index][4*centroid_index+2]+df_landmarks[spin_index][4*centroid_index+3])
        spin_centroid_list.append(centroid_list)
    columns = np.arange(number_centroid)

    new_landmarks_df = pd.DataFrame(np.array(spin_centroid_list))
    #Create the new csv
    new_landmarks_df.to_csv(path_save_landmarks,index = None,header=columns)

landmarksToMiddle('../../../DATA/labels/training_distance_map/final_landmarks_512_128.csv','../../../DATA/labels/training_distance_map/middle_final_landmarks_512_128.csv')
