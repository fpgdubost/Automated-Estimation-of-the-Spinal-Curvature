import pandas as pd
from utils import *

if __name__ == "__main__":
    print('Launching ...')

    testData = 'test_challenge' # 'training' / 'test' / 'test_challenge'

    models_path = 'Models/'
    filenames_csv = 'filenames.csv'
    save_plot_path = 'Final_plots/'


    if testData == "test_challenge":

        save_distance_map_path = 'Test_challenge_results/'
        test_data_path = '../../../DATA/data/test_training_distance_map_2//'
        test_labels_path = '../../../DATA/labels/test/'
        dataframe_filenames = pd.read_csv('../../../DATA/data/test/filenames_challenge.csv')

    elif testData == "test":
        save_distance_map_path = 'Test_results/'
        test_data_path = '../../../DATA/data/training_distance_map_2/from_network_1/'
        test_labels_path = '../../../DATA/labels/training_distance_map_2/distance_map_white_1_512_128/'
        dataframe_filenames = pd.read_csv('../../../SpinEva_2019/Training_code/test_files_fixes.csv')
        distance_maps= loadDistanceMap(test_labels_path, dataframe_filenames)

    elif testData == "training":
        save_distance_map_path = 'Training_results/'
        test_data_path = '../../../DATA/data/training_distance_map_2/from_network_1/'
        test_labels_path = '../../../DATA/labels/training_distance_map_2/distance_map_white_1_512_128/'
        dataframe_filenames = pd.read_csv(test_labels_path + '../'+ filenames_csv)
        print('Load distance map  ...')
        distance_maps= loadDistanceMap(test_labels_path,dataframe_filenames)




    ## charger le model
    print('Load model ...')
    models = loadModels(models_path)

    ## boucle for sur le nom des fichiers dans filename
    for spin_index in range(len(dataframe_filenames)):
        print(spin_index)

        filename = dataframe_filenames.iloc[spin_index].iloc[0]
        ## charge l'image a tester
        filename = filename.split('.')[0] + ".png"
        img = loadImage(test_data_path,filename)
        image = img/float(np.amax(img))

        ## charge les landmarks correspondants
        #original_landmarks = labels[spin_index]
        if testData == "training" or testData == "test":
            original_distance_map = distance_maps[spin_index]
        ## on la fait evaluer au reseau
        #predicted_landmarks = np.zeros(len(original_landmarks))
        predicted_distance_map = np.zeros((image.shape[0],image.shape[1]))
        number_models = len(models)

        for i in range(number_models):
            images_reshape = np.array([image.reshape(len(image), len(image[0]) , 1)])
            predicted_distance_map += ((models[i].predict(images_reshape))[0]).reshape(len(image),len(image[0]))

        predicted_distance_map /= float(number_models)
        predicted_distance_map = np.array(predicted_distance_map*255,dtype=int)

        ## on plot img + predictions + landmarks
        if testData == "training" or testData == "test":
            plotDistanceMap(original_distance_map, predicted_distance_map, image, save_plot_path + filename)
        saveDistanceMap(predicted_distance_map, save_distance_map_path + filename)
        ## calcul d erreur
            # tracer une fonction qui donne l erreur moyenne pour chaque milieu 2 points par 2
            # idee de la gausienne inverse pour chaque barycentre
