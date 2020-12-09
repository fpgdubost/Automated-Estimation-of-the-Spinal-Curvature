import pandas as pd
from utils import *

if __name__ == "__main__":
    print('Launching ...')

    testData = 'test' # 'training' / 'test' / 'test_challenge'

    models_path = 'Models/'
    filenames_csv = 'filenames.csv'
    save_plot_path = 'Final_plots/'


    if testData == "test_challenge":

        save_distance_map_path = 'Test_challenge_results/'
        #test_data_path = '../../../DATA/data/test/reduced_images_test_challenge_512_128/'
        test_data_path = '../../../DATA/data/test/reduced_images_sans_coccyx_test_challenge_1024_256/'
        dataframe_filenames = pd.read_csv('../../../DATA/data/test/filenames_challenge.csv')
        images = loadDistanceMap(test_data_path, dataframe_filenames)

    elif testData == "test":
        save_distance_map_path = 'Test_results/'
        test_data_path = '../../../DATA/data/training_distance_map/reduced_images_1024_256/'
        test_labels_path = '../../../DATA/labels/training_distance_map_2/distance_map_white_1_1024_256/'
        dataframe_filenames = pd.read_csv('../../../Results/Multi_exp/36_network=GpunetBn_loss=custom_loss_datagen=spinal2D_inputShape=reducedImages-1024_256_labelsShape=distanceMapWhite-12-1024_256/Sets/Test/test.csv')
        images = loadDistanceMap(test_data_path, dataframe_filenames)

    elif testData == "training":
        save_distance_map_path = 'Training_results/'
        test_data_path = '../../../DATA/data/training_distance_map/reduced_images_512_128/'
        test_labels_path = '../../../DATA/labels/training_distance_map/distance_map_white_1_1024_256/'
        dataframe_filenames = pd.read_csv('../../../DATA/labels/training_distance_map/filenames.csv')
        print('Load distance map  ...')
        images = loadDistanceMap(test_data_path, dataframe_filenames)




    ## charger le model
    print('Load model ...')
    models = loadModels(models_path)

    ## boucle for sur le nom des fichiers dans filename
    for spin_index in range(len(dataframe_filenames)):
        print(spin_index)

        filename = dataframe_filenames.iloc[spin_index].iloc[0] + ".png"
        ## charge l'image a tester
        img = images[spin_index]

        image = img/float(np.amax(img))

        ## on la fait evaluer au reseau
        #predicted_landmarks = np.zeros(len(original_landmarks))
        predicted_distance_map_2 = np.zeros((image.shape[0],image.shape[1]))
        number_models = len(models)

        for i in range(number_models):
            images_reshape = np.array([image.reshape(len(image), len(image[0]) , 1)])

            predicted_distance_map_2 += ((models[i].predict(images_reshape))[1]).reshape(len(image),len(image[0]))

        predicted_distance_map_2 /= float(number_models)
        predicted_distance_map_2 = np.array(predicted_distance_map_2*255,dtype=int)

        # saveDistanceMap(predicted_distance_map, save_distance_map_path+'intermediate_1/', filename)
        saveDistanceMap(predicted_distance_map_2, save_distance_map_path+'final/', filename)
        ## calcul d erreur
            # tracer une fonction qui donne l erreur moyenne pour chaque milieu 2 points par 2
            # idee de la gausienne inverse pour chaque barycentre
