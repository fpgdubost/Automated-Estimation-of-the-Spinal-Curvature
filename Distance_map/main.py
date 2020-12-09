import pandas as pd
from utils import *

if __name__ == "__main__":
    print('Launching ...')

    testData = 'test' # 'training' / 'test' / 'test_challenge'

    models_path = 'Models/'
    filenames_csv = 'filenames.csv'
    save_plot_path = 'Final_plots/'


    if testData == 'test_challenge':
        print('Challenge test set')
        save_distance_map_path = 'Test_challenge_results/'
        test_data_path = '../../../DATA/data/test/reduced_images_test_challenge_512_128/'
        test_labels_path = '../../../DATA/labels/test/'
        dataframe_filenames = pd.read_csv('../../../DATA/data/test/filenames_challenge.csv')

    elif testData == "test":
        save_distance_map_path = 'Test_results/'
        test_data_path = '../../../DATA/data/training_distance_map/reduced_images_512_128/'
        test_labels_path = '../../../DATA/labels/training_distance_map/distance_map_white_3_512_128/'
        dataframe_filenames = pd.read_csv('../../../SpinEva_2019/Training_code/test_files_fixes.csv')
        distance_maps= loadDistanceMap(test_labels_path,dataframe_filenames)

    elif testData == "training":
        save_distance_map_path = 'Training_results/'
        test_data_path = '../../../DATA/data/training_distance_map/contour_images_512_128/'
        test_labels_path = '../../../DATA/labels/training_distance_map/distance_map_white_3_512_128/'
        # test_data_path = getValueFromJson(models_path + 'dictionnary.json','input_path')
        # test_labels_path = getValueFromJson(models_path + 'dictionnary.json','labels_path')
        dataframe_filenames = pd.read_csv(test_labels_path + "../" + filenames_csv)
        dataframe_filenames = pd.read_csv(test_labels_path + '../' + filenames_csv)
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
        # img = loadImage(test_data_path,filename)
        img = loadImage(test_data_path, filename, False)
        print(img.shape)
        # image = img/float(np.amax(img))

        ## charge les landmarks correspondants
        #original_landmarks = labels[spin_index]
        if testData == "training" or testData == "test":
            original_distance_map = distance_maps[spin_index]
        ## on la fait evaluer au reseau
        #predicted_landmarks = np.zeros(len(original_landmarks))
        predicted_distance_map = np.zeros((img.shape[1],img.shape[2]))
        number_models = len(models)

        for i in range(number_models):
            # images_reshape = np.array([image.reshape(len(image), len(image[0]) , 1)])
            predicted_distance_map += ((models[i].predict(img))[0]).reshape(len(img[0]),len(img[0][0]))

        predicted_distance_map /= float(number_models)
        predicted_distance_map = np.array(predicted_distance_map*255,dtype=int)

        ## on plot img + predictions + landmarks
        if testData == "training" or testData == "test":
            plotDistanceMap(original_distance_map, predicted_distance_map, img[0].reshape(len(img[0]) , len(img[0][0]) ), save_plot_path + filename)

        print(filename)
        saveDistanceMap(predicted_distance_map, save_distance_map_path, filename)
