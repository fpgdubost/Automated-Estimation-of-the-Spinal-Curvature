from utils import *

if __name__ == "__main__":

    print('Launching ...')

    testData = 'training' # 'training' / 'test' / 'test_challenge'

    models_path = 'Models/'

    filenames_csv = '../../../SpinEva_2019/Training_code/validation_files_fixes.csv'
    angles_csv = 'angles.csv'
    save_result_path = 'Final_results/'
    save_error_path = 'Final_results/'
    save_distance_map_path = '../../../DATA/data/training_angles/'

    if testData == 'test_challenge':
        print('Challenge test set')
        save_distance_map_path = 'Test_challenge_results/'
        test_data_path = '../../../DATA/data/test/reduced_images_test_challenge_512_128/'
        test_labels_path = '../../../DATA/labels/test/'
        dataframe_filenames = pd.read_csv('../../../DATA/data/test/filenames_challenge.csv')

    elif testData == "test":
        save_distance_map_path = 'Test_results/'
        test_data_path = '../../../DATA/data/training_angles/'
        test_labels_path = '../../../DATA/labels/training_angles/'
        dataframe_filenames = pd.read_csv('../../../SpinEva_2019/Training_code/test_files_fixes.csv')

    elif testData == "training":
        save_distance_map_path = 'Training_results/'
        test_data_path = '../../../DATA/data/training_angles/'
        test_labels_path = '../../../DATA/labels/training_angles/'
        dataframe_filenames = pd.read_csv(test_labels_path + filenames_csv)


    print('Load Distance Maps  ...')
    distance_maps= loadDistanceMap(test_data_path, dataframe_filenames)
    GT_angles =  loadAngles('../../../DATA/labels/training_angles/angles.csv' , 'validation_files.csv')

    print('Loading models ...')
    models_1 = loadModels(models_path)

    results = []
    for distance_map_index in range(len(dataframe_filenames)):


        # load the image and normalize it
        img = distance_maps[distance_map_index]
        image = img/float(np.amax(img))

        predicted_angles= np.zeros(3)

        number_models = len(models_1)

        for i in range(number_models):
            reshaped_distance_maps = np.array([image.reshape(len(image) , len(image[0]), 1)])

            predicted_angles += (models_1[i].predict(reshaped_distance_maps)).reshape(3)


        predicted_angles /= float(number_models)
        # unnormalize the predicted angles
        predicted_angles = np.array(predicted_angles*90 , dtype=float)

        results.append(predicted_angles)

    writeResultsCSV(save_result_path, results)
    # writeResults1AngleCSV(save_result_path, results)


    #To Do:

    #Load models
    #Load distance_map
    #Load Angles

    #Faire evaluer par le reseau

    #Afficher le resultat et comparer a la realite
