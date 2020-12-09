from save_experience import *
from models import *
from utils import *
from keras.utils import plot_model
from keras import callbacks
from keras.models import model_from_json
from callbacks import LossHistory, saveEveryNModels
from data_generator import DataGenerator
import h5py

def trainModel(dict):
    print("Loading data ...")

    train_set_x,train_set_x_label_1,train_set_x_label_2,valid_set_x,valid_set_x_label_1,valid_set_x_label_2 = loadSets(dict["training_path"],dict["input_path"],dict["labels_path_1"],dict["labels_path_2"],dict["augmented_training_set_path"],dict["augmented_validation_set_path"])

    print("Normalizing ...")
    train_set_x = normalize(train_set_x)
    valid_set_x = normalize(valid_set_x)
    train_set_x_label_1 = normalize(train_set_x_label_1)
    valid_set_x_label_1 = normalize(valid_set_x_label_1)
    train_set_x_label_2 = normalize(train_set_x_label_2)
    valid_set_x_label_2 = normalize(valid_set_x_label_2)

    path_to_save = dict["result_path_exp"] + dict["subexp_name"]
    if dict["datagen"]["name"] == 'spinal2D':
        datagen = DataGenerator(train_set_x, [train_set_x_label_1 , train_set_x_label_2], dict["datagen"]["params"], batch_size=dict["batch_size"], shuffle=True, plotgenerator = 5)

    if dict["network"] == "GpunetBn":
        model = get_gpunet_bn()
    else:
        raise ValueError('Incorrect network: '+dict["network"])

    print('saving model ...')
    plot_model(model,show_shapes = True,to_file=path_to_save + '/model.png')

    #To do : save the model in a JSON file
    model_json = model.to_json()
    with open(os.path.join(path_to_save, 'model.json'),'w') as json_file:
        json_file.write(model_json)


    best_weights_path = os.path.join(path_to_save , 'best_weights.hdf5')
    saveBestModel = callbacks.ModelCheckpoint(best_weights_path, monitor='val_loss',verbose=1, save_best_only=True, mode='auto')
    history = LossHistory(path_to_save, loss=dict["loss"] , batch_size = dict["batch_size"])

    ## To do : faire un fichier models, ou on stocke tous les models, avec le meme nom que l exp sauf le num
    ## save_models_path = os.path.join(result_path, exp_name , 'best_weights.hdf5')

    step_per_epoch = len(train_set_x) / dict["batch_size"]
    print(step_per_epoch)

    if dict["datagen"]["name"] is not None:
        print('Training exp #'+(dict["subexp_name"])[0]+ ' using data generator : '+dict["datagen"]["name"])
        model.fit_generator(datagen.flow(train_set_x, [train_set_x_label_1 , train_set_x_label_2], batch_size = dict["batch_size"], shuffle=True),
        steps_per_epoch = step_per_epoch, epochs= dict["nb_epoch"],
        verbose = 1, validation_data=(valid_set_x, [valid_set_x_label_1 , valid_set_x_label_2]),
        callbacks=[saveBestModel, history],
        max_q_size=1
        )
    else:
        print('training experience #' + (dict["subexp_name"])[0]+ ' without data generator')
        model.fit(train_set_x, [train_set_x_label_1 , train_set_x_label_2], batch_size = dict["batch_size"], epochs=dict["nb_epoch"], validation_data=(valid_set_x , [valid_set_x_label_1 , valid_set_x_label_2]), callbacks=[saveBestModel, history])

def testExperience(dict):

    print("Test models")

    path_to_save = dict["result_path_exp"] + dict["subexp_name"]
    save_distance_map_path = os.path.join(path_to_save,'Training_results/')
    os.mkdir(save_distance_map_path)
    dataframe_filenames = pd.read_csv(os.path.join(dict["test_path"],"test.csv"))

    print('Load distance map  ...')
    images = loadDistanceMap(dict["input_path"], dataframe_filenames)


    ## charger le model
    print('Load model ...')
    models = loadModels(path_to_save)

    ## boucle for sur le nom des fichiers dans filename
    for spin_index in range(len(dataframe_filenames)):
        print(spin_index)

        filename = dataframe_filenames.iloc[spin_index].iloc[0] + ".png"
        ## charge l'image a tester
        img = images[spin_index]

        image = img/float(np.amax(img))

        predicted_distance_map_2 = np.zeros((image.shape[0],image.shape[1]))
        number_models = len(models)

        for i in range(number_models):
            images_reshape = np.array([image.reshape(len(image), len(image[0]) , 1)])
            predicted_distance_map_2 += ((models[i].predict(images_reshape))[1]).reshape(len(image),len(image[0]))

        predicted_distance_map_2 /= float(number_models)
        predicted_distance_map_2 = np.array(predicted_distance_map_2*255,dtype=int)

        saveDistanceMap(predicted_distance_map_2, save_distance_map_path+'final/', filename)

def computeAngles(dict):

    path_to_save = dict["result_path_exp"] + dict["subexp_name"]
    distance_map_path_initial = os.path.join(path_to_save,"Training_results","final/")
    distance_map_path_improved = os.path.join(path_to_save,"Training_results","final_improved/")

    print "Remove Noise"
    removeNoise(distance_map_path_initial,distance_map_path_improved,dict["height_filter"],dict["width_filter"])

    print "Comput Angles"
    distanceMapToCurveValidation(distance_map_path_improved,dict["filenames_path"],os.path.join(dict["test_path"],"test.csv"),dict["angles_path"],path_to_save,dict["R2Threshold"],tmax=dict["tmax"])

def launchExperience(dict):

    trainModel(dict)


if __name__ == '__main__':


    exp = "exp"


    exp_dico = {}
    exp_dico[exp] = {}
    exp_dico[exp]["loss"] = 'custom_loss' # mean_squared_error , dice_loss
    exp_dico[exp]["network"] = 'GpunetBn' # smallRegNet simpleGpunet GpunetBn
    exp_dico[exp]["batch_size"] = 10
    exp_dico[exp]["nb_epoch"] = 400
    exp_dico[exp]["nb_exp"] = 500
    exp_dico[exp]["R2Threshold"] = 0.9
    exp_dico[exp]["tmax"] = 27000
    exp_dico[exp]["height_filter"] = 70
    exp_dico[exp]["width_filter"] = 20
    exp_dico[exp]["filenames_path"] = "../../../DATA/labels/training_distance_map/filenames.csv"
    exp_dico[exp]["angles_path"] = "../../../DATA/labels/training_angles/angles.csv"
    exp_dico[exp]["result_path"] = '../../../Results/Multi_exp/'
    exp_dico[exp]["input_path"] = '../../../DATA/data/training_distance_map/reduced_images_1024_256/'
    exp_dico[exp]["labels_path_1"] = '../../../DATA/labels/training_distance_map/distance_map_white_12_1024_256/'
    exp_dico[exp]["labels_path_2"] = '../../../DATA/labels/training_distance_map/distance_map_white_2_1024_256/'
    exp_dico[exp]["augmented_training_set_path"] = "../training_files_fixes_added_boost.csv"
    exp_dico[exp]["augmented_validation_set_path"] = "../validation_files_fixes_added_boost.csv"
    exp_dico[exp]["input_shape"] = 'reducedImages-1024_256'
    exp_dico[exp]["labels_shape"] = 'distanceMapWhite-12-1024_256'
    exp_dico[exp]["datagen"] = {}
    exp_dico[exp]["datagen"]["name"] = 'spinal2D'# 'spinal2D' , None
    exp_dico[exp]["exp_name"] = 'network=' + exp_dico[exp]["network"] + '_loss=' + exp_dico[exp]["loss"] + '_datagen=' + exp_dico[exp]["datagen"]["name"] +'_inputShape='+ exp_dico[exp]["input_shape"] + '_labelsShape=' + exp_dico[exp]["labels_shape"]
    exp_dico[exp]["py_list"] = ['metrics.py','models.py','image_augmentation.py','train_net.py']
    exp_dico[exp]["sh_list"] = []
    exp_dico[exp]["exp_name"] = saveAllExperience(exp_dico[exp]["result_path"],exp_dico[exp]["exp_name"])
    exp_dico[exp]["test_path"] = exp_dico[exp]["result_path"] + exp_dico[exp]["exp_name"] + '/Sets/Test/'
    exp_dico[exp]["training_path"] = exp_dico[exp]["result_path"] + exp_dico[exp]["exp_name"] + '/Sets/Training/'
    exp_dico[exp]["result_path_exp"] = exp_dico[exp]["result_path"] + exp_dico[exp]["exp_name"] + '/Results/'
    datagen_params = {}
    datagen_params["augmentation"] = {}
    datagen_params["augmentation"]['augmentation_choices'] = [True] # True if you want to use random_tranform
    datagen_params["augmentation"]['random_transform'] = {}
    datagen_params["augmentation"]['random_transform']['horizontal_switch']=True
    datagen_params["augmentation"]['random_transform']['vertical_switch']= False
    datagen_params["augmentation"]['random_transform']['width_shift_range']=True
    datagen_params["augmentation"]['random_transform']['height_shift_range']=True
    datagen_params["augmentation"]['random_transform']['rotate']=True
    datagen_params["augmentation"]['random_transform']['light']=True
    datagen_params["augmentation"]['random_transform']['gaussian']= False
    datagen_params["augmentation"]['random_transform']['dark']= True
    createTenSets(exp_dico[exp]["filenames_path"],exp_dico[exp]["training_path"],exp_dico[exp]["test_path"])

    for i in range(exp_dico[exp]["nb_exp"]):
        exp_dico[exp]["subexp_name"] = ""
        exp_dico[exp]["subexp_name"] = saveExperience(exp_dico[exp]["result_path_exp"],exp_dico[exp]["subexp_name"],exp_dico[exp]["py_list"],exp_dico[exp]["sh_list"])
        datagen_params["augmentation"]['save_folder']= exp_dico[exp]["result_path_exp"] +'/' + exp_dico[exp]["subexp_name"] +'/Augmented_images/'
        exp_dico[exp]["datagen"]["params"] = datagen_params
        saveAsJson(exp_dico[exp])
        launchExperience(exp_dico[exp])
        testExperience(exp_dico[exp])
        computeAngles(exp_dico[exp])
