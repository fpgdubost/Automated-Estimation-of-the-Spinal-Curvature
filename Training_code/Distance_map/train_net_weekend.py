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
    set_path = os.path.join(dict["result_path"],dict["exp_name"],'Sets')

    train_set_x = loadImages(os.path.join(set_path,'training_files.csv') , dict["input_path"])
    train_set_x_label = loadImages(os.path.join(set_path,'training_files.csv') , dict["labels_path"] )
    valid_set_x = loadImages(os.path.join(set_path,'validation_files.csv') , dict["input_path"])
    valid_set_x_label = loadImages(os.path.join(set_path,'validation_files.csv') , dict["labels_path"] )

    print("Normalizing ...")
    train_set_x = normalize(train_set_x)
    valid_set_x = normalize(valid_set_x)
    train_set_x_label = normalize(train_set_x_label)
    valid_set_x_label = normalize(valid_set_x_label)

    if dict["datagen"]["name"] == 'spinal2D':
        datagen = DataGenerator(train_set_x, train_set_x_label, dict["datagen"]["params"], batch_size=dict["batch_size"], shuffle=True, plotgenerator = 5)

    if dict["network"] == "GpunetBn":
        model = get_gpunet_bn(dict["loss"])
    else:
        raise ValueError('Incorrect network: '+dict["network"])

    print('saving model ...')
    plot_model(model,show_shapes = True,to_file=dict["result_path"] +'/'+ dict["exp_name"] + '/model.png')

    #To do : save the model in a JSON file
    model_json = model.to_json()
    with open(os.path.join(dict["result_path"], dict["exp_name"], 'model.json'),'w') as json_file:
        json_file.write(model_json)


    best_weights_path = os.path.join(dict["result_path"], dict["exp_name"] , 'best_weights.hdf5')
    saveBestModel = callbacks.ModelCheckpoint(best_weights_path, monitor='val_loss',verbose=1, save_best_only=True, mode='auto')
    history = LossHistory(os.path.join(dict["result_path"],dict["exp_name"]), loss=dict["loss"] , batch_size = dict["batch_size"])

    ## To do : faire un fichier models, ou on stocke tous les models, avec le meme nom que l exp sauf le num
    ## save_models_path = os.path.join(result_path, exp_name , 'best_weights.hdf5')

    step_per_epoch = len(train_set_x) / dict["batch_size"]
    print(step_per_epoch)

    if dict["datagen"]["name"] is not None:
        print('Training exp #'+(dict["exp_name"])[0]+ ' using data generator : '+dict["datagen"]["name"])
        model.fit_generator(datagen.flow(train_set_x, train_set_x_label, batch_size = dict["batch_size"], shuffle=True),
        steps_per_epoch = step_per_epoch, epochs= dict["nb_epoch"],
        verbose = 1, validation_data=(valid_set_x, valid_set_x_label),
        callbacks=[saveBestModel, history],
        max_q_size=1
        )
    else:
        print('training experience #' + (dict["exp_name"])[0]+ ' without data generator')
        model.fit(train_set_x, train_set_x_label, batch_size = dict["batch_size"], epochs=dict["nb_epoch"], validation_data=(valid_set_x , valid_set_x_label), callbacks=[saveBestModel, history])



def launchExperience(dict):

    generateTrainingAndValidSetsCSV(dict["percent_of_training_file"], dict["exp_name"])
    trainModel(dict)


if __name__ == '__main__':


    exp = "exp1"
    exp_dico = {}
    exp_dico[exp] = {}
    exp_dico[exp]["loss"] = 'dice_loss' # mean_squared_error , dice_loss
    exp_dico[exp]["network"] = 'GpunetBn' # smallRegNet simpleGpunet GpunetBn
    exp_dico[exp]["batch_size"] = 4
    exp_dico[exp]["nb_epoch"] = 2
    exp_dico[exp]["percent_of_training_file"] = 0.8
    exp_dico[exp]["result_path"] = '../../../Results/Distance_map_prediction/'
    exp_dico[exp]["input_path"] = '../../../DATA/data/training_distance_map/reduced_images/'
    exp_dico[exp]["labels_path"] = '../../../DATA/labels/training_distance_map/distance_map_white_3_512_128/'
    exp_dico[exp]["input_shape"] = 'reducedImages-512-128'
    exp_dico[exp]["labels_shape"] = 'distanceMapWhite-3-512-128'
    exp_dico[exp]["datagen"] = {}
    exp_dico[exp]["datagen"]["name"] = 'spinal2D'# 'spinal2D' , None
    exp_dico[exp]["exp_name"] = 'network=' + exp_dico[exp]["network"] + '_loss=' + exp_dico[exp]["loss"] + '_datagen=' + exp_dico[exp]["datagen"]["name"] +'_inputShape='+ exp_dico[exp]["input_shape"] + '_labelsShape=' + exp_dico[exp]["labels_shape"]
    exp_dico[exp]["py_list"] = ['metrics.py','models.py','image_augmentation.py']
    exp_dico[exp]["sh_list"] = []
    exp_dico[exp]["exp_name"] = saveExperience(exp_dico[exp]["result_path"],exp_dico[exp]["exp_name"],exp_dico[exp]["py_list"],exp_dico[exp]["sh_list"])
    datagen_params = {}
    datagen_params["augmentation"] = {}
    datagen_params["augmentation"]['augmentation_choices'] = [True] # True if you want to use random_tranform
    datagen_params["augmentation"]['random_transform'] = {}
    datagen_params["augmentation"]['random_transform']['horizontal_switch']=True
    datagen_params["augmentation"]['random_transform']['vertical_switch']=True
    datagen_params["augmentation"]['random_transform']['width_shift_range']=True
    datagen_params["augmentation"]['random_transform']['height_shift_range']=True
    datagen_params["augmentation"]['random_transform']['rotate']=True
    datagen_params["augmentation"]['save_folder']= exp_dico[exp]["result_path"] +'/' + exp_dico[exp]["exp_name"] +'/Augmented_images/'
    exp_dico[exp]["datagen"]["params"] = datagen_params

    exp = "exp2"
    exp_dico = {}
    exp_dico[exp] = {}
    exp_dico[exp]["loss"] = 'dice_loss' # mean_squared_error , dice_loss
    exp_dico[exp]["network"] = 'GpunetBn' # smallRegNet simpleGpunet GpunetBn
    exp_dico[exp]["batch_size"] = 4
    exp_dico[exp]["nb_epoch"] = 2
    exp_dico[exp]["percent_of_training_file"] = 0.8
    exp_dico[exp]["result_path"] = '../../../Results/Distance_map_prediction/'
    exp_dico[exp]["input_path"] = '../../../DATA/data/training_distance_map/reduced_images_1024_256/'
    exp_dico[exp]["labels_path"] = '../../../DATA/labels/training_distance_map/distance_map_white_6_1024_256/'
    exp_dico[exp]["input_shape"] = 'reducedImages-1024-256'
    exp_dico[exp]["labels_shape"] = 'distanceMapWhite-6-1024-256'
    exp_dico[exp]["datagen"] = {}
    exp_dico[exp]["datagen"]["name"] = 'spinal2D'# 'spinal2D' , None
    exp_dico[exp]["exp_name"] = 'network=' + exp_dico[exp]["network"] + '_loss=' + exp_dico[exp]["loss"] + '_datagen=' + exp_dico[exp]["datagen"]["name"] +'_inputShape='+ exp_dico[exp]["input_shape"] + '_labelShape=' + exp_dico[exp]["labels_shape"]
    exp_dico[exp]["py_list"] = ['metrics.py','models.py','image_augmentation.py']
    exp_dico[exp]["sh_list"] = []
    exp_dico[exp]["exp_name"] = saveExperience(exp_dico[exp]["result_path"],exp_dico[exp]["exp_name"],exp_dico[exp]["py_list"],exp_dico[exp]["sh_list"])
    datagen_params = {}
    datagen_params["augmentation"] = {}
    datagen_params["augmentation"]['augmentation_choices'] = [True] # True if you want to use random_tranform
    datagen_params["augmentation"]['random_transform'] = {}
    datagen_params["augmentation"]['random_transform']['horizontal_switch']=True
    datagen_params["augmentation"]['random_transform']['vertical_switch']=True
    datagen_params["augmentation"]['random_transform']['width_shift_range']=True
    datagen_params["augmentation"]['random_transform']['height_shift_range']=True
    datagen_params["augmentation"]['random_transform']['rotate']=True
    datagen_params["augmentation"]['save_folder']= exp_dico[exp]["result_path"] +'/' + exp_dico[exp]["exp_name"] +'/Augmented_images/'
    exp_dico[exp]["datagen"]["params"] = datagen_params


    ## dictionnary that contains the information for the data DataGenerator
    for key in exp_dico:
        print(exp_dico[key]["input_shape"] , exp_dico[key]["labels_shape"])
        saveAsJson(exp_dico[key])
        launchExperience(exp_dico[key])
