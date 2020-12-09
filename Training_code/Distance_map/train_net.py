from save_experience import saveExperience
from models import *
from utils import *
from keras.utils import plot_model
from keras import callbacks
from keras.models import model_from_json
from callbacks import LossHistory, saveEveryNModels
from data_generator import DataGenerator

import h5py

def trainModel():
    print("Loading data ...")


    train_set_x = loadImages('../training_files_fixes.csv' , "../../../DATA/data/training_distance_map/reduced_images_1024_256/")
    train_set_x_label = loadDistanceMap('../training_files_fixes.csv', '../../../DATA/labels/training_distance_map/distance_map_white_6_1024_256/' )
    valid_set_x = loadImages('../validation_files_fixes.csv' , "../../../DATA/data/training_distance_map/reduced_images_1024_256/")
    valid_set_x_label = loadDistanceMap('../validation_files_fixes.csv' , '../../../DATA/labels/training_distance_map/distance_map_white_6_1024_256/' )

    print("Normalizing ...")
    train_set_x = normalize(train_set_x)
    valid_set_x = normalize(valid_set_x)
    train_set_x_label = normalize(train_set_x_label)
    valid_set_x_label = normalize(valid_set_x_label)

    if datagenerator_name == 'spinal2D':
        datagen = DataGenerator(train_set_x, train_set_x_label, datagen_params, batch_size=batch_size, shuffle=True, plotgenerator = 5)

    if network == "GpunetBn":
        model = get_gpunet_bn(loss)
    else:
        raise ValueError('Incorrect network: '+network)

    print('saving model ...')
    plot_model(model,show_shapes = True,to_file=result_path +'/'+ exp_name + '/model.png')

    #To do : save the model in a JSON file
    model_json = model.to_json()
    with open(os.path.join(result_path, exp_name, 'model.json'),'w') as json_file:
        json_file.write(model_json)


    best_weights_path = os.path.join(result_path, exp_name , 'best_weights.hdf5')
    saveBestModel = callbacks.ModelCheckpoint(best_weights_path, monitor='val_loss',verbose=1, save_best_only=True, mode='auto')
    history = LossHistory(os.path.join(result_path,exp_name), loss=loss , batch_size = batch_size)

    ## To do : faire un fichier models, ou on stocke tous les models, avec le meme nom que l exp sauf le num
    ## save_models_path = os.path.join(result_path, exp_name , 'best_weights.hdf5')

    step_per_epoch = len(train_set_x) / batch_size
    print(step_per_epoch)

    if datagenerator_name is not None:
        print('Training exp #'+exp_name[0]+ ' using data generator : '+datagenerator_name)
        model.fit_generator(datagen.flow(train_set_x, train_set_x_label, batch_size = batch_size, shuffle=True),
        steps_per_epoch = step_per_epoch, epochs= nb_epoch,
        verbose = 1, validation_data=(valid_set_x, valid_set_x_label),
        callbacks=[saveBestModel, history],
        max_q_size=1
        )
    else:
        print('training experience #' + exp_name[0]+ ' without data generator')
        model.fit(train_set_x, train_set_x_label, batch_size = batch_size, epochs=nb_epoch, validation_data=(valid_set_x , valid_set_x_label), callbacks=[saveBestModel, history])

    return 0


if __name__ == '__main__':

    network = "GpunetBn" # smallRegNet simpleGpunet GpunetBn
    loss = 'dice_loss' # mean_squared_error , dice_loss , dice_loss_total_variation
    conv_activation = 'sigmoid' # replace by activ
    dense_activation = 'relu'
    datagenerator_name = 'spinal2D' # 'spinal2D' , None

    batch_size = 4
    nb_epoch = 1000

    ## dictionnary that contains the number of layers of each model
    network_dic = {}
    network_dic["smallRegNet"]= {}
    network_dic["smallRegNet"]["nb_layers"] = 2

    result_path = '../../../Results/Distance_map_prediction/'


    # py_list contains the names of the files.py to save
    # sh_list contains the names of the files.sh to save
    py_list = ['metrics.py','models.py']
    sh_list = []
    exp_name = 'network='+network+ '_loss='+loss +'_convActivation='+conv_activation +'_denseActivation='+dense_activation
    exp_name = saveExperience(result_path,exp_name,py_list,sh_list)


    ## dictionnary that contains the information for the data DataGenerator
    datagen_params = {}
    datagen_params["augmentation"] = {}
    datagen_params["augmentation"]['augmentation_choices'] = [True] # True if you want to use random_tranform
    datagen_params["augmentation"]['random_transform'] = {}
    datagen_params["augmentation"]['random_transform']['horizontal_switch']=True
    datagen_params["augmentation"]['random_transform']['vertical_switch']=True
    datagen_params["augmentation"]['random_transform']['width_shift_range']=True
    datagen_params["augmentation"]['random_transform']['height_shift_range']=True
    datagen_params["augmentation"]['random_transform']['rotate']=True
    datagen_params["augmentation"]['random_transform']['light']=True
    datagen_params["augmentation"]['random_transform']['gaussian']=True
    datagen_params["augmentation"]['random_transform']['dark']=False
    datagen_params["augmentation"]['random_transform']['transparency']=False
    datagen_params["augmentation"]['save_folder']= ''+ result_path +'/' + exp_name +'/Augmented_images/'

    # generateTrainingAndValidSetsCSV(percent_of_training_file, exp_name)

    trainModel()
