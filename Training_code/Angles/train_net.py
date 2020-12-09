from save_experience import saveExperience
from models import *
from utils import *
from keras.utils import plot_model
from keras import callbacks
from keras.models import model_from_json
from callbacks import LossHistory, saveEveryNModels
from data_generator import DataGenerator
from preprocess import *

import h5py

def trainModel():
    print("Loading data ... ")

    resizeProportionnalDataSet(128,32, '../../../DATA/data/training_angles_initial_size/' , '../../../DATA/data/training_angles')

    set_path = os.path.join(result_path,exp_name,'Sets')

    train_set_x = loadDistanceMap( '../training_files_fixes.csv' , '../../../DATA/data/training_angles/')
    train_set_x_label = loadAngles('../../../DATA/labels/training_angles/filenames.csv' , '../training_files_fixes.csv','../../../DATA/labels/training_angles/angles.csv')
    valid_set_x = loadDistanceMap('../validation_files_fixes.csv' , '../../../DATA/data/training_angles/')
    valid_set_x_label = loadAngles('../../../DATA/labels/training_angles//filenames.csv' , '../validation_files_fixes.csv','../../../DATA/labels/training_angles/angles.csv')


    train_set_x = np.reshape(train_set_x, (len(train_set_x),len(train_set_x[0]),len(train_set_x[0][0]),1))
    valid_set_x = np.reshape(valid_set_x, (len(valid_set_x),len(valid_set_x[0]),len(valid_set_x[0][0]),1))
    # train_set_x_label = np.reshape(train_set_x_label, (len(train_set_x_label),len(train_set_x_label[0])))
    # valid_set_x_label = np.reshape(valid_set_x_label, (len(valid_set_x_label),len(valid_set_x_label[0])))
    print train_set_x_label.shape
    train_set_x_label = np.reshape(train_set_x_label,(train_set_x_label.shape[0],train_set_x_label.shape[1]))
    valid_set_x_label = np.reshape(valid_set_x_label, (valid_set_x_label.shape[0],valid_set_x_label.shape[1]))

    # global val_size = len(valid_set_x)

    print("Normalizing ...")
    train_set_x = normalize(train_set_x)
    valid_set_x = normalize(valid_set_x)

    if datagenerator_name == 'spinal2D':
        datagen = DataGenerator(train_set_x, train_set_x_label, datagen_params, paddingGT=padding, batch_size=batch_size, shuffle=True, plotgenerator = 5)

    if network == "GpunetBn":
        model = get_gpunet_bn(loss)
    elif network == 'smallRegNet':
        model = smallRegNet(loss , conv_activation , dense_activation)
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

    step_per_epoch = len(train_set_x)*10 / batch_size

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
    loss = 'mean_squared_error'
    conv_activation = 'relu' # replace by activ
    dense_activation = 'relu'
    datagenerator_name = None # 'spinal2D' , None

    val_size = 1

    percent_of_training_file = 0.8

    batch_size = 10
    nb_epoch = 1000
    padding = 0

    ## dictionnary that contains the number of layers of each model
    network_dic = {}
    network_dic["smallRegNet"]= {}
    network_dic["smallRegNet"]["nb_layers"] = 2

    result_path = '../../../Results/Angles_prediction'


    # py_list contains the names of the files.py to save
    # sh_list contains the names of the files.sh to save
    py_list = ['metrics.py','models.py']
    sh_list = []
    exp_name = 'network='+network+ '_loss='+loss +'_convActivation='+conv_activation +'_denseActivation='+dense_activation
    exp_name = saveExperience(result_path,exp_name,py_list,sh_list)


    ## dictionnary that contains the information for the data DataGenerator
    datagen_params = {}
    datagen_params["augmentation"] = {}
    datagen_params["augmentation"]['augmentation_choices'] = [False] # True if you want to use random_tranform
    datagen_params["augmentation"]['random_transform'] = {}
    datagen_params["augmentation"]['random_transform']['horizontal_switch']=True
    datagen_params["augmentation"]['random_transform']['vertical_switch']=True
    datagen_params["augmentation"]['random_transform']['width_shift_range']=True
    datagen_params["augmentation"]['random_transform']['height_shift_range']=True
    datagen_params["augmentation"]['random_transform']['rotate']=True
    datagen_params["augmentation"]['save_folder']= ''+ result_path +'/' + exp_name +'/Augmented_images/'


    path_training_images = '../../../DATA/data/training_angles'


    generateTrainingAndValidSetsCSV(percent_of_training_file, exp_name)

    trainModel()
