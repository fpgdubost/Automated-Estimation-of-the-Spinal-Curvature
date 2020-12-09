import os
from shutil import copyfile
import json

def saveExperience(path, exp_name, py_list, sh_list):
    """
        param : the path of the result directory , list of .py codes , list of .sh codes
        function : create the new experience directory and save the codes
        return : void
    """
    next_exp_nb = experienceNumber(path)
    new_exp_name = str(next_exp_nb)+'_' + exp_name
    os.makedirs(os.path.join(path,new_exp_name))
    # first, we begin by saving all the code
    os.makedirs(os.path.join(path,new_exp_name,'Code'))
    os.makedirs(os.path.join(path,new_exp_name,'Augmented_images'))
    for py_code_file_name in py_list:
        copyfile('./%s' % py_code_file_name, os.path.join(path,new_exp_name,'Code',py_code_file_name))
    for sh_code_file_name in sh_list:
        copyfile('./%s' % sh_code_file_name, os.path.join(path,new_exp_name,'Code',sh_code_file_name))
    return new_exp_name


def experienceNumber(path):
    """
        param : path of results directory
        return : number of the next experience
    """
    list_exp_name = os.listdir(path)
    next_exp_nb = 1
    for exp_name in list_exp_name:
        nb = int(exp_name.split('_')[0])
        if nb >= next_exp_nb:
            next_exp_nb = nb+1
    return next_exp_nb

def saveAsJson(dict):

    with open('dictionnary.json', 'w') as json_file:
        json.dump(dict, json_file,sort_keys=True, indent=4)
