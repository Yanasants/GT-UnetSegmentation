import glob

import tensorflow as tf
import numpy as np
import pandas as pd
from skimage import io

from tensorflow import keras
from keras import backend as K
K.clear_session()

from utils_YS import create_folder, resize_one_img, load_images_array, load_images_array_return_shape, save_results

from utils_metrics import calcula_metricas

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true) # GT
    y_pred_f = K.flatten(y_pred) # Predicted
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f))

ORIGINAL_SIZE = 850 #Antigo Size Img
NEW_SIZE = 256 #Tamanho para qual as imagens serão convertidas, deixe igual ao original se não for alterar

#working_folder = './TM40_46prod/Exec_1/' 
working_folder = './outputs/Exec_2022-09-29-09-48-01.063568/' #att: precisará ser alterado conforme a pasta da execução
n_fold_path = working_folder.split('/')[2].lower() #att: nome da pasta (exec)
# working_folder = './TM46_40prod/Exec_1/'

batch = [4, 8]
index = 0

print("Carregando novas imagens")
img_folder = './'
#img_folder = './TM40_46prod/'
# img_folder = './TM46_40prod/'

new_imgs = sorted(glob.glob(img_folder + '/Producao/*'))
new_imgs_load , img_shape = load_images_array_return_shape(new_imgs, ORIGINAL_SIZE, NEW_SIZE)

GT_Test = sorted(glob.glob(img_folder + '/GT_Producao/*'))
GT_Test_dice = load_images_array(GT_Test, new_size = NEW_SIZE)

n_exec = 1
n_fold = 2 #10

general_statistics = {'MEAN':[],'STD':[],'MEDIAN':[],'MAX':[],'MIN':[]} #att: salva as estatísticas nas respectivas listas

for i in range(n_fold):
    
    print("\n\n\nRealizando Execução: %i\n\n"%i)
    
    fold_name = 'fold_%i'%i
    n_fold_folder_name = working_folder + fold_name + '/' #att: pastas para cada fold
    #filename = ['girino_4_100_%s' % fold_name.lower()] #original
    filename = ['girino_4_2_%s' %n_fold_path +'_'+ fold_name.lower()] #att: nome do arquivo 

    model = keras.models.load_model(n_fold_folder_name + '%s.h5'%(filename[index]), compile=False)

    print("Predizendo " + str(len(new_imgs_load)) + " Imagens")
    
    new_predicao = model.predict(new_imgs_load)
    new_predicao = np.uint8(255*(new_predicao > 0.5))
    #new_predicao = np.uint8(255*np.float64(new_predicao)) # nao é mais necessario
    
    # Gravando imagens no disco
    create_folder(n_fold_folder_name + 'outputs_prod')
    for i in range(len(new_predicao)):
        io.imsave(n_fold_folder_name + 'outputs_prod/predicao_%s_%s.png'%(str(GT_Test[i][-7:-4]), str(batch[index])), resize_one_img(new_predicao[i], img_shape[1], img_shape[0]))

    print("Calculando o dice de produção")
    # dice_metric = []
    # sess = tf.compat.v1.InteractiveSession() # Para TF2.8
    # for i in range(len(new_predicao)):
    #     dice_metric.append(dice_coef(new_predicao[i], GT_Test_dice[i]).eval())
    #     print("Dice número", i, " = ", dice_metric[i])
    # sess.close()

    # Calcula iou e dice para todas as imagens deste fold (i)
    iou_list, dice_list = [], []
    iou_list, dice_list = calcula_metricas(GT_Test_dice, new_predicao)  
    
    #estatística
    mean = round(np.mean(dice_list), 4)
    std = round(np.std(dice_list), 4)
    median = round(np.median(dice_list), 4)
    max_value = round(np.max(dice_list),4)
    min_value = round(np.min(dice_list),4)
    #adicionando no dicionário general_statistics
    general_statistics['MEAN'].append(mean)
    general_statistics['STD'].append(std)
    general_statistics['MEDIAN'].append(median)
    general_statistics['MAX'].append(max_value)
    general_statistics['MIN'].append(min_value)
    
    print('Salvando valores de Dice...\nMédia dos Dices: ' + str(round(np.mean(dice_list),4)))
    with open(n_fold_folder_name + 'outputs_prod/dice_production.txt', 'w') as file:
        file.write('mean: ' + str(mean) + '\n')
        file.write('std dev: ' + str(std) + '\n')
        file.write('median: ' + str(median) + '\n')
        file.write('max: ' + str(max_value) + '\n')
        file.write('min: ' + str(min_value) + '\n')
        file.write(str(dice_list))
    
    K.clear_session()

save_results(general_statistics=general_statistics, n_fold=n_fold, n_fold_path =n_fold_path)
