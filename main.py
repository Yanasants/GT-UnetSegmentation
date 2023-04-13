import tensorflow as tf

#from keras.callbacks.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

import sys
import glob
import time
import random
from PIL import Image

from unet import unet_completa, dice_coef
from unet_hedden import unet_hedden
from utils import create_folder, load_images_array
from sklearn.model_selection import train_test_split
from utils_metrics import jaccard_coef, dice_coef_2, calcula_metricas
from utils_img import save_images

from segmentation_models import Unet, Linknet, FPN, PSPNet
from segmentation_models.losses import bce_jaccard_loss, DiceLoss
from segmentation_models.metrics import iou_score

# Data augmentation 1 - 2022.05.07 Acrescentando DA no conj de valid
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.data import AUTOTUNE
from tensorflow.keras.optimizers import Adam

trainAug = Sequential([
	#preprocessing.Rescaling(scale=1.0 / 255),
	preprocessing.RandomFlip("horizontal"),
	preprocessing.RandomZoom(
		height_factor=(-0.2, +0.2),
		width_factor=(-0.2, +0.2)),
	preprocessing.RandomRotation(0.1)
])

valAug = Sequential([
	#preprocessing.Rescaling(scale=1.0 / 255),
	preprocessing.RandomFlip("horizontal"),
	preprocessing.RandomZoom(
		height_factor=(-0.2, +0.2),
		width_factor=(-0.2, +0.2)),
	preprocessing.RandomRotation(0.1)
])

data_gen_args = dict(shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    validation_split=0.1)

image_datagen = ImageDataGenerator(**data_gen_args)

TEMPO = []
ORIGINAL_SIZE = 850
NEW_SIZE = 256

# Escolha do diretorio de Treino TM40 ou TM46
_folder = './dados_girino/TM40_46prod'
# _folder = './dados_girino/TM46_40prod'

norm_imgs = sorted(glob.glob(_folder + '/A1_norm_images/*'))
GT_imgs = sorted(glob.glob(_folder + '/A2_GT_images/*'))

for i in range(len(norm_imgs)):
    if norm_imgs[i][-8:-4] != GT_imgs[i][-8:-4]:
        print('Algo está errado com as imagens')

X = load_images_array(norm_imgs, new_size = NEW_SIZE)
Y = load_images_array(GT_imgs, new_size = NEW_SIZE)
#Y = Y > 0
#Y = np.float32(Y)
print(X.shape)

# print("Maximo: ", np.max(X))
# print("Minimo: ", np.min(X))
# print("Maximo: ", np.max(Y))
# print("Minimo: ", np.min(Y))

#es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=5)
#callback = [es]
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)

use_batch_size = 4

#epoch = 100
epochs = 100 #2 #60
#spe = 385
spe = 100 #300

create_folder('outputs')

n_exec = 1
n_fold = 10 #11 #2  

iou_cv, dice_cv = [], [] # lista de listas desta execucao de cv
 
for i in range(n_fold):
        
    TEMPO = []
    time_train_1 = time.time()

    random.seed(time.time())
    seed_min = 0
    seed_max = 2**20
    SEED_1 = random.randint(seed_min, seed_max)
    SEED_2 = random.randint(seed_min, seed_max)
    SEED_3 = random.randint(seed_min, seed_max)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=SEED_1)
    
    image_generator = image_datagen.flow(X_train, Y_train,
        batch_size=use_batch_size,
        seed=SEED_2)
    # image_generator = image_datagen.flow(X, Y,
    #     batch_size=use_batch_size,
    #     seed=SEED_2)
    
    # validation_generator = image_datagen.flow(X_test, Y_test,
    #     batch_size=use_batch_size,
    #     seed=SEED_2)

    # Data Augmentation 2 - 2022.05.07 Fazendo DA no conj de valid
    trainDS = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    trainDS = trainDS.repeat(3)
    trainDS = (
     	trainDS
     	.shuffle(use_batch_size * 100)
     	.batch(use_batch_size)
     	.map(lambda x, y: (trainAug(x), trainAug(y)), num_parallel_calls=AUTOTUNE)
     	.prefetch(tf.data.AUTOTUNE)
     )

    valDS = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    valDS = valDS.repeat(3)
    valDS = (
     	valDS
     	.shuffle(use_batch_size * 100)
     	.batch(use_batch_size)
     	.map(lambda x, y: (valAug(x), valAug(y)), num_parallel_calls=AUTOTUNE)
     	.prefetch(tf.data.AUTOTUNE)
     )
     
    # N = X.shape[-1]
    N = X_train.shape[-1]

    # Modelos Mauro
    #model = unet_completa(NEW_SIZE, NEW_SIZE, SEED_3)
    #model = unet_hedden(NEW_SIZE, SEED_3)
    
    # Unet vgg16
    model = Unet(backbone_name='vgg16', encoder_weights=None,
                 input_shape=(None,None,N))
    
    # Unet resnet34
    # model = Unet(backbone_name='resnet34', encoder_weights=None,
    #               input_shape=(None,None,N))

    # Unet effnet 
    # model = Unet(backbone_name='efficientnetb0', encoder_weights=None,
    #               input_shape=(None,None,N))
    
    # Linknet vgg16
    # model = Linknet(backbone_name='vgg16', encoder_weights=None,
    #               input_shape=(None,None,N))
    
    # Linknet resnet34 
    #model = Linknet(backbone_name='resnet34', encoder_weights=None,
                  #input_shape=(None,None,N))

    # Linknet effnet 
    # model = Linknet(backbone_name='efficientnetb0', encoder_weights=None,
    #               input_shape=(None,None,N))

    
    # model.compile('Adam', 'binary_crossentropy', ['binary_accuracy']) # learning_rate=4e-5  learning_rate=1e-4
    model.compile(optimizer=Adam(), loss=bce_jaccard_loss, metrics=[iou_score]) #  loss=DiceLoss()
    
    #model.fit_generator(image_generator, epochs=epochs)
    history = model.fit(trainDS, #image_generator, 
              epochs=epochs, callbacks=callback, #steps_per_epoch=spe,
              validation_data=valDS)
    #callbacks=callback, , steps_per_epoch=spe
    
    # valendo history = model.fit_generator(image_generator, steps_per_epoch=spe, epochs=epoch, callbacks=callback, validation_data=validation_generator)
    #history = model.fit_generator(image_generator, steps_per_epoch=spe, epochs=epoch, validation_data=validation_generator)

    time_train_2 = time.time()
    TEMPO.append(time_train_2 - time_train_1)

    folder_name = './outputs/Exec_%s'%str(n_exec)
    create_folder(folder_name)
    name_file = str(use_batch_size) + "_" + str(epochs) + "_exec_%i"%n_exec + "_fold_%i"%i
    model.save(folder_name + '/girino_%s.h5'%name_file)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(folder_name + '/loss_%i.png'%i)
    plt.close()
    np.save(folder_name + '/history_%i.npy'%i, history.history)

    print("Calculando o dice para as imagens de teste")

    time_test_1 = time.time()

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED_1,)
   
    predicao = np.float64(model.predict(X_val))
    
    # save_images(X_val,Y_val,predicao,i)
    
    # X_test_save = X_test[:,:,:,0]
    # for i in range(2): # alterar para X_test.shape[0]
    #     #i = 0
    #     img_aux = Image.fromarray(np.uint8(X_test_save[i]*255),'L')
    #     fp = './save_images/orig/X_test_' + str(i) + '.tif'
    #     img_aux.save(fp, format = 'TIFF')
    
    
    # Calcula iou e dice para todas as imagens deste fold (i)
    iou_list, dice_list = [], []
    iou_list, dice_list = calcula_metricas(Y_val, predicao)    
    # append
    iou_cv.append(iou_list)
    dice_cv.append(dice_list)
    
    # time_test_2 = time.time()
    # TEMPO.append(time_test_2 - time_test_1)

# PAREI AQUI, SALVAR FILE COM VALORES E SALVAR IMAGENS NO UTILS_IMG
    # print('Salvando valores de Dice...\nMédia dos Dices: ' + str(np.mean(dice_metric)))
    # with open(folder_name + '/dice_metric_%s.txt'%name_file, 'w') as file:
    #     file.write(str(dice_metric))

    # print('Calculando e gravando tempo')

    # TEMPO.append(TEMPO[0] + TEMPO[1])

    # d = {'Tempo de treinamento': TEMPO[0],
    #     'Tempo de teste': TEMPO[1],
    #     'Tempo total': TEMPO[2]}

    # with open(folder_name + '/tempos_%s.txt'%name_file, 'w') as file:
    #     file.write(str(d))

    # d_s = {'Seed do split': SEED_1,
    #     'Seed do Data Augmentation': SEED_2,
    #     'Seed dos pesos': SEED_3}

    # with open(folder_name + '/seeds_%s.txt'%name_file, 'w') as file:
    #     file.write(str(d_s))
    
    # K.clear_session()
