# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 08:56:37 2022
@author: Labic
"""

import tensorflow as tf

#from keras.callbacks.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import time
import datetime
import random

from utils import create_folder, load_images_array
from sklearn.model_selection import train_test_split

from segmentation_models import Unet, Linknet
from segmentation_models.losses import bce_jaccard_loss
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

# Choose train folder TM40 ou TM46
_folder = './TM40_Original'


norm_imgs = sorted(glob.glob(_folder + '/Norm_images/*')) 
GT_imgs = sorted(glob.glob(_folder + '/GT_images/*'))

for i in range(len(norm_imgs)):
    if norm_imgs[i][-8:-4] != GT_imgs[i][-8:-4]:
        print('Algo está errado com as imagens')

X = load_images_array(norm_imgs, new_size = NEW_SIZE)
Y = load_images_array(GT_imgs, new_size = NEW_SIZE)

print(X.shape)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)

use_batch_size = 8

epochs = 100

create_folder('./TM40_46Prod/outputs') #att

n_fold = 0 #must be manually changed to organize the results in different folders 
    
trainAug = Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomZoom(
        height_factor=(-0.2, +0.2),
        width_factor=(-0.2, +0.2)),
    preprocessing.RandomRotation(0.1)
])

valAug = Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomZoom(
        height_factor=(-0.2, +0.2),
        width_factor=(-0.2, +0.2)),
    preprocessing.RandomRotation(0.1)
])


time_train_1 = time.time()

random.seed(time.time())
seed_min = 0
seed_max = 2**20
SEED_1 = random.randint(seed_min, seed_max)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=SEED_1)


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

N = X_train.shape[-1]

# Models 

# Unet effnet 
# model = Unet(backbone_name='efficientnetb0', encoder_weights=None,
#               input_shape=(None,None,N))

#Unet vgg16
model = Unet(backbone_name='vgg16', encoder_weights=None,
            input_shape=(None,None,N))

# Linknet resnet34 
# model = Linknet(backbone_name='resnet34', encoder_weights=None,
            # input_shape=(None,None,N))




model.compile(optimizer=Adam(), loss=bce_jaccard_loss, metrics=[iou_score]) #bce_jaccard_loss

history = model.fit(trainDS, 
        epochs=epochs, #callbacks=callback, 
        validation_data=valDS)

if (n_fold == 0):
    exec_moment = str(datetime.datetime.now()).replace(':','-').replace(' ','-') #att
    exec_folder_name = './TM40_46Prod/outputs/Exec_%s'%(exec_moment) #first execution
else:
    exec_folder_name = './TM40_46Prod/outputs/Exec_2022-10-27-11-24-23.150514' #must be manually changed
    exec_moment = exec_folder_name.split('/')[-1].split('_')[1]
    
create_folder(exec_folder_name)
n_fold_folder_name = './%s'%(exec_folder_name) + "/fold_%i"%n_fold
create_folder(n_fold_folder_name)
name_file = str(use_batch_size) + "_" + str(epochs) + "_exec_%s"%(exec_moment) + "_fold_%i"%n_fold
model.save(n_fold_folder_name + '/girino_%s.h5'%name_file)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(n_fold_folder_name + '/loss_%i.png'%n_fold)
plt.close()
np.save(n_fold_folder_name + '/history_%i.npy'%n_fold, history.history)
