# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 19:04:14 2023

@author: Labic
"""
import glob
import numpy as np
from skimage.util import img_as_float
from skimage import io
import cv2
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.data import AUTOTUNE

import time
import random
import datetime

from segmentation_models import Unet, Linknet
from tensorflow.keras.optimizers import Adam
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

from skimage.util import img_as_float
from tensorflow import keras

#from utils_YS import create_folder, load_images_array

class Dataset:
    def __init__(self, folder:str, norm_imgs_folder:str, gt_folder:str, ORIGINAL_SIZE=None, NEW_SIZE=None, X=None, Y=None):
        self.folder = folder
        self.norm_imgs_folder = norm_imgs_folder
        self.gt_folder = gt_folder
        self.ORIGINAL_SIZE = ORIGINAL_SIZE
        self.NEW_SIZE = NEW_SIZE
        self.X = X
        self.Y = Y
        self.X_train, self.X_val, self.Y_train, self.Y_val = (None, None, None, None)
        self.img_shape = None
        self.norm_imgs, self.GT_imgs = None, None
        self.load_images()

    def resize_one_img(self, img, width, height):
        self.curr_img = cv2.resize(img, (width, height))
        return self.curr_img
        
    def load_images_array(self, img_list, original_size=160, new_size = None):
        '''
        Recebe um glob das imagens e converte em um numpy array no formato que o Keras aceita
        '''
        img = np.zeros((len(img_list),new_size,new_size), dtype=float)
    
        for i in range(len(img_list)):
            
            im = np.copy(img_as_float(io.imread(img_list[i])))
            im = self.resize_one_img(im, new_size, new_size)
            img[i] = im
    
        # Padrão Keras
        img = img.reshape(-1, img.shape[-2], img.shape[-1], 1)
        return img
        
    def load_images(self):

        self.norm_imgs = sorted(glob.glob(f"{self.folder}{self.norm_imgs_folder}")) 
        self.GT_imgs = sorted(glob.glob(f"{self.folder}{self.gt_folder}")) 
                                        
        for i in range(len(self.norm_imgs)):
            if self.norm_imgs[i][-8:-4] != self.GT_imgs[i][-8:-4]:
                print('Algo está errado com as imagens')

        self.X = self.load_images_array(img_list=self.norm_imgs, original_size=self.ORIGINAL_SIZE, new_size = self.NEW_SIZE)
        self.Y = self.load_images_array(img_list=self.GT_imgs, original_size=self.ORIGINAL_SIZE, new_size = self.NEW_SIZE)
        print("\nImagens carregadas com sucesso.")
        self.img_shape = self.X.shape[0]
    
    def split_dataset(self, seed_min=0, seed_max =2**20, test_size=0.2):

        random.seed(time.time())
        seed_min = 0
        seed_max = 2**20
        SEED_1 = random.randint(seed_min, seed_max)

        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X, self.Y, test_size=test_size, random_state=SEED_1)
        print(f"\nDataset subdividido.\n{test_size*100}% dos dados para validação.\n{(1-test_size)*100}% dos dados para treino.")
                
        
    
class DataAugmentation:
    def __init__(self, X_train, Y_train, X_val, Y_val, factor=0.2, direction="horizontal", rotation=0.1):
        self.factor = factor
        self.direction = direction
        self.rotation = rotation
        self.X_train, self.Y_train, self.X_val, self.Y_val = X_train, Y_train, X_val, Y_val
        self.trainAug, self.valAug = self.train_val_sequential()
        self.trainDS, self.valDS = self.train_val_DS()
        
    def train_val_sequential(self):
        self.trainAug = Sequential([
            preprocessing.RandomFlip(self.direction),
            preprocessing.RandomZoom(
                height_factor=(-self.factor, +self.factor),
                width_factor=(-self.factor, +self.factor)),
            preprocessing.RandomRotation(self.rotation)
        ])

        self.valAug = Sequential([
            preprocessing.RandomFlip(self.direction),
            preprocessing.RandomZoom(
                height_factor=(-self.factor, +self.factor),
                width_factor=(-self.factor, +self.factor)),
            preprocessing.RandomRotation(self.rotation)
        ])
        return self.trainAug, self.valAug
    
    def train_val_DS(self, use_batch_size=4):
        self.trainDS = tf.data.Dataset.from_tensor_slices((self.X_train, self.Y_train))
        self.trainDS = self.trainDS.repeat(3)
        self.trainDS = (
            self.trainDS
            .shuffle(use_batch_size * 100)
            .batch(use_batch_size)
            .map(lambda x, y: (self.trainAug(x), self.trainAug(y)), num_parallel_calls=AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
    
        self.valDS = tf.data.Dataset.from_tensor_slices((self.X_val, self.Y_val))
        self.valDS = self.valDS.repeat(3)
        self.valDS = (
            self.valDS
            .shuffle(use_batch_size * 100)
            .batch(use_batch_size)
            .map(lambda x, y: (self.valAug(x), self.valAug(y)), num_parallel_calls=AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
        print("\nEtapa de Data Augmentation concluída.")
        return self.trainDS, self.valDS
    

class SegmentationModel:
    def __init__(self, X_train, trainDS,valDS, epochs, callback, blackbone_name="vgg16",encoder_weights=None):
        self.N = X_train.shape[-1]
        self.blackbone_name = blackbone_name
        self.encoder_weights = encoder_weights
        self.trainDS = trainDS
        self.valDS = valDS
        self.epochs = epochs
        self.callback = callback

        self.model = self.generate_model()
        
    def generate_model(self):   
        model = Unet(backbone_name=self.blackbone_name, encoder_weights=self.encoder_weights,
                    input_shape=(None,None, self.N))
        
        model.compile(optimizer=Adam(), loss=bce_jaccard_loss, metrics=[iou_score])
    
        model.fit(self.trainDS, 
                epochs=self.epochs, callbacks=[self.callback],
                validation_data=self.valDS)
        return model


    
class SaveReport:
    def __init__(self, model, folder_name, n_fold,epochs, use_batch_size=4):
        self.folder_name = folder_name
        self.n_fold = n_fold
        self.epochs = epochs
        self.use_batch_size = use_batch_size
        self.model = model

        self.dir_predict = None
        self.save_model()

        
    def create_folder(self, dirName):
            # Create target Directory if don't exist
            if not os.path.exists(dirName):
                os.mkdir(dirName)
                print("Diretorio " , dirName ,  " Criado ")
            else:    
                print("Diretorio " , dirName ,  " ja existe")
    
    def organize_folders(self):

        if (self.n_fold == 0):
            exec_moment = str(datetime.datetime.now()).replace(':','-').replace(' ','-') 
            self.output_folder = f"/outputs/Exec_{exec_moment}"
            self.exec_folder_name = f"{self.folder_name}{self.output_folder}"
        else:
            self.exec_folder_name = input("Diretório para report: ")
            self.output_folder= f"/outputs/{self.exec_folder_name}"
            self.exec_folder_name = f"{self.folder_name}{self.output_folder}"
            exec_moment = self.exec_folder_name.split('/')[-1].split('_')[1]
            
        self.create_folder(self.exec_folder_name)
        self.dir_predict = f"{self.output_folder}/fold_{self.n_fold}"
        n_fold_folder_name = f"{self.exec_folder_name}/fold_{self.n_fold}"
        self.create_folder(n_fold_folder_name)
        return exec_moment, n_fold_folder_name, self.exec_folder_name
    
    def save_model(self):
        exec_moment, self.n_fold_folder_name, exec_folder_name = self.organize_folders()
        self.name_file = "model_"+ str(self.use_batch_size) + "_" + str(self.epochs) + "_exec_%s"%(exec_moment) + "_fold_%i"%self.n_fold
        self.model_name = self.n_fold_folder_name + '/%s.h5'%self.name_file
        self.model.save(self.model_name)
        print(f"\nModelo salvo.\nNome: {self.name_file}\nSalvo em: {exec_folder_name}")
        

class PredictImages:
    def __init__(self, test_images, n_fold_folder_name, model_name, use_batch_size, img_shape):
        self.test_images = test_images
        self.model_name = model_name
        self.n_fold_folder_name = n_fold_folder_name
        self.dir_predict = n_fold_folder_name
        self.batch = use_batch_size
        self.img_shape = img_shape

        self.new_predicao = None

        self.predict()

    def predict(self):
        model = keras.models.load_model(self.model_name, compile=False)
        self.new_predicao = model.predict(self.test_images.X)
        self.new_predicao = np.uint8(255*(self.new_predicao > 0.5))

        SaveReport.create_folder(self, self.n_fold_folder_name + '/outputs_prod')
        for i in range(len(self.new_predicao)):
            io.imsave(self.n_fold_folder_name + '/outputs_prod/predicao_%s_%s.png'%(str(self.test_images.GT_imgs[i][-7:-4]), str(self.batch)),\
                       Dataset.resize_one_img(self, self.new_predicao[i], self.test_images.img_shape, self.test_images.img_shape))
        
        print("\nImagens preditas com sucesso.")


class DiceCoef(Dataset):
    def __init__(self, gt_imgs, pred_folder, new_size):
        self.gt_imgs = gt_imgs
        self.pred_folder = pred_folder
        self.new_size = new_size

        list_pred_imgs = sorted(glob.glob(f"{self.pred_folder}")) 
        self.pred_imgs = Dataset.load_images_array(self, img_list=list_pred_imgs,original_size=self.new_size, new_size=self.new_size)

        self.dice = self.dice_coef(y_true=self.gt_imgs, y_pred=self.pred_imgs)
        print(f"Coeficiente Dice: {self.dice}")

    def dice_coef(self, y_true, y_pred):
        y_true_f = keras.backend.flatten(y_true) 
        y_pred_f = keras.backend.flatten(y_pred) 
        intersection = keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + \
            keras.backend.epsilon()) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + keras.backend.epsilon())  


