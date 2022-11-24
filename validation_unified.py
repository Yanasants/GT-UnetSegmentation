"""
Describe: DSC software (models v1 and v2), receives the GTs and the segmentations.
Authors: Eduardo Destefani Stefanato & Vitor Souza Premoli Pinto de Oliveira.
Contact: edustefanato@gmail.com
Date: 17/11/2022.
MIT License | Copyright (c) 2022 Eduardo Destefani Stefanato
"""
# imports (not all modules are needed to avoid some unnecessary dependencies and errors)
from pandas import DataFrame
import tensorflow as tf
# from model_v1 import *
from keras import *
from image import *
import numpy as np
import glob

def dice_coef(y_true, y_pred):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2.*intersection+1)/(backend.sum(y_true_f)+backend.sum(y_pred_f)+1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

names_model_v1 = sorted(glob.glob('drive/MyDrive/Data/model_saves/mymodel*v1*'))
names_model_v2 = sorted(glob.glob('drive/MyDrive/Data/model_saves/mymodel*v2*'))

dc_ls = []; bt_ls = []
for n1, n2 in zip(names_model_v1, names_model_v2):
    loaded_model_v1 = tf.keras.models.load_model(n1, compile=False)
    loaded_model_v2 = tf.keras.models.load_model(n2, compile=False)
    y_pred1 = loaded_model_v1.predict(X_new)
    y_pred2 = loaded_model_v2.predict(X_new)
    dc1 = dice_coef(y_new.astype('float32'),
                    y_pred1.reshape(y_new.shape).astype('float32'))
    dc2 = dice_coef(y_new.astype('float32'),
                    y_pred2.reshape(y_new.shape).astype('float32'))
    dc_ls.append((float(dc1), float(dc2)))
    bt = int(n1.split('s')[3].split('b')[0])
    bt_ls.append(bt)

df = DataFrame({'dice_coef v1': [i[0] for i in dc_ls],
                'dice_coef v2': [i[1] for i in dc_ls]}, index=[bt_ls])
                
df.to_csv('drive/MyDrive/df_dice.csv')