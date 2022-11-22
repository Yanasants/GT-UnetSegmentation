"""
Describe: DSC software, receives the GTs and the segmentations.
Authors: Eduardo Destefani Stefanato & Vitor Souza Premoli Pinto de Oliveira.
Contact: edustefanato@gmail.com
Date: 17/11/2022.
MIT License | Copyright (c) 2022 Eduardo Destefani Stefanato
"""
# imports (not all modules are needed to avoid some unnecessary dependencies and errors)
from pandas import DataFrame
import tensorflow as tf
from model_v1 import *
from image import *
import numpy as np
import glob

def dice_coef(y_true, y_pred):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f*y_pred_f)
    return (2.*intersection + 1)/(backend.sum(y_true_f) + backend.sum(y_pred_f) + 1)
                 
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

names_model = sorted(glob.glob('Data/model_saves/mymodel*'))

dc_ls = []; bt_ls = []; bt = 0
for n in names_model:
    loaded_model = tf.keras.models.load_model(n, compile=False)
    y_pred = loaded_model.predict(X_new)
    dc = dice_coef(y_new.astype('float32'),
                   y_pred.reshape(y_new.shape).astype('float32'))
    dc_ls.append(float(dc))
    bt += 2
    bt_ls.append(bt)

df = DataFrame({'dice_coef': dc_ls}, index=[bt]); df