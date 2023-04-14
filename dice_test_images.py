# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:31:10 2023

@author: Labic
"""
import glob
from utils_ys import create_folder, load_images_array
from utils_metrics import dice_coef

import os
import numpy as np
import cv2

TEMPO = []
ORIGINAL_SIZE = 850
NEW_SIZE = 512 #256

# Choose train folder TM40 ou TM46
_folder = './TM40_46Prod'
# _folder = './dados_girino/TM46_40prod'

exec_folder = '/Exec_2023-04-13-08-40-48.875605_new_dice/'
n_fold = 6

for i in range(n_fold):
    
    GT_imgs = sorted(glob.glob('./GT_Producao/*')) 
    pred_imgs = sorted(glob.glob('./outputs/Exec_2023-04-13-08-40-48.875605_new_dice/fold_'+str(i) +'/outputs_prod/*'))
    
    gt = load_images_array(GT_imgs, new_size = NEW_SIZE)
    pred = load_images_array(pred_imgs, new_size = NEW_SIZE)
    

    # VERSÃO SEGMENTAÇÃO PULMÃO
    # não deu certo com o imread
    dice = dice_coef(gt, pred)
    dice = str(dice).split(',')[0].split('(')[1]
    print(f'Coeficiente Dice: {dice}')
