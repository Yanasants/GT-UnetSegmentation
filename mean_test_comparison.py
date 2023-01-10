# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 16:38:14 2023

@author: Yana Santos
"""

import glob 
import numpy as np
from mean_functions import dice_coef_EV, dice_coef_girino, dice_coef_lung, jaccard_coef, calcula_metricas
from utils import load_images_array

TEMPO = []
ORIGINAL_SIZE = 4
NEW_SIZE = 4
for i in range(1,5):
    # Choose train folder TM40 ou TM46
    _folder = './imgs_testes_metricas/teste_' + str(i)
    # _folder = './dados_girino/TM46_40prod'
    
    GT_imgs = sorted(glob.glob(_folder + '/conjunto_a/*')) 
    pred_imgs = sorted(glob.glob(_folder + '/conjunto_b/*'))
    
    gt = load_images_array(GT_imgs, new_size = NEW_SIZE)
    pred = load_images_array(pred_imgs, new_size = NEW_SIZE)
    
    # VERSÃO EDUARDO E VITOR
    dice = dice_coef_EV(gt, pred)
    dice = str(dice).split(',')[0].split('(')[1]
    print(f'Coeficiente Dice (Eduardo e Vitor): {dice}')
    
    
    # VERSÃO ATUAL GIRINO
    # Calcula iou e dice para todas as imagens deste fold (i)
    dice_list = []
    dice_list = calcula_metricas(gt, pred)  
    
    #estatística
    mean = round(np.mean(dice_list), 4)
    print(f'Coeficiente Dice (Versão do Laboratório): {mean}')
    
    # VERSÃO SEGMENTAÇÃO PULMÃO
    dice = dice_coef_lung(gt, pred)
    dice = str(dice).split(',')[0].split('(')[1]
    print(f'Coeficiente Dice (Versão usada com as tomografias de pulmão): {dice}')