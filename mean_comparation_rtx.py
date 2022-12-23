# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 09:04:14 2022

@author: Labic
"""

# COMPARANDO DIFERENTES CÁLCULOS DA MÉDIA

import glob 
import numpy as np
from mean_functions import dice_coef_EV, dice_coef_girino, dice_coef_lung, jaccard_coef, calcula_metricas
from utils import load_images_array

TEMPO = []
ORIGINAL_SIZE = 850
NEW_SIZE = 256

# Choose train folder TM40 ou TM46
_folder = './TM40_46Prod'
# _folder = './dados_girino/TM46_40prod'

GT_imgs = sorted(glob.glob(_folder + '/GT_Producao/*')) 
pred_imgs = sorted(glob.glob(_folder + '/outputs/Exec_2022-11-06-18-25-35.562984_com_loop_ES_50/fold_0/Outputs_prod/*'))

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

'''
OUTPUT
Coeficiente Dice (Eduardo e Vitor): 0.9242797248104198
Coeficiente Dice (Versão do Laboratório): 0.8682
Coeficiente Dice (Versão usada com as tomografias de pulmão): 0.9242796967128766
'''