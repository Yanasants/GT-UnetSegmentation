import numpy as np
#from numpy import asarray
from sklearn.metrics import jaccard_score

def jaccard_coef(y_true, y_pred):
    
    # Recebe dois 2d-array de bool
    
    # array_true = asarray(y_true)
    # array_pred = asarray(y_pred)
    iou = jaccard_score(y_true.flatten(),y_pred.flatten())
    return iou


def dice_coef_2(y_true, y_pred):
    
    # Recebe dois 2d-array de bool
    
    iou = jaccard_coef(y_true, y_pred)
    dice = (2*iou)/(iou+1)
    return dice


def calcula_metricas(Y_test, predicao):
    
    # Recebe predicao e Y_test (no padrao keras, do tipo (78, 256, 256, 1) )
    
    # Reshape e transforma em boolean
    Y_test = Y_test[:,:,:,0]
    Y_test_bool = Y_test > 0.5 #
    
    # Reshape e transforma em boolean
    predicao = predicao[:,:,:,0] #
    predicao_bool = predicao > 0.5 #
    
    #iou = np.zeros(predicao.shape[0])
    #dice = np.zeros(predicao.shape[0])
    
    iou_list = []
    dice_list = []
    
    for i in range(predicao.shape[0]):
        iou_list.append(jaccard_coef(Y_test_bool[i], predicao_bool[i]))
        dice_list.append(dice_coef_2(Y_test_bool[i], predicao_bool[i]))
    
    return iou_list, dice_list