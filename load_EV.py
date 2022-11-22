"""
Describe: Software to load the model and apply the test with new images.
Authors: Eduardo Destefani Stefanato & Vitor Souza Premoli Pinto de Oliveira.
Contact: edustefanato@gmail.com
Date: 17/11/2022.
MIT License | Copyright (c) 2022 Eduardo Destefani Stefanato
"""
# imports
import matplotlib.pyplot as plt
import tensorflow as tf
#from image import X_new, X, y
from os import listdir
import numpy as np
"""
def display_multiple_img(images, rows=1, cols=1):
    figure, ax = plt.subplots(nrows=rows, ncols=cols)
    for ind, l in enumerate(images):
        ax.ravel()[ind].imshow(images[l], cmap='gray')
        ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()
"""
#att YS - need to load images from the directories
X = #TM40 images
X_new = #TM46 images

# Objects selection
names_model = sorted(listdir('model_saves/'))
tp_idx = tuple(range(len(names_model)))
print("Select your model's save {}:".format(tp_idx))
select = int(input())

for i, n in enumerate(names_model):
    if i == select:
        dir_name = names_model[i]

print('\nmodel: "{}"\n'.format(dir_name))
print('Select the data (0 or 1):\n\n 0 - training data;\n 1 - new data.')
sel = int(input())

# load selected model
try:
    dir_string = 'model_saves/{}'.format(dir_name)
    loaded_model = tf.keras.models.load_model(dir_string)
except:
    dir_string = 'model_saves/{}'.format(dir_name)
    loaded_model = tf.keras.models.load_model(dir_string, compile=False)

# view
chs_candidates = np.array([52,  57,  35, 41,  62, 
                              66, 53,  82,  30,  73])
if sel == 0:
    pred_X = loaded_model.predict(X)
    pred_candidates_X = chs_candidates # np.random.randint(1, X.shape[0], 10)
else:
    pred = loaded_model.predict(X_new)
    pred_candidates_new = chs_candidates # np.random.randint(1, X_new.shape[0], 10)
    gt_new = pred_candidates_new

gt = y
if sel == 0:
    preds = pred_X
    pred_candidates = pred_candidates_X
    vol = X
    gt_sel = gt
else:
    preds = pred
    pred_candidates = pred_candidates_new
    vol = X_new
    gt_sel = X_new

# predict new images
"""
total_images = 36
sample = preds[50:86]
images = {'Image'+str(i): sample[i] for i in range(total_images)}
display_multiple_img(images, 6, 6)
"""
# predict, mask of selected data (X or X_new)
plt.figure(figsize=(12,12))
for i in range(0,9,3):
    plt.subplot(3,3,i+1)

    plt.imshow(np.squeeze(vol[pred_candidates[i]]), cmap='gray')
    plt.axis('off')
    plt.title("Base Image")

    plt.subplot(3,3,i+2)
    plt.imshow(np.squeeze(gt_sel[pred_candidates[i]]), cmap='gray')
    plt.axis('off')
    plt.title("Mask")

    plt.subplot(3,3,i+3)
    plt.imshow(np.squeeze(preds[pred_candidates[i]]), cmap='gray')
    plt.axis('off')
    plt.title("Pridiction")
