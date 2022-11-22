"""
Describe: Pre-processing of images and date augmentation.
Authors: Eduardo Destefani Stefanato & Vitor Souza Premoli Pinto de Oliveira.
Contact: edustefanato@gmail.com
Date: 17/11/2022.
MIT License | Copyright (c) 2022 Eduardo Destefani Stefanato
"""
# imports
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from PIL import Image
import numpy as np
import warnings
import random
import glob
warnings.simplefilter(action="ignore", category=FutureWarning)

# for make a 1-1 correspondance from mask to images
imagePath = sorted(glob.glob('drive/MyDrive/Data/lung_slices/*.tif'))
imagePath_gt = sorted(glob.glob('drive/MyDrive/Data/lung_gt/*.tif'))
imagePath_new = sorted(glob.glob('drive/MyDrive/Data/lung_new/*.tif'))
imagePath_new_gt = sorted(glob.glob('drive/MyDrive/Data/gt_new/*.tif'))

# reshape values
im_pote = []; out_pote = []; new_pote = []; new_gt = []
dim1 = 512; dim2 = 512

# load image and convert to and from NumPy array
for i in imagePath:
    a = np.array(Image.open(i))
    b = resize(a, (dim1, dim2))
    im_pote.append(b)

for i in imagePath_gt:
    c = np.array(Image.open(i))
    d = resize(c, (dim1, dim2))
    out_pote.append(d)
    
for i in imagePath_new:
    e = np.array(Image.open(i))
    f = resize(e, (dim1, dim2))
    new_pote.append(f)

for i in imagePath_new_gt:
    e = np.array(Image.open(i))
    f = resize(e, (dim1, dim2))
    new_gt.append(f)

X = np.stack(im_pote)
y = np.stack(out_pote)
X_new = np.stack(new_pote)
y_new = np.stack(new_gt)

# Random seed
key_random = random.SystemRandom()
seed = key_random.randint(0, 30)

# For training/splitdata
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state = seed)

# data augmentation
X_train = np.append(X_train, [np.fliplr(x) for x in X_train], axis=0)
y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)