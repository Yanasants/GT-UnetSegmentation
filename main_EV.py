"""
Describe: Training and saving the model.
Authors: Eduardo Destefani Stefanato & Vitor Souza Premoli Pinto de Oliveira.
Contact: edustefanato@gmail.com
Date: 17/11/2022.
MIT License | Copyright (c) 2022 Eduardo Destefani Stefanato
"""
# imports
try:
    import model_v1 as un
    flag_v1 = True
except:
    import model_v2 as un
    flag_v1 = False
import numpy as np
from pandas import DataFrame
from IPython.display import clear_output
# from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from image import X_train, X_test, y_train, y_test, X_new
# from keras.losses import BinaryCrossentropy

def compile_model(batch_size, epochs):
    global X_train, y_train, X_test, y_test, flag_v1

    # start building ->
    fv = 1 if flag_v1 == True else 2
    weight_pt = "drive/MyDrive/Data/unet_v{}_ep{}bt{}_{}_weights.best.hdf5".format(fv,
                                                                                epochs, 
                                                                                batch_size, 
                                                                                'cxr_reg')
    # checkpoint and wight_pt scribe
    checkpoint = ModelCheckpoint(weight_pt, monitor='val_loss', verbose=1,
                                save_best_only=True, mode='min',
                                save_weights_only = True)
    # reduce learning rate 
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', 
                                        factor=0.5, patience=3, verbose=1, 
                                        mode='min', epsilon=0.0001,
                                        cooldown=2, min_lr=1e-6)
    # early stopping
    early = EarlyStopping(monitor="val_loss", mode="min", patience=15) 
    callbacks_list = [checkpoint, early, reduceLROnPlat]
    # compile
    un.model.compile(optimizer='Adam', loss= un.dice_coef_loss, 
                    metrics = [un.dice_coef])
    # fit model
    loss_history = un.model.fit(X_train, y_train,
                                batch_size = batch_size, epochs = epochs,
                                validation_data = (X_test, y_test),
                                callbacks = callbacks_list)
    # end building <-
    return loss_history, batch_size, epochs, un.model

def save_model(cv, trained_model):
    global X_new, flag_v1

    # save model
    fv = 1 if flag_v1 == True else 2
    # pred = un.model.predict(X_new)
    trained_model[3].save('drive/MyDrive/Data/mymodel{}_{}epochs{}bs_v{}_20220427'.format(cv, trained_model[2], 
                            trained_model[1], fv))
    # save log
    df1 = DataFrame({'epoch': trained_model[0].epoch})
    df2 = DataFrame(trained_model[0].history); df = df1.join(df2)
    saved_file = df.to_csv('drive/MyDrive/Data/history_mymodel{}epochs{}bs_v{}_20220427.csv'.format(trained_model[2], 
                            trained_model[1], fv), index=False)

    return saved_file

# Training model
# trained_model = compile_model(4, 80)
# saved_model = save_model(trained_model)
# clear_output()
