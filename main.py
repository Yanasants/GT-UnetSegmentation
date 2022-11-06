# Choose train folder TM40 ou TM46
_folder = './TM40_Original'
# _folder = './dados_girino/TM46_40prod'

norm_imgs = sorted(glob.glob(_folder + '/Norm_images/*')) 
GT_imgs = sorted(glob.glob(_folder + '/GT_images/*'))
	@@ -77,106 +77,106 @@

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)

use_batch_size = 4

epochs = 100   #100 

create_folder('./TM40_46Prod/outputs') #att

n_exec = 1
n_fold = 10 #10

exec_moment = str(datetime.datetime.now()).replace(':','-').replace(' ','-') #att

for i in range(n_fold):

    trainAug = Sequential([
       	preprocessing.RandomFlip("horizontal"),
       	preprocessing.RandomZoom(
       		height_factor=(-0.2, +0.2),
       		width_factor=(-0.2, +0.2)),
       	preprocessing.RandomRotation(0.1)
    ])

    valAug = Sequential([
       	preprocessing.RandomFlip("horizontal"),
       	preprocessing.RandomZoom(
       		height_factor=(-0.2, +0.2),
       		width_factor=(-0.2, +0.2)),
       	preprocessing.RandomRotation(0.1)
    ])


    time_train_1 = time.time()

    random.seed(time.time())
    seed_min = 0
    seed_max = 2**20
    SEED_1 = random.randint(seed_min, seed_max)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=SEED_1)


    # Data Augmentation 2 - 2022.05.07 Fazendo DA no conj de valid
    trainDS = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    trainDS = trainDS.repeat(3)
    trainDS = (
     	trainDS
     	.shuffle(use_batch_size * 100)
     	.batch(use_batch_size)
     	.map(lambda x, y: (trainAug(x), trainAug(y)), num_parallel_calls=AUTOTUNE)
     	.prefetch(tf.data.AUTOTUNE)
     )

    valDS = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    valDS = valDS.repeat(3)
    valDS = (
     	valDS
     	.shuffle(use_batch_size * 100)
     	.batch(use_batch_size)
     	.map(lambda x, y: (valAug(x), valAug(y)), num_parallel_calls=AUTOTUNE)
     	.prefetch(tf.data.AUTOTUNE)
     )

    N = X_train.shape[-1]

    # Models 

    # Unet effnet 
    # model = Unet(backbone_name='efficientnetb0', encoder_weights=None,
    #               input_shape=(None,None,N))

    #Unet vgg16
    model = Unet(backbone_name='vgg16', encoder_weights=None,
                  input_shape=(None,None,N))

    # Linknet resnet34 
    # model = Linknet(backbone_name='resnet34', encoder_weights=None,
                  # input_shape=(None,None,N))




    model.compile(optimizer=Adam(), loss=bce_jaccard_loss, metrics=[iou_score]) #bce_jaccard_loss

    history = model.fit(trainDS, 
              epochs=epochs, callbacks=callback, 
              validation_data=valDS)

    #att
    exec_folder_name = './TM40_46Prod/outputs/Exec_%s'%(exec_moment)
    create_folder(exec_folder_name)
    n_fold_folder_name = './%s'%(exec_folder_name) + "/fold_%i"%i
    create_folder(n_fold_folder_name)
    name_file = str(use_batch_size) + "_" + str(epochs) + "_exec_%s"%(exec_moment) + "_fold_%i"%i
    model.save(n_fold_folder_name + '/girino_%s.h5'%name_file)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(n_fold_folder_name + '/loss_%i.png'%i)
    plt.close()
    np.save(n_fold_folder_name + '/history_%i.npy'%i, history.history)
