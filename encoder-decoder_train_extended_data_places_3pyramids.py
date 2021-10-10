"""
Author: Tomasz Hachaj, 2021
Department of Signal Processing and Pattern Recognition
Institute of Computer Science in Pedagogical University of Krakow, Poland
https://sppr.up.krakow.pl/hachaj/
Data source:

https://drive.google.com/file/d/13VIyqFNzQ6zIGmWll9tEHjOdXp5R2GZt/view
https://drive.google.com/file/d/1U8bwYA8PgNuNYQnv5TNtR2az3AleyrEZ/view
https://drive.google.com/file/d/1h5udf2tB64q6-N3lEh0vDhvfIyDOD43N/view
"""

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.nasnet import preprocess_input
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
import os
from keras.callbacks import CSVLogger
from keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from vgg16_places_365 import VGG16_Places365

#Run this code 14-teen times changing stimulus_id from 0 to 13
stimulus_id = 12

stimulus = ['ALG_1_v1_Page_1.jpg', 'ALG_1_v2_Page_1.jpg', 'ALG_2_v1_Page_1.jpg', 'ALG_2_v2_Page_1.jpg', 'BIO_Page_1.jpg',
    'FIZ_WB1_Page_1.jpg', 'FIZ_WB2.jpg', 'FIZ_WB3_v1_Page_1.jpg', 'FIZ_WB3_v2_Page_1.jpg', 'FIZ_WB4_stereo_Page_1.jpg',
    'FIZ_WZORY_Page_1.jpg', 'rz 1_Page_1.jpg', 'rz 2_Page_1.jpg', 'rz 3_Page_1.jpg']

def make_dir_with_check(my_path):
    try:
        os.mkdir(my_path)
    except OSError:
        print(my_path + ' exists')
    else:
        print(my_path + ' created')

make_dir_with_check('data')
make_dir_with_check('res_places_3pyramids')
make_dir_with_check('checkpoint_places_3pyramids')
for stim_name in stimulus:
    make_dir_with_check('res_places_3pyramids/' + stim_name)
    make_dir_with_check('checkpoint_places_3pyramids/' + stim_name)

def enable_tensorflow():
    #Enables tensorflow on GPU
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
    return physical_devices

physical_devices = enable_tensorflow()
tf.compat.v1.disable_eager_execution()

my_model = VGG16(weights='imagenet', include_top=False)
for layer in my_model.layers[:]:
    layer.trainable = False

my_model_places = VGG16_Places365(weights='places', include_top=False)
for layer in my_model_places.layers[:]:
    layer.trainable = False
    layer._name = layer._name + str("_2x")


X = np.load('data/students.np.npy')
Y = np.load('data/students_map_gray.np.npy')

import csv
stimulus_X = []
with open('data/students_stimulus.txt', newline='') as f:
    reader = csv.reader(f)
    stimulus_X = list(reader)


extractor5 = Model(inputs=my_model.inputs,
                        outputs=my_model.get_layer("block5_pool").output)

extractor4 = Model(inputs=my_model.inputs,
                        outputs=my_model.get_layer("block4_pool").output)

extractor3 = Model(inputs=my_model.inputs,
                        outputs=my_model.get_layer("block3_pool").output)


xx = np.expand_dims(X[0], axis=0)
xx = preprocess_input(xx)

# build the encoder models
encoder5 = Model(extractor5.input, extractor5.output, name="encoder")
encoder4 = Model(extractor4.input, extractor4.output, name="encoder")
encoder3 = Model(extractor3.input, extractor3.output, name="encoder")

extractor5_places = Model(inputs=my_model_places.inputs,
                        outputs=my_model_places.get_layer("block5_pool_2x").output)

extractor4_places = Model(inputs=my_model_places.inputs,
                        outputs=my_model_places.get_layer("block4_pool_2x").output)

extractor3_places = Model(inputs=my_model_places.inputs,
                        outputs=my_model_places.get_layer("block3_pool_2x").output)


encoder5_places = Model(extractor5_places.input, extractor5_places.output, name="encoder")
encoder4_places = Model(extractor4_places.input, extractor4_places.output, name="encoder")
encoder3_places = Model(extractor3_places.input, extractor3_places.output, name="encoder")

def decoder5(my_modelxx, encoder5xx, name2):
    filters = (64, 128, 256, 512, 512)
    chanDim = -1
    depth_out = 1

    dec = encoder5xx.predict(xx)
    print('decoder5' + name2)
    print(dec.shape)

    x = my_modelxx.get_layer("block5_pool" + name2).output
    for f in filters[::-1]:
        # apply a CONV_TRANSPOSE => RELU => BN operation
        x = Conv2DTranspose(f, (3, 3), strides=2,
                            padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=chanDim)(x)

    x = Conv2DTranspose(depth_out, (3, 3), padding="same", name='decoder51_out'+name2)(x)
    outputs = Activation("sigmoid", name='decoder5_out'+name2)(x)
    decoder = Model(my_modelxx.inputs, outputs, name="decoder")
    return decoder

def decoder4(my_modelxx, encoder4xx, name2):
    filters = (64, 128, 256, 512)
    chanDim = -1
    depth_out = 1

    dec = encoder4xx.predict(xx)
    print('decoder4'+name2)
    print(dec.shape)

    x = my_modelxx.get_layer("block4_pool"+name2).output
    for f in filters[::-1]:
        # apply a CONV_TRANSPOSE => RELU => BN operation
        x = Conv2DTranspose(f, (3, 3), strides=2,
                            padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=chanDim)(x)

    x = Conv2DTranspose(depth_out, (3, 3), padding="same", name='decoder41_out'+name2)(x)
    outputs = Activation("sigmoid", name='decoder4_out'+name2)(x)

    decoder = Model(my_modelxx.inputs, outputs, name="decoder")
    return decoder

def decoder3(my_modelxx, encoder3xx, name2):
    filters = (64, 128, 256)
    chanDim = -1
    depth_out = 1

    dec = encoder3xx.predict(xx)
    print('decoder3'+name2)
    print(dec.shape)

    x = my_modelxx.get_layer("block3_pool"+name2).output
    for f in filters[::-1]:
        # apply a CONV_TRANSPOSE => RELU => BN operation
        x = Conv2DTranspose(f, (3, 3), strides=2,
                            padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=chanDim)(x)

    x = Conv2DTranspose(depth_out, (3, 3), padding="same", name='decoder31_out'+name2)(x)
    outputs = Activation("sigmoid", name='decoder3_out'+name2)(x)

    decoder = Model(my_modelxx.inputs, outputs, name="decoder")
    return decoder



def decoder345_v2():
    attention = Concatenate()([d4.output, d5.output, d3_places.output, d4_places.output, d5_places.output])
    x = Conv2D(1, (3, 3), strides=1, padding="same")(attention)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=-1)(x)
    decoder = Model(my_model.inputs, x, name="decoder")
    return decoder


def decoder345_v3():
    attention = Concatenate()([d5.output, d5_places.output])
    x = Conv2D(1, (3, 3), strides=1, padding="same")(attention)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=-1)(x)

    decoder = Model([my_model.input, my_model_places.input], x, name="decoder")
    return decoder


d5 = decoder5(my_model, encoder5_places, "")
d4 = decoder4(my_model, encoder4_places, "")
#d3 = decoder3(my_model, encoder3_places, "")

d5_places = decoder5(my_model_places, encoder5_places, "_2x")
d4_places = decoder4(my_model_places, encoder4_places, "_2x")
#d3_places = decoder3(my_model_places, encoder3_places, "_2x")

d345 = decoder345_v3()
print(d345.summary())
vv = d345.predict([xx,xx])
print(vv.shape)
#exit()

strings = sum(stimulus_X, [])
substring = stimulus[stimulus_id]
indices_train = [i for i, s in enumerate(strings) if substring not in s]
indices_test = [i for i, s in enumerate(strings) if substring in s]
print(len(indices_train))
print(len(indices_test))

print(len(strings))
X_train = X[indices_train, :, :, :]
X_test = X[indices_test, :, :, :]

Y_train = Y[indices_train, :, :, np.newaxis]
Y_test = Y[indices_test, :, :, np.newaxis]

X_train = X_train.astype("float32") / 255.0
Y_train = Y_train.astype("float32")

X_test = X_test.astype("float32") / 255.0
Y_test = Y_test.astype("float32")


X_2 = np.load('data/images.np.npy')
Y_2 = np.load('data/salMap_gray.np.npy')
X_2 = X_2[:, :, :, :]
Y_2 = Y_2[:, :, :, np.newaxis]

X_2 = X_2.astype("float32") / 255.0
Y_2 = Y_2.astype("float32")

print(X_train.shape)
print(X_2.shape)
X_train = np.concatenate((X_train, X_2), axis=0)
Y_train = np.concatenate((Y_train, Y_2), axis=0)



# checkpoint
latent_size = 512
EPOCHS = 10
BS = 16


path_to_checkpoints = "checkpoint_places_3pyramids/" + stimulus[stimulus_id]
DataFile = 'results.txt'
csv_logger = CSVLogger(path_to_checkpoints + "/" + DataFile + '.log')

# checkpoint
filepath= path_to_checkpoints + "/improvement-{epoch:02d}-{loss:.5f}-{val_loss:.5f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max', save_weights_only=True)

learning_rate_step = 10
def lr_scheduler(epoch, lr):
    if epoch % learning_rate_step == 0 and epoch > 1:
        lr = lr * 0.1
    print(lr)
    return lr


callbacks_list = [checkpoint, LearningRateScheduler(lr_scheduler, verbose=1), csv_logger]


def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

def custom_loss(y_true, y_pred):
    #loss1 = binary_crossentropy(y_true, d3.output)
    #loss2 = binary_crossentropy(y_true, d4.output)
    loss3 = binary_crossentropy(y_true, d5.output)
    loss4 = binary_crossentropy(y_true, d345.output)

    #loss1p = binary_crossentropy(y_true, d3_places.output)
    #loss2p = binary_crossentropy(y_true, d4_places.output)
    loss3p = binary_crossentropy(y_true, d5_places.output)

    return (loss3 + loss3p + loss4) / 3.0

opt = Adam(lr=1e-2)
d345.compile(loss=custom_loss, optimizer=opt)

# train the model
H = d345.fit(
    x = [X_train, X_train], y = Y_train,
    validation_data=([X_test, X_test], Y_test),
    epochs=EPOCHS,
    batch_size=BS,
    callbacks=callbacks_list)
