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
import statistics
import glob
import os

#Run this code 14-teen times changing stimulus_id from 0 to 13
stimulus_id = 13

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
make_dir_with_check('res_1pyramids')
make_dir_with_check('checkpoint_1pyramids')
for stim_name in stimulus:
    make_dir_with_check('data/' + stim_name)
    make_dir_with_check('res_1pyramids/' + stim_name)
    make_dir_with_check('checkpoint_1pyramids/' + stim_name)

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

X = np.load('data/students.np.npy')
Y = np.load('data/students_map_gray.np.npy')

import csv
stimulus_X = []
with open('data/students_stimulus.txt', newline='') as f:
    reader = csv.reader(f)
    stimulus_X = list(reader)


string_to_find = stimulus[stimulus_id]
print(my_model.summary())


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

def decoder5():
    filters = (64, 128, 256, 512, 512)
    chanDim = -1
    depth_out = 1

    dec = encoder5.predict(xx)
    print('decoder5')
    print(dec.shape)

    x = my_model.get_layer("block5_pool").output
    for f in filters[::-1]:
        # apply a CONV_TRANSPOSE => RELU => BN operation
        x = Conv2DTranspose(f, (3, 3), strides=2,
                            padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=chanDim)(x)

    x = Conv2DTranspose(depth_out, (3, 3), padding="same", name='decoder51_out')(x)
    outputs = Activation("sigmoid", name='decoder5_out')(x)
    decoder = Model(my_model.inputs, outputs, name="decoder")
    return decoder

def decoder4():
    filters = (64, 128, 256, 512)
    chanDim = -1
    depth_out = 1

    dec = encoder4.predict(xx)
    print('decoder4')
    print(dec.shape)

    x = my_model.get_layer("block4_pool").output
    for f in filters[::-1]:
        # apply a CONV_TRANSPOSE => RELU => BN operation
        x = Conv2DTranspose(f, (3, 3), strides=2,
                            padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=chanDim)(x)

    x = Conv2DTranspose(depth_out, (3, 3), padding="same", name='decoder41_out')(x)
    outputs = Activation("sigmoid", name='decoder4_out')(x)

    decoder = Model(my_model.inputs, outputs, name="decoder")
    return decoder

def decoder3():
    filters = (64, 128, 256)
    chanDim = -1
    depth_out = 1

    dec = encoder3.predict(xx)
    print('decoder3')
    print(dec.shape)

    x = my_model.get_layer("block3_pool").output
    for f in filters[::-1]:
        # apply a CONV_TRANSPOSE => RELU => BN operation
        x = Conv2DTranspose(f, (3, 3), strides=2,
                            padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=chanDim)(x)

    x = Conv2DTranspose(depth_out, (3, 3), padding="same", name='decoder31_out')(x)
    outputs = Activation("sigmoid", name='decoder3_out')(x)

    decoder = Model(my_model.inputs, outputs, name="decoder")
    return decoder



def decoder5_():
    attention = d5.output
    x = Conv2D(1, (3, 3), strides=1, padding="same")(attention)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=-1)(x)
    decoder = Model(my_model.inputs, x, name="decoder")
    return decoder


d5 = decoder5()
d4 = decoder4()
d3 = decoder3()
d345 = decoder5_()
print(d345.summary())
#exit()
vv = d345.predict(xx)
print(vv.shape)

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

path_to_checkpoints = 'checkpoint_1pyramids/' + stimulus[stimulus_id] + '/'
print(path_to_checkpoints)
list_of_files = glob.glob(path_to_checkpoints + '*.hdf5') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)

d345.load_weights(latest_file)
print("[INFO] making predictions...")
decoded = d345.predict(X_test)

def Average(lst):
    return sum(lst) / len(lst)


import cv2

cc_l = []
for i in range(0, Y_test.shape[0]):
    # grab the original image and reconstructed image
    print(i)
    original = (Y_test[i] * 255.0).astype("uint8")
    recon = (decoded[i] * 255).astype("uint8")

    if np.sum(original) > 0:
        cc_l.append(np.corrcoef(original.flatten(), recon.flatten())[0,1])

for i in range(0, Y_test.shape[0]):
    original = (Y_test[i] * 255.0).astype("uint8")
    recon = (decoded[i] * 255).astype("uint8")
    if np.sum(original) > 0:
        cv2.imwrite('res_1pyramids/' + stimulus[stimulus_id] + "/original" + str(i) + ".png", original)
        cv2.imwrite('res_1pyramids/' + stimulus[stimulus_id] + "/recon" + str(i) + ".png", recon)

print(round(Average(cc_l),3))
print(round(statistics.stdev(cc_l),3))


