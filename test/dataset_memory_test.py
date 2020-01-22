# train_kinetic_400_rgb_scratch

import tensorflow as tf
import datetime
import os

import model
import data

# from define import ucf_101_RGB_scratch_define

_BATCH_SIZE = 9
_EPOCH = 10000
_LEARNING_RATE = 0.001 / 32 * _BATCH_SIZE
_MOMENTUM = 0.9

# get dataset
train_ds = data.get_ucf_101_rgb_train_dataset(_BATCH_SIZE)
test_ds = data.get_ucf_101_rgb_test_dataset(_BATCH_SIZE)

# build model
I3D = tf.keras.Sequential()
I3D.add( tf.keras.layers.InputLayer((64, 224, 224, 3)))
I3D.add( tf.keras.layers.Conv3D(10, (1,1,1), (1,1,1), padding='SAME'))
I3D.summary()


# test
@tf.function
def test_step(images):
    predictions = I3D(images, training=False)
    return predictions
    
for epoch in range(_EPOCH):
    # test
    for i, (names, videos, labels) in enumerate(test_ds):
        result = test_step(videos)
        print(result.numpy())