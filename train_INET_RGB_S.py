# train_kinetic_400_rgb_scratch

import tensorflow_datasets as tfds
import tensorflow as tf
import datetime
import os

import I3D

# from define import ucf_101_RGB_scratch_define
directory = os.path.dirname(os.path.abspath(__file__))
_BATCH_SIZE = 6
_EPOCH = 10000
_LEARNING_RATE = 0.001 / 32 * _BATCH_SIZE
_LEARNING_RATE = 0.001
# _MOMENTUM = 0.9

# get dataset
IMG_SIZE = 224

SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
print(splits)
(raw_train, raw_validation, raw_test), metadata = tfds.load(name="imagenet2012:5.0.0", with_info=True, split=list(splits), as_supervised=True)

get_label_name = metadata.features['label'].int2str
NUM_CLASSES = metadata.features['label'].num_classes

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = image / 255.0
    # Resize the image
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.reshape(image, (1, 224, 224, 3))
    image = tf.tile(image, (64,1,1,1))
    label = tf.one_hot([label],NUM_CLASSES)[0]
    return image, label

train_ds = raw_train.map(format_example).batch(_BATCH_SIZE)
validation_ds = raw_validation.map(format_example).batch(_BATCH_SIZE)
test_ds = raw_test.map(format_example).batch(_BATCH_SIZE)

if __name__ == "__main__" :
    dataset = train_ds
    for epoch in range(40):
        for i, (jpg_sequences, label_onehots) in enumerate(dataset):
            jpg_sequences = tf.cast( (jpg_sequences+1)/2*255 , tf.uint8)
            for jpg_sequence in jpg_sequences:
                for jpg in jpg_sequence:
                    jpg = cv2.cvtColor(jpg.numpy(), cv2.COLOR_RGB2BGR)
                    cv2.imshow('frame', jpg)
                    cv2.waitKey(1)