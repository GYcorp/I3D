# train_kinetic_400_rgb_scratch

import tensorflow_datasets as tfds
import tensorflow as tf
import datetime
import os

import model_2D
import VGG16
import data

# from define import ucf_101_RGB_scratch_define
directory = os.path.dirname(os.path.abspath(__file__))
_BATCH_SIZE = 6
_EPOCH = 10000
_LEARNING_RATE = 0.001 / 32 * _BATCH_SIZE
_LEARNING_RATE = 0.001
_MOMENTUM = 0.9

# get dataset
IMG_SIZE = 224

SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
(raw_train, raw_validation, raw_test), metadata = tfds.load(name="tf_flowers", with_info=True, split=list(splits), as_supervised=True)

get_label_name = metadata.features['label'].int2str
NUM_CLASSES = metadata.features['label'].num_classes

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = image / 255.0
    # Resize the image
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.reshape(image, (224, 224, 3))
    label = tf.one_hot([label],NUM_CLASSES)[0]
    return image, label

train_ds = raw_train.map(format_example).batch(_BATCH_SIZE)
validation_ds = raw_validation.map(format_example).batch(_BATCH_SIZE)
test_ds = raw_test.map(format_example).batch(_BATCH_SIZE)

# build model
I2D = model_2D.InceptionI2D((224, 224, 3), NUM_CLASSES)
# I2D = VGG16.VGG16((224, 224, 3), NUM_CLASSES)

# loss , optimizer
loss_object = tf.keras.losses.CategoricalCrossentropy()
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.SGD(learning_rate=_LEARNING_RATE, momentum=_MOMENTUM)

# checkpoint
checkpoint_dir = './checkpoints/flower_rgb_scratch_2d/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, I2D=I2D)

# tensorboard
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

validation_loss = tf.keras.metrics.Mean(name='validation_loss')
validation_accuracy = tf.keras.metrics.CategoricalAccuracy(name='validation_accuracy')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(f'logs/{current_time}/') # checkpoints 에 저장해볼까?? 어차피 모델저장 한번하고 체크포인트도 거기에 종속된건데.. 관리하기 편하지 않을까?

# train
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = I2D(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, I2D.trainable_variables)
    optimizer.apply_gradients(zip(gradients, I2D.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
    return loss, tf.reduce_mean(tf.keras.metrics.categorical_accuracy(labels, predictions))

# validation
@tf.function
def validation_step(images, labels):
    predictions = I2D(images, training=False)
    loss = loss_object(labels, predictions)

    validation_loss(loss)
    validation_accuracy(labels, predictions)
    return loss, tf.reduce_mean(tf.keras.metrics.categorical_accuracy(labels, predictions))
    # TODO learning rate schedule

# test
@tf.function
def test_step(images, labels):
    predictions = I2D(images, training=False)
    loss = loss_object(labels, predictions)

    return loss, tf.reduce_mean(tf.keras.metrics.categorical_accuracy(labels, predictions))

save_index = 0
for epoch in range(_EPOCH):
    # train
    for i, (images, labels) in enumerate(train_ds):
        loss, acc = train_step(images, labels)
        print(f'epoch:{epoch} batch:{i*_BATCH_SIZE} loss: {loss.numpy()} acc: {acc.numpy()}')
        with summary_writer.as_default():
            tf.summary.scalar('train/train_loss', loss, step=save_index)
            tf.summary.scalar('train/train_acc', acc, step=save_index)
            save_index+=1

    # validation
    for i, (images, labels) in enumerate(validation_ds):
        loss, acc = validation_step(images, labels)
        print(f'[validate] epoch:{epoch} batch:{i*_BATCH_SIZE} loss: {loss.numpy()} acc: {acc.numpy()}')
    with summary_writer.as_default():
        tf.summary.scalar('validation/validation_loss', validation_loss.result(), step=save_index)
        tf.summary.scalar('validation/validation_acc', validation_accuracy.result(), step=save_index)

    # log
    template = '[result] Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
        train_loss.result(), 
        train_accuracy.result()*100,
        validation_loss.result(), 
        validation_accuracy.result()*100))

    train_loss.reset_states()
    train_accuracy.reset_states()
    validation_loss.reset_states()
    validation_accuracy.reset_states()

    # save
    if (epoch+1) % 50 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)