# train_kinetic_400_rgb_scratch

import tensorflow as tf
import datetime
import os

import model
import data

# from define import ucf_101_RGB_scratch_define

directory = os.path.dirname(os.path.abspath(__file__))
_BATCH_SIZE = 6
_EPOCH = 10000
_LEARNING_RATE = 0.001 / 32 * _BATCH_SIZE
_LEARNING_RATE = 0.001
_MOMENTUM = 0.9
_CLASS_NUM = 5

# get dataset
tfrecord_filenames = os.path.join(directory, 'data', 'ucf', 'trainlist01 - small.tfrecord')
train_ds = data.get_ucf_dataset(tfrecord_filenames, _BATCH_SIZE)

tfrecord_filenames = os.path.join(directory, 'data', 'ucf', 'testlist01 - small.tfrecord')
validate_ds = data.get_ucf_dataset(tfrecord_filenames, _BATCH_SIZE)

# build model
I3D = model.InceptionI3d((64, 224, 224, 3), _CLASS_NUM)

# loss , optimizer
loss_object = tf.keras.losses.CategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam(learning_rate=_LEARNING_RATE)
# optimizer = tf.keras.optimizers.SGD(learning_rate=_LEARNING_RATE, momentum=_MOMENTUM)

# checkpoint
checkpoint_dir = './checkpoints/ucf_101_rgb_scratch/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, I3D=I3D)

# tensorboard
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

validation_loss = tf.keras.metrics.Mean(name='validation_loss')
validation_accuracy = tf.keras.metrics.CategoricalAccuracy(name='validation_accuracy')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(f'logs/{current_time}/') # checkpoints 에 저장해볼까?? 어차피 모델저장 한번하고 체크포인트도 거기에 종속된건데.. 관리하기 편하지 않을까?

# train
@tf.function
def train_step(videos, labels):
    with tf.GradientTape() as tape:
        predictions = I3D(videos, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, I3D.trainable_variables)
    optimizer.apply_gradients(zip(gradients, I3D.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
    return loss, tf.reduce_mean(tf.keras.metrics.categorical_accuracy(labels, predictions))

# validation
@tf.function
def validation_step(videos, labels):
    predictions = I3D(videos, training=False)
    loss = loss_object(labels, predictions)

    validation_loss(loss)
    validation_accuracy(labels, predictions)
    return loss, tf.reduce_mean(tf.keras.metrics.categorical_accuracy(labels, predictions))
    # TODO learning rate schedule

# test
@tf.function
def test_step(videos, labels):
    predictions = I3D(videos, training=False)
    loss = loss_object(labels, predictions)

    return loss, tf.reduce_mean(tf.keras.metrics.categorical_accuracy(labels, predictions))

save_index = 0
for epoch in range(_EPOCH):
    # train
    for i, (names, videos, labels) in enumerate(train_ds):
        loss, acc = train_step(videos, labels)
        print(f'epoch:{epoch} batch:{i*_BATCH_SIZE} loss: {loss.numpy()} acc: {acc.numpy()}')
        with summary_writer.as_default():
            tf.summary.scalar('train/train_loss', loss, step=save_index)
            tf.summary.scalar('train/train_acc', acc, step=save_index)
            save_index+=1

    # validation
    for i, (names, videos, labels) in enumerate(validate_ds):
        loss, acc = validation_step(videos, labels)
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