# train_kinetic_400_rgb_scratch

import tensorflow as tf
import numpy as np
import datetime
import os
import cv2

import I3D
import data

_BATCH_SIZE = 1

# get dataset
test_ds = data.get_ucf_101_rgb_test_dataset(_BATCH_SIZE)

# build model
I3D = I3D.InceptionI3d((64, 224, 224, 3), 101)

# checkpoint
checkpoint_dir = './checkpoints/ucf_101_rgb_scratch/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(I3D=I3D)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# tensorboard
test_matrix =  np.zeros([101,101])
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(f'logs/{current_time}/U101_RGB_S_TEST')

# test
for i, (names, videos, labels) in enumerate(test_ds):
    for video, label in zip(videos, labels):

        # split video into multiple sequence
        video_parts = []
        unit = (len(video)-64)/4
        for i in range(5):
            video_parts.append(video[int(unit * i) : int(unit * i + 64)])
        video_parts = tf.stack(video_parts)
        prediction = I3D(video_parts, training=False)
        prediction = tf.reduce_mean(prediction, 0)

        # record matrix
        prediction = tf.argmax(prediction).numpy()
        label = tf.argmax(label).numpy()
        test_matrix[label][prediction] += 1
        print(f'predict:{data.ucf_101_int2str(prediction):<10} \
                true:{data.ucf_101_int2str(label):<10} \
                {prediction==label and "true" or "false"}')

        # show video
        video = tf.cast( (video+1)/2*255 , tf.uint8).numpy()
        for frame in video:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
np.save('./test_matrix.npy', test_matrix)

test_matrix = np.load('./test_matrix.npy')
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
im = ax.imshow(test_matrix)
# We want to show all ticks...
ax.set_xticks(np.arange(len(data.ucf_101_labels)))
ax.set_yticks(np.arange(len(data.ucf_101_labels)))
# ... and label them with the respective list entries
ax.set_xticklabels(data.ucf_101_labels)
ax.set_yticklabels(data.ucf_101_labels)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
# set title and show
ax.set_title("test_matrix")
fig.tight_layout()
plt.show()