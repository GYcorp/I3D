import os
import cv2
import tensorflow as tf
import time

# _SAVE_FPS = 25
# _MIN_SIDE_LENGTH = 256
_CROP_SIZE = 224
_CROP_LENG = 64
_FRAME_SKIP = 1
_BUFFER_SIZE = 10000

directory = os.path.dirname(os.path.abspath(__file__))

# 공간적으로
# 짧은면이 256 이 되도록 resize
# 무작위 224x224 crop


# 시간적으로
# 원하는 수의 프레임을 보장하여 시작 프레임을 선택
# 짧은비디오는 모델의 인풋에 맞도록 반복재생함


# 추가로
# 랜덤 left right flipping 를 비디오마다 적용함


# 테스팅시 224x224 center-crop 만 적용됨


def get_kinetic_400_rgb_train_dataset():
    pass

def get_kinetic_400_rgb_test_dataset():
    pass

def get_kinetic_400_flow_train_dataset():
    pass

def get_kinetic_400_flow_test_dataset():
    pass


def ucf_101_int2str(index):
    ucf_101_labels_filenames = os.path.join(directory, 'data', 'ucf', 'UCF-101_label.txt')
    ucf_101_labels = open(ucf_101_labels_filenames).read().splitlines()
    return ucf_101_labels[index]
ucf_101_labels_filenames = os.path.join(directory, 'data', 'ucf', 'UCF-101_label.txt')
ucf_101_labels = open(ucf_101_labels_filenames).read().splitlines()

def get_ucf_101_rgb_train_dataset(batch_size):
    tfrecord_filenames = os.path.join(directory, 'data', 'ucf', 'trainlist01.tfrecord')
    def _parse_batch(example_protos):    
        def _parse_video(example_proto):
            parsed_data = tf.io.parse_single_example(example_proto, {
                'video_name' : tf.io.FixedLenFeature([], tf.string, default_value=""),
                'jpg_sequence' : tf.io.FixedLenSequenceFeature([], tf.string, default_value=[""], allow_missing=True),
                'label' : tf.io.FixedLenFeature([], tf.string, default_value=""),
                'label_onehot' : tf.io.FixedLenSequenceFeature([], tf.int64, default_value=[0], allow_missing=True),
                'shape' : tf.io.FixedLenSequenceFeature([], tf.int64, default_value=[0], allow_missing=True)
            })
            def _parse_frame(_bytes):
                shape = parsed_data['shape']
                parsed_frame = tf.image.decode_jpeg(_bytes, 3)
                parsed_frame = tf.reshape(parsed_frame, shape[1:])
                parsed_frame = tf.image.convert_image_dtype(parsed_frame, tf.float32) # 0 ~ 1
                parsed_frame = parsed_frame * 2 - 1 # -1 ~ 1
                return parsed_frame

            # skip # repeat # crop_length # parse # crop_width_height # flip
            parsed_data['jpg_sequence'] = tf.strided_slice(parsed_data['jpg_sequence'], [0], [parsed_data['shape'][0]], [_FRAME_SKIP])
            parsed_data['jpg_sequence'] = tf.tile(parsed_data['jpg_sequence'], [tf.math.ceil(_CROP_LENG/parsed_data['shape'][0])])
            parsed_data['jpg_sequence'] = tf.image.random_crop(parsed_data['jpg_sequence'], (_CROP_LENG,))
            parsed_data['jpg_sequence'] = tf.map_fn(_parse_frame, parsed_data['jpg_sequence'], dtype=tf.float32)
            parsed_data['jpg_sequence'] = tf.image.random_crop(parsed_data['jpg_sequence'], (_CROP_LENG, _CROP_SIZE, _CROP_SIZE, 3))
            parsed_data['jpg_sequence'] = tf.cond(tf.random.normal([1]) > 0, lambda: tf.image.flip_left_right(parsed_data['jpg_sequence']), lambda: parsed_data['jpg_sequence'])

            return parsed_data['video_name'], parsed_data['jpg_sequence'] , parsed_data['label_onehot']

        example_protos = tf.map_fn(_parse_video, example_protos, dtype=(tf.string, tf.float32, tf.int64))
        return example_protos
        
    dataset = tf.data.TFRecordDataset(tfrecord_filenames)
    
    dataset = dataset.shuffle(_BUFFER_SIZE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_parse_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def get_ucf_101_rgb_validate_dataset(batch_size):
    tfrecord_filenames = os.path.join(directory, 'data', 'ucf', 'testlist01.tfrecord')
    def _parse_batch(example_protos):    
        def _parse_video(example_proto):
            parsed_data = tf.io.parse_single_example(example_proto, {
                'video_name' : tf.io.FixedLenFeature([], tf.string, default_value=""),
                'jpg_sequence' : tf.io.FixedLenSequenceFeature([], tf.string, default_value=[""], allow_missing=True),
                'label' : tf.io.FixedLenFeature([], tf.string, default_value=""),
                'label_onehot' : tf.io.FixedLenSequenceFeature([], tf.int64, default_value=[0], allow_missing=True),
                'shape' : tf.io.FixedLenSequenceFeature([], tf.int64, default_value=[0], allow_missing=True)
            })
            def _parse_frame(_bytes):
                shape = parsed_data['shape']
                parsed_frame = tf.image.decode_jpeg(_bytes, 3)
                parsed_frame = tf.reshape(parsed_frame, shape[1:])
                parsed_frame = tf.image.convert_image_dtype(parsed_frame, tf.float32) # 0 ~ 1
                parsed_frame = parsed_frame * 2 - 1 # -1 ~ 1
                return parsed_frame

            # skip # repeat # crop_length # parse # crop_width_height # flip
            parsed_data['jpg_sequence'] = tf.strided_slice(parsed_data['jpg_sequence'], [0], [parsed_data['shape'][0]], [_FRAME_SKIP])
            parsed_data['jpg_sequence'] = tf.tile(parsed_data['jpg_sequence'], [tf.math.ceil(_CROP_LENG/parsed_data['shape'][0])])
            parsed_data['jpg_sequence'] = tf.image.random_crop(parsed_data['jpg_sequence'], (_CROP_LENG,))
            parsed_data['jpg_sequence'] = tf.map_fn(_parse_frame, parsed_data['jpg_sequence'], dtype=tf.float32)
            parsed_data['jpg_sequence'] = tf.image.random_crop(parsed_data['jpg_sequence'], (_CROP_LENG, _CROP_SIZE, _CROP_SIZE, 3))
            parsed_data['jpg_sequence'] = tf.cond(tf.random.normal([1]) > 0, lambda: tf.image.flip_left_right(parsed_data['jpg_sequence']), lambda: parsed_data['jpg_sequence'])

            return parsed_data['video_name'], parsed_data['jpg_sequence'] , parsed_data['label_onehot']

        example_protos = tf.map_fn(_parse_video, example_protos, dtype=(tf.string, tf.float32, tf.int64))
        return example_protos
        
    dataset = tf.data.TFRecordDataset(tfrecord_filenames)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_parse_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
        
def get_ucf_101_rgb_test_dataset(batch_size):
    tfrecord_filenames = os.path.join(directory, 'data', 'ucf', 'testlist01.tfrecord')
    def _parse_batch(example_protos):    
        def _parse_video(example_proto):
            parsed_data = tf.io.parse_single_example(example_proto, {
                'video_name' : tf.io.FixedLenFeature([], tf.string, default_value=""),
                'jpg_sequence' : tf.io.FixedLenSequenceFeature([], tf.string, default_value=[""], allow_missing=True),
                'label' : tf.io.FixedLenFeature([], tf.string, default_value=""),
                'label_onehot' : tf.io.FixedLenSequenceFeature([], tf.int64, default_value=[0], allow_missing=True),
                'shape' : tf.io.FixedLenSequenceFeature([], tf.int64, default_value=[0], allow_missing=True)
            })
            def _parse_frame(_bytes):
                shape = parsed_data['shape']
                parsed_frame = tf.image.decode_jpeg(_bytes, 3)
                parsed_frame = tf.reshape(parsed_frame, shape[1:])

                offset_height = tf.cast((parsed_data['shape'][1]-_CROP_SIZE)/2, tf.int32)
                offset_width = tf.cast((parsed_data['shape'][2]-_CROP_SIZE)/2, tf.int32)
                parsed_frame = tf.image.crop_to_bounding_box(parsed_frame, offset_height, offset_width, _CROP_SIZE, _CROP_SIZE)

                parsed_frame = tf.image.convert_image_dtype(parsed_frame, tf.float32) # 0 ~ 1
                parsed_frame = parsed_frame * 2 - 1 # -1 ~ 1
                return parsed_frame

            # skip # repeat # crop_length # parse # crop_width_height # flip
            parsed_data['jpg_sequence'] = tf.strided_slice(parsed_data['jpg_sequence'], [0], [parsed_data['shape'][0]], [_FRAME_SKIP])
            parsed_data['jpg_sequence'] = tf.tile(parsed_data['jpg_sequence'], [tf.math.ceil(_CROP_LENG/parsed_data['shape'][0])])
            parsed_data['jpg_sequence'] = tf.map_fn(_parse_frame, parsed_data['jpg_sequence'], dtype=tf.float32)
            # _VIDEO_LENG = (parsed_data['shape'][0] * tf.cast(tf.math.ceil(_CROP_LENG/parsed_data['shape'][0]), tf.int64))
            # parsed_data['jpg_sequence'] = tf.image.random_crop(parsed_data['jpg_sequence'], (_VIDEO_LENG, _CROP_SIZE, _CROP_SIZE, 3))

            return parsed_data['video_name'], parsed_data['jpg_sequence'] , parsed_data['label_onehot']

        example_protos = tf.map_fn(_parse_video, example_protos, dtype=(tf.string, tf.float32, tf.int64))
        return example_protos
        
    dataset = tf.data.TFRecordDataset(tfrecord_filenames)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_parse_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
        



if __name__ == "__main__" :
    dataset = get_ucf_101_rgb_test_dataset(1)
    for epoch in range(40):
        for i, (video_names, jpg_sequences, label_onehots) in enumerate(dataset):
            jpg_sequences = tf.cast( (jpg_sequences+1)/2*255 , tf.uint8)
            # for video_name in video_names:
            #     print(f'{epoch:03} {i:06}', video_name.numpy())
            for jpg_sequence in jpg_sequences:
                for jpg in jpg_sequence:
                    jpg = cv2.cvtColor(jpg.numpy(), cv2.COLOR_RGB2BGR)
                    cv2.imshow('frame', jpg)
                    cv2.waitKey(1)