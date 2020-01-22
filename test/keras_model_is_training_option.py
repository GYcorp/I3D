# Q. class 로 만든 모델은 is training 이 자동으로 들어갈 까??

import tensorflow as tf

x = tf.ones([10,10], dtype=tf.float32)

class class_model(tf.keras.Model):
    def __init__(self, dropout_keep_prob=0.9, name='InceptionI3d'):
        super(class_model, self).__init__()
        self._name = name
        self._dropout_keep_prob = dropout_keep_prob
        self.Logits_Dropout = tf.keras.layers.Dropout(self._dropout_keep_prob)

    def call(self, inputs):
        net = self.Logits_Dropout(inputs)
        return net

def function_model(input_shape, dropout_keep_prob):
    video_input = tf.keras.layers.Input(shape=input_shape)
    net = tf.keras.layers.Dropout(dropout_keep_prob)(video_input)
    return tf.keras.Model(inputs=video_input, outputs=net)


model_1 = tf.keras.Sequential(name = "InceptionI3d")
model_1.add( tf.keras.layers.InputLayer((10,10)) )
model_1.add( class_model(dropout_keep_prob=0.9, name='InceptionI3d') )
print("model_1:", model_1(x, training=True))

model_2 = function_model((10,10), 0.9)
print("model_2:", model_2(x, training=False))

# A. 들어간다 ㅎㅎ 