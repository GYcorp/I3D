import tensorflow as tf


def Unit3D(filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu', use_batch_norm=True, use_bias=False, name='unit_3d'):
    net = tf.keras.Sequential(name=name)
    net.add( tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding='SAME', use_bias=use_bias) )
    if use_batch_norm: net.add( tf.keras.layers.BatchNormalization() )
    if activation is not None: net.add( tf.keras.layers.Activation(activation) )
    return net

class Inc(tf.keras.Model):
    def __init__(self, filters = [64, [96,128], [16,32], 32], name='Inc'):
        super(Inc, self).__init__()
        self._name = name
        self.branch_0 = Unit3D(filters=filters[0], kernel_size=[1, 1, 1], name='Conv3d_0a_1x1')
        self.branch_1_a = Unit3D(filters=filters[1][0], kernel_size=[1, 1, 1], name='Conv3d_0a_1x1')
        self.branch_1_b = Unit3D(filters=filters[1][1], kernel_size=[1, 1, 1], name='Conv3d_0a_1x1')
        self.branch_2_a = Unit3D(filters=filters[2][0], kernel_size=[1, 1, 1], name='Conv3d_0a_1x1')
        self.branch_2_b = Unit3D(filters=filters[2][1], kernel_size=[1, 1, 1], name='Conv3d_0a_1x1')
        self.branch_3_a = tf.keras.layers.MaxPool3D(pool_size=[3,3,3], strides=[1,1,1], padding='SAME', name='MaxPool3d_0a_3x3')
        self.branch_3_b = Unit3D(filters=filters[3], kernel_size=[1, 1, 1], name='Conv3d_0a_1x1')

    def call(self, inputs):
        # with tf.name_scope(self._name):
        with tf.name_scope('Branch_0'):
            branch_0 = self.branch_0(inputs)
        with tf.name_scope('Branch_1'):
            branch_1 = self.branch_1_a(inputs)
            branch_1 = self.branch_1_b(branch_1)
        with tf.name_scope('Branch_2'):
            branch_2 = self.branch_2_a(inputs)
            branch_2 = self.branch_2_b(branch_2)
        with tf.name_scope('Branch_3'):
            branch_3 = self.branch_3_a(inputs)
            branch_3 = self.branch_3_b(branch_3)
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        return net

class InceptionI3d(tf.keras.Model):
    def __init__(self, num_classes=400, dropout_keep_prob=0.9, name='InceptionI3d'):
        super(InceptionI3d, self).__init__()
        self._name = name
        self._dropout_keep_prob = dropout_keep_prob
        self.Conv3d_1a_7x7 = Unit3D(filters=64, kernel_size=[7, 7, 7], strides=[2, 2, 2], name='Conv3d_1a_7x7')
        self.MaxPool3d_2a_3x3 = tf.keras.layers.MaxPool3D(pool_size=[1,3,3], strides=[1,2,2], padding='SAME', name='MaxPool3d_2a_3x3')
        self.Conv3d_2b_1x1 = Unit3D(filters=64, kernel_size=[1, 1, 1], name='Conv3d_2b_1x1')
        self.Conv3d_2c_3x3 = Unit3D(filters=192, kernel_size=[3, 3, 3], name='Conv3d_2c_3x3')
        self.MaxPool3d_3a_3x3 = tf.keras.layers.MaxPool3D(pool_size=[1,3,3], strides=[1,2,2], padding='SAME', name='MaxPool3d_3a_3x3')
        self.Mixed_3b = Inc([64, [96,128], [16,32], 32], 'Mixed_3b')
        self.Mixed_3c = Inc([128, [128,192], [32,96], 64], 'Mixed_3c')
        self.MaxPool3d_4a_3x3 = tf.keras.layers.MaxPool3D(pool_size=[3,3,3], strides=[2,2,2], padding='SAME', name='MaxPool3d_4a_3x3')
        self.Mixed_4b = Inc([192, [96,208], [16,48], 64], 'Mixed_4b')
        self.Mixed_4c = Inc([160, [112,224], [24,64], 64], 'Mixed_4c')
        self.Mixed_4d = Inc([128, [128,256], [24,64], 64], 'Mixed_4d')
        self.Mixed_4e = Inc([112, [144,288], [32,64], 64], 'Mixed_4e')
        self.Mixed_4f = Inc([256, [160,320], [32,128], 128], 'Mixed_4f')
        self.MaxPool3d_5a_2x2 = tf.keras.layers.MaxPool3D(pool_size=[2,2,2], strides=[2,2,2], padding='SAME', name='MaxPool3d_5a_2x2')
        self.Mixed_5b = Inc([256, [160,320], [32,128], 128], 'Mixed_5b')
        self.Mixed_5c = Inc([384, [192,384], [48,128], 128], 'Mixed_5c')
        self.Logits_MaxPool3d = tf.keras.layers.MaxPool3D(pool_size=[2,7,7], strides=[1,1,1], padding='VALID')
        self.Logits_Dropout = tf.keras.layers.Dropout(self._dropout_keep_prob)
        self.Logits_Conv3d = Unit3D(filters=num_classes, kernel_size=[1, 1, 1], activation=None, use_batch_norm=False, use_bias=True, name='Conv3d_0c_1x1')
        # self.Logits_squeeze = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
        # self.Logits_reduce_mean = tf.reduce_mean(logits, axis=1)
        self.Predictions = tf.keras.layers.Softmax()

    def call(self, inputs, training=False):
        net = self.Conv3d_1a_7x7(inputs)
        net = self.MaxPool3d_2a_3x3(net)
        net = self.Conv3d_2b_1x1(net)
        net = self.Conv3d_2c_3x3(net)
        net = self.MaxPool3d_3a_3x3(net)
        net = self.Mixed_3b(net)
        net = self.Mixed_3c(net)
        net = self.MaxPool3d_4a_3x3(net)
        net = self.Mixed_4b(net)
        net = self.Mixed_4c(net)
        net = self.Mixed_4d(net)
        net = self.Mixed_4e(net)
        net = self.Mixed_4f(net)
        net = self.MaxPool3d_5a_2x2(net)
        net = self.Mixed_5b(net)
        net = self.Mixed_5c(net)
        with tf.name_scope('Logits'):
            net = self.Logits_MaxPool3d(net)
            net = self.Logits_Dropout(net, training=training)
            net = self.Logits_Conv3d(net)
            net = tf.squeeze(net, [2, 3], name='SpatialSqueeze')
            net = tf.reduce_mean(net, axis=1)
        net = self.Predictions(net)
        return net

# model = InceptionI3d(num_classes=400, spatial_squeeze=True, dropout_keep_prob=0.9, name='InceptionI3d')

model = tf.keras.Sequential(name = "InceptionI3d")
model.add( tf.keras.layers.InputLayer((64,224,224,3)) )
model.add( InceptionI3d(num_classes=400, spatial_squeeze=True, dropout_keep_prob=0.9, name='InceptionI3d') )
model.summary()