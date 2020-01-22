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


def InceptionI3d(input_shape, num_classes=400, spatial_squeeze=True, final_endpoint='Logits', dropout_keep_prob=0.9, name='inception_i3d'):
    """Connects the model to inputs.

    Args:
        inputs: Inputs to the model, which should have dimensions
            `batch_size` x `num_frames` x 224 x 224 x `num_channels`.
        is_training: whether to use training mode for snt.BatchNorm (boolean).
        dropout_keep_prob: Probability for the tf.nn.dropout layer (float in
            [0, 1)).

    Returns:
        A tuple consisting of:
        1. Network output at location `final_endpoint`.
        2. Dictionary containing all endpoints up to `final_endpoint`,
            indexed by endpoint name.

    Raises:
        ValueError: if `final_endpoint` is not recognized.
    """
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )
    if final_endpoint not in VALID_ENDPOINTS:
        raise ValueError('Unknown final endpoint %s' % final_endpoint)

    input_layer = tf.keras.layers.Input(shape=input_shape)
    end_points = {}
    end_point = 'Conv3d_1a_7x7'
    net = Unit3D(filters=64, kernel_size=[7, 7, 7], strides=[2, 2, 2], name=end_point)(input_layer)
    end_points[end_point] = net
    if final_endpoint == end_point: return net, end_points
    end_point = 'MaxPool3d_2a_3x3'
    net = tf.keras.layers.MaxPool3D(pool_size=[1,3,3], strides=[1,2,2], padding='SAME', name=end_point)(net)
    '''
    ksize: An int or list of ints that has length 1, 3 or 5. The size of the window for each dimension of the input tensor.
    strides: An int or list of ints that has length 1, 3 or 5. The stride of the sliding window for each dimension of the input tensor.
    
    pool_size: Tuple of 3 integers, factors by which to downscale (dim1, dim2, dim3). (2, 2, 2) will halve the size of the 3D input in each dimension.
    strides: tuple of 3 integers, or None. Strides values.
    '''
    end_points[end_point] = net
    if final_endpoint == end_point: return net, end_points
    end_point = 'Conv3d_2b_1x1'
    net = Unit3D(filters=64, kernel_size=[1, 1, 1], name=end_point)(net)
    end_points[end_point] = net
    if final_endpoint == end_point: return net, end_points
    end_point = 'Conv3d_2c_3x3'
    net = Unit3D(filters=192, kernel_size=[3, 3, 3], name=end_point)(net)
    end_points[end_point] = net
    if final_endpoint == end_point: return net, end_points
    end_point = 'MaxPool3d_3a_3x3'
    net = tf.keras.layers.MaxPool3D(pool_size=[1,3,3], strides=[1,2,2], padding='SAME', name=end_point)(net)
    end_points[end_point] = net
    if final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_3b'
    net = Inc([64, [96,128], [16,32], 32], 'Mixed_3b')(net)
    end_points[end_point] = net
    if final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_3c'
    net = Inc([128, [128,192], [32,96], 64], 'Mixed_3c')(net)
    end_points[end_point] = net
    if final_endpoint == end_point: return net, end_points

    end_point = 'MaxPool3d_4a_3x3'
    net = tf.keras.layers.MaxPool3D(pool_size=[3,3,3], strides=[2,2,2], padding='SAME', name=end_point)(net)
    end_points[end_point] = net
    if final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4b'
    net = Inc([192, [96,208], [16,48], 64], 'Mixed_4b')(net)
    end_points[end_point] = net
    if final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4c'
    net = Inc([160, [112,224], [24,64], 64], 'Mixed_4c')(net)
    end_points[end_point] = net
    if final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4d'
    net = Inc([128, [128,256], [24,64], 64], 'Mixed_4d')(net)
    end_points[end_point] = net
    if final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4e'
    net = Inc([112, [144,288], [32,64], 64], 'Mixed_4e')(net)
    end_points[end_point] = net
    if final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4f'
    net = Inc([256, [160,320], [32,128], 128], 'Mixed_4f')(net)
    end_points[end_point] = net
    if final_endpoint == end_point: return net, end_points

    end_point = 'MaxPool3d_5a_2x2'
    net = tf.keras.layers.MaxPool3D(pool_size=[2,2,2], strides=[2,2,2], padding='SAME', name=end_point)(net)
    end_points[end_point] = net
    if final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_5b'
    net = Inc([256, [160,320], [32,128], 128], 'Mixed_5b')(net)
    end_points[end_point] = net
    if final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_5c'
    net = Inc([384, [192,384], [48,128], 128], 'Mixed_5c')(net)
    end_points[end_point] = net
    if final_endpoint == end_point: return net, end_points

    end_point = 'Logits'
    with tf.name_scope('Logits'):
        net = tf.keras.layers.MaxPool3D(pool_size=[2,7,7], strides=[1,1,1], padding='VALID')(net)
        net = tf.keras.layers.Dropout(dropout_keep_prob)(net)
        logits = Unit3D(filters=num_classes, kernel_size=[1, 1, 1], activation=None, use_batch_norm=False, use_bias=True, name='Conv3d_0c_1x1')(net)
        if spatial_squeeze:
            logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
    averaged_logits = tf.reduce_mean(logits, axis=1)
    end_points[end_point] = averaged_logits
    if final_endpoint == end_point: return averaged_logits, input_layer, end_points

    end_point = 'Predictions'
    predictions = tf.keras.activations.softmax(averaged_logits)
    end_points[end_point] = predictions
    return predictions, end_points

if __name__ == '__main__':
    # model = tf.keras.Sequential(name = "discriminator")
    # model.add( tf.keras.layers.InputLayer((64,224,224,3)) )
    # model.add( Unit3D(64) )
    # model.add( Inc([64, [96,128], [16,32], 32], name = 'Inc') )
    # model.add( Inc([64, [96,128], [16,32], 32], name = 'Inc2') )
    # model.summary()

    averaged_logits, input_layer, end_points = InceptionI3d((64, 224, 224, 3), 400)
    model = tf.keras.models.Model(input_layer, end_points['Logits'])
    model.summary()

    # for layer in model.layers:
    #     print(layer.name)
    #     if 'Mixed' in layer.name:
    #         for layer_ in layer.layers:
    #             print('\t', layer_.name)

    # for weight in model.weights:
    #     print(weight.name)
    # rand = tf.random.normal([10,64,224,224,3])
    # print(model(rand))