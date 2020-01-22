import tensorflow as tf
# from tensorflow.keras.layers import MaxPool3D, AvgPool3D, Conv3D, BatchNormalization, Dropout, Activation
from tensorflow.keras.layers import MaxPool2D, AvgPool2D, Conv2D, BatchNormalization, Dropout, Activation
# MaxPool3D = tf.keras.layers.MaxPool3D
# Dropout = tf.keras.layers.Dropout


class GroupNormalization(tf.keras.layers.Layer):
    """Group normalization layer
    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. GN's computation is independent of batch sizes,
    and its accuracy is stable in a wide range of batch sizes
    # Arguments
        groups: Integer, the number of groups for Group Normalization.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape), axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.keras.backend.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = tf.keras.backend.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = tf.keras.backend.stack(group_shape)
        inputs = tf.keras.backend.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = tf.keras.backend.mean(inputs, axis=group_reduction_axes, keepdims=True)
        variance = tf.keras.backend.var(inputs, axis=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / (tf.keras.backend.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = tf.keras.backend.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = tf.keras.backend.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = tf.keras.backend.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = tf.keras.backend.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint': tf.keras.constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class Unit2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(1, 1), strides=(1, 1), activation='relu', use_batch_norm=True, use_bias=False, name='unit_2d', **kwargs):
        super(Unit2D, self).__init__(name=name, **kwargs)
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.conv2d = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='SAME', use_bias=use_bias)
        if use_batch_norm:          self.batchnorm = BatchNormalization()
        if activation is not None:  self.activation = Activation(activation)

    def call(self, inputs, training=None):
        net = self.conv2d(inputs)
        if self.use_batch_norm:         net = self.batchnorm(net, training=training)
        if self.activation is not None: net = self.activation(net)
        return net

class Inc(tf.keras.layers.Layer):
    def __init__(self, filters = [64, [96,128], [16,32], 32], name='Inc', **kwargs):
        super(Inc, self).__init__(name=name, **kwargs)
        self.branch_0 = Unit2D(filters=filters[0], kernel_size=[1, 1], name='Conv3d_0a_1x1')
        self.branch_1_a = Unit2D(filters=filters[1][0], kernel_size=[1, 1], name='Conv3d_0a_1x1')
        self.branch_1_b = Unit2D(filters=filters[1][1], kernel_size=[1, 1], name='Conv3d_0a_1x1')
        self.branch_2_a = Unit2D(filters=filters[2][0], kernel_size=[1, 1], name='Conv3d_0a_1x1')
        self.branch_2_b = Unit2D(filters=filters[2][1], kernel_size=[1, 1], name='Conv3d_0a_1x1')
        self.branch_3_a = MaxPool2D(pool_size=[3,3], strides=[1,1], padding='SAME', name='MaxPool3d_0a_3x3')
        self.branch_3_b = Unit2D(filters=filters[3], kernel_size=[1, 1], name='Conv3d_0a_1x1')

    def call(self, inputs, training=None):
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
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        return net

def InceptionI2D(input_shape, num_classes=400, dropout_rate=0.1, name='inception_i3d'):
    video_input = tf.keras.layers.Input(shape=input_shape)
    net = Unit2D(filters=64 , kernel_size=[7, 7], strides=[2, 2], name='Conv3d_1a_7x7')(video_input)
    net = MaxPool2D(pool_size=[3,3], strides=[2,2], padding='SAME', name='MaxPool3d_2a_3x3')(net)
    net = Unit2D(filters=64 , kernel_size=[1, 1], name='Conv3d_2b_1x1')(net)
    net = Unit2D(filters=192, kernel_size=[3, 3], name='Conv3d_2c_3x3')(net)
    net = MaxPool2D(pool_size=[3,3], strides=[2,2], padding='SAME', name='MaxPool3d_3a_3x3')(net)
    net = Inc([64,  [96, 128], [16, 32], 32 ], 'Mixed_3b')(net)
    net = Inc([128, [128,192], [32, 96], 64 ], 'Mixed_3c')(net)
    net = MaxPool2D(pool_size=[3,3], strides=[2,2], padding='SAME', name='MaxPool3d_4a_3x3')(net)
    net = Inc([192, [96, 208], [16, 48], 64 ], 'Mixed_4b')(net)
    net = Inc([160, [112,224], [24, 64], 64 ], 'Mixed_4c')(net)
    net = Inc([128, [128,256], [24, 64], 64 ], 'Mixed_4d')(net)
    net = Inc([112, [144,288], [32, 64], 64 ], 'Mixed_4e')(net)
    net = Inc([256, [160,320], [32,128], 128], 'Mixed_4f')(net)
    net = MaxPool2D(pool_size=[2,2], strides=[2,2], padding='SAME', name='MaxPool3d_5a_2x2')(net)
    net = Inc([256, [160,320], [32,128], 128], 'Mixed_5b')(net)
    net = Inc([384, [192,384], [48,128], 128], 'Mixed_5c')(net)
    with tf.name_scope('Logits'):
        net = AvgPool2D(pool_size=[7,7], strides=[1,1], padding='VALID')(net)
        net = Dropout(dropout_rate)(net)
        logits = Unit2D(filters=num_classes, kernel_size=[1, 1], activation=None, use_batch_norm=False, use_bias=True, name='Conv3d_0c_1x1')(net)
        logits = tf.squeeze(logits, [2,], name='SpatialSqueeze')
        # logits = AvgPool3D(pool_size=[7,1,1], strides=[1,1,1], padding='VALID')(logits)
        # logits = tf.keras.layers.Flatten()(logits)
        # logits = tf.keras.layers.Reshape([num_classes])(logits)
    averaged_logits = tf.reduce_mean(logits, axis=1)
    prediction = tf.keras.activations.softmax(averaged_logits)
    return tf.keras.Model(inputs=video_input, outputs=prediction)

if __name__ == '__main__':
    model = InceptionI2D((224, 224, 3), 400)
    model.summary()
    tf.keras.utils.plot_model(model, 'I3D_function.png', show_shapes=True)
