# kinetic 400 RGB scratch
class kinetic_400_RGB_scratch_define():
    _IMAGE_SIZE = 224

    _EPOCH = 100
    _BATCH = 1

    _LEARNING_RATE = 0.02 # 10x reduction of learning rate when validation loss saturated.
    _MOMENTUM = 0.9

# kinetic 400 FLOW scratch
class kinetic_400_FLOW_scratch_define():
    _IMAGE_SIZE = 224

    _EPOCH = 100
    _BATCH = 1

    _LEARNING_RATE = 0.02 # 10x reduction of learning rate when validation loss saturated.
    _MOMENTUM = 0.9

# UCF 101 RGB scratch
class UCF_101_RGB_scratch_define():
    _IMAGE_SIZE = 224

    _EPOCH = 100
    _BATCH = 1

    _LEARNING_RATE = 0.02 # 10x reduction of learning rate when validation loss saturated.
    _MOMENTUM = 0.9

# UCF 101 FLOW scratch
class UCF_101_FLOW_scratch_define():
    _IMAGE_SIZE = 224

    _EPOCH = 100
    _BATCH = 1

    _LEARNING_RATE = 0.02 # 10x reduction of learning rate when validation loss saturated.
    _MOMENTUM = 0.9