# parameters
import os

root = os.path.expanduser('~/DATA/cifar10/')

model = 'MobileNetV2'
'''
MobileNet
MobileNetV2
...
'''
workers = 4

epochs = 90

batch_size = 128

learning_rate = 1e-4

momentum = 0.9

weight_decay = 1e-4

frequency = 20

use_cuda = True

num_classes = 10
