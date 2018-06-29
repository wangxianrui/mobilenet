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

epochs = 100

batch_size = 64

learning_rate = 1e-1

momentum = 0.9

weight_decay = 1e-5

frequency = 20

use_cuda = True

num_classes = 10

scheduler_step = [10, 30, 80, 120, 140]
