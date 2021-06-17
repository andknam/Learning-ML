from keras.datasets import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

# mnist takes care of stuff for us
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# use the first 1000 testing examples
test_images[:1000]
test_labels[:1000]

conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10

def forward(image, label):
    '''
    Complete a forward pass of CNN and calculate the accuracy and cross-entropy loss
    - image is a 2d numpy array
    - label is a digit
    '''
    # transform image from [0, 255] to [-0.5, 0.5] --> easier to work with
    # this is standard practice
    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    # calculate cross-entropy loss and accuracy
    loss = -np.log(out[label])
    if np.argmax(out) == label:
        acc = 1
    else:
        acc = 0

    return out, loss, acc

print('MNIST CNN intialized!')

loss = 0
num_correct = 0
for i, (image, label) in enumerate(zip(test_images, test_labels)):
    # do forward pass
    _, l, acc = forward(image, label)
    loss += l
    num_correct += acc

    # print stats every 100 steps
    if i % 100 == 99:
        print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (i + 1, loss / 100, num_correct))
        loss = 0
        num_correct = 0