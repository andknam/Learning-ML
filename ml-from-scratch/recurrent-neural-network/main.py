import numpy as np 
import random

from rnn import RNN
from data import train_data, test_data

# create vocab
vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)

# assign indices to ea word
word_to_idx = { w : i for i, w in enumerate(vocab)}
idx_to_word = { i : w for i, w in enumerate(vocab)}

def createInputs(text):
    '''
    Returns array of one-hot vectors rep word in the input text string
    - text is a string
    - ea one-hot vector has shape (vocab_size, 1)
    '''
    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
    
    return inputs

def softmax(xs):
    # applies Softmax funct to input array
    return np.exp(xs) / sum(np.exp(xs))

# Initialize our RNN!
rnn = RNN(vocab_size, 2)

def processData(data, backprop=True):
    '''
    Returns the RNN's loss and accuracy for given data.
    - data is dict mapping text to True or False
    - backprop determines if the backward phase should be run
    '''
    items = list(data.items())
    random.shuffle(items)

    loss = 0
    num_correct = 0
    
    for x, y in items:
        inputs = createInputs(x)
        target = int(y)

        # Forward
        out, _ = rnn.forward(inputs)
        probs = softmax(out)

        # Calculate loss / accuracy
        loss -= np.log(probs[target])
        num_correct += int(np.argmax(probs) == target)

        if backprop:
            # Build dL/dy
            d_L_d_y = probs
            d_L_d_y[target] -= 1

            # Backward
            rnn.backprop(d_L_d_y)

    return loss / len(data), num_correct / len(data)

# Training loop
for epoch in range(1000):
    train_loss, train_acc = processData(train_data)

    if epoch % 100 == 99:
        print('--- Epoch %d' % (epoch + 1))
        print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

        test_loss, test_acc = processData(test_data, backprop=False)
        print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))
