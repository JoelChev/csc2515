from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
plt.ion()


def LoadData(fname):
    """Loads data from an NPZ file.

    Args:
        fname: NPZ filename.

    Returns:
        data: Tuple {inputs, target}_{train, valid, test}.
              Row-major, outer axis to be the number of observations.
    """
    npzfile = np.load(fname)

    inputs_train = npzfile['inputs_train'].T / 255.0
    inputs_valid = npzfile['inputs_valid'].T / 255.0
    inputs_test = npzfile['inputs_test'].T / 255.0
    target_train = npzfile['target_train'].tolist()
    target_valid = npzfile['target_valid'].tolist()
    target_test = npzfile['target_test'].tolist()

    num_class = max(target_train + target_valid + target_test) + 1
    target_train_1hot = np.zeros([num_class, len(target_train)])
    target_valid_1hot = np.zeros([num_class, len(target_valid)])
    target_test_1hot = np.zeros([num_class, len(target_test)])

    for ii, xx in enumerate(target_train):
        target_train_1hot[xx, ii] = 1.0

    for ii, xx in enumerate(target_valid):
        target_valid_1hot[xx, ii] = 1.0

    for ii, xx in enumerate(target_test):
        target_test_1hot[xx, ii] = 1.0

    inputs_train = inputs_train.T
    inputs_valid = inputs_valid.T
    inputs_test = inputs_test.T
    target_train_1hot = target_train_1hot.T
    target_valid_1hot = target_valid_1hot.T
    target_test_1hot = target_test_1hot.T
    return inputs_train, inputs_valid, inputs_test, target_train_1hot, target_valid_1hot, target_test_1hot


def Save(fname, data):
    """Saves the model to a numpy file."""
    print('Writing to ' + fname)
    np.savez_compressed(fname, **data)


def Load(fname):
    """Loads model from numpy file."""
    print('Loading from ' + fname)
    return dict(np.load(fname))


def DisplayPlot(train, valid, ylabel, eps, momentum, batchsize, nh0, nh1, nf0, nf1, number=0, cnn=True, hidden_layer=False, filter_number=False):
    """Displays training curve.

    Args:
        train: Training statistics.
        valid: Validation statistics.
        ylabel: Y-axis label of the plot.
    """
    plt.figure(number)
    plt.clf()
    train = np.array(train)
    valid = np.array(valid)
    plt.plot(train[:, 0], train[:, 1], 'b', label='Train')
    plt.plot(valid[:, 0], valid[:, 1], 'g', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.draw()
    plt.pause(0.0001)
    if cnn:
        if filter_number:
            if number == 0:
                title='CNN_CE,NF_0=' + str(nf0) + ',NF_1=' + str(nf1)
                plt.title(title)
                plt.savefig('./plots/filter_number/' + title + '.png')
            else:
                title = 'CNN_Acc,NF_0=' + str(nf0) + ',NF_1=' + str(nf1)
                plt.title(title)
                plt.savefig('./plots/filter_number/' + title + '.png')
        else:
            if number == 0:
                title= 'CNN_CE,Eps='+str(eps) + ',M=' + str(momentum) + ',BS=' +str(batchsize)
                plt.title(title)
                plt.savefig('./plots/cnn/' + title + '.png')
            else:
                title = 'CNN_Acc,Eps='+str(eps) + ',M=' + str(momentum) + ',BS=' +str(batchsize)
                plt.title(title)
                plt.savefig('./plots/cnn/' + title + '.png')
    else:
        if hidden_layer:
            if number == 0:
                title = 'NN_CE,NH_0=' + str(nh0) + ',NH_1=' + str(nh1)
                plt.title(title)
                plt.savefig('./plots/hidden_layer/' + title + '.png')
            else:
                title = 'NN_Acc,NH_0=' + str(nh0) + ',NH_1=' + str(nh1)
                plt.title(title)
                plt.savefig('./plots/hidden_layer/' + title + '.png')
        else:
            if number == 0:
                title ='NN_CE,Eps='+str(eps) + ',M=' + str(momentum) + ',BS=' +str(batchsize)
                plt.title(title)
                plt.savefig('./plots/nn/' + title + '.png')
            else:
                title = 'NN_Acc,Eps='+str(eps) + ',M=' + str(momentum) + ',BS=' +str(batchsize)
                plt.title(title)
                plt.savefig('./plots/nn/' + title + '.png')

