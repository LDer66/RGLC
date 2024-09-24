import keras
from keras.datasets import mnist , fashion_mnist
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from numpy.testing import assert_array_almost_equal

def uniform_trans(n_class, noise_ratio):
    # uniform transition matrix
    assert (noise_ratio >= 0.) and (noise_ratio <= 1.0)
    trans = np.float64(noise_ratio) / np.float64(n_class - 1) * np.ones((n_class,n_class))
    np.fill_diagonal(trans,(np.float64(1) - np.float64(noise_ratio))*np.ones(n_class))
    # print(trans.sum(0))
    diag_idx = np.arange(n_class)
    # make sure that sum of every row is 0
    trans[diag_idx,diag_idx] = trans[diag_idx,diag_idx] + 1.0 - trans.sum(0)
    # assert_array_almost_equal(a,b,decimal=6)
    assert_array_almost_equal(trans.sum(axis=1),1,1)
    # print(diag_idx)
    return trans

def pair_trans(n_class, noise_ratio):
    # pair_transition matrix
    assert (noise_ratio >= 0.) and (noise_ratio <= 1.0)
    trans = (1.0 - np.float64(noise_ratio)) * np.eye(n_class)
    for i in range(n_class):
        trans[i-1,i] = np.float64(noise_ratio)
    assert_array_almost_equal(trans.sum(axis=1), 1, 1)
    return  trans

def inter_class_noisify(labels, trans,random_state=0):
    # flip classes according to transition matrix
    assert trans.shape[0] == trans.shape[1]
    # print(np.max(labels))
    # print(trans.shape[0])
    assert np.max(labels) < trans.shape[0]
    # assert torch.max(labels) < trans.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(trans.sum(axis=1), np.ones(trans.shape[1]))
    assert (trans >= 0.0).all()

    # trans = torch.from_numpy(trans)
    m = labels.shape[0]
    new_labels = labels.copy()
    # new_labels = torch.clone(labels)
    # random number generator
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        # i ranges from 0 to n_class-1
        i = labels[idx]
        flipped = flipper.multinomial(1, trans[i, :], 1)[0]
        # sample_label = torch.multinomial(trans[i,:],1)[0]
        new_labels[idx] = np.where(flipped == 1)[0]
        # new_labels[idx] = sample_label

    return new_labels

def noisify_p(labels,n_class,noise_ratio,noise_type, random_state=0):
    if noise_ratio > 0.0:
        if noise_type == 'uniform':
            print("Uniform noise")
            trans = uniform_trans(n_class,noise_ratio)
        elif noise_type == 'pair':
            print("Pair noise")
            trans = pair_trans(n_class,noise_ratio)
        else:
            print("Noise type not implemented")

        noisy_labels = inter_class_noisify(labels,trans,random_state)
        actual_noise_ratio = (noisy_labels != labels).mean()
        assert actual_noise_ratio > 0.0
        print("noise ratio:{:.2f}".format(noise_ratio))
        labels = noisy_labels
    else:
        print("noise ratio:0")
        trans = np.eye(n_class)

    return labels,trans


def get_data_for_class(images, labels, cls):
    if type(cls) == list:
        idx = np.zeros(labels.shape, dtype=bool)
        for c in cls:
            idx = np.logical_or(idx, labels == c)
    else:
        idx = (labels == cls)
    return images[idx], labels[idx]





