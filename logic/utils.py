__author__ = 'RodXander'

import math as m
import random as r
from decimal import Decimal as d

import numpy as np
from scipy import ndimage as nd


def convert_to_decimal(sequence):
    """Converts the sequence's elements to a decimal

    :param sequence: A numpy array to convert its elements to decimal
    :param diag: Optional flag to determine if the sequence to convert is a numpy matrix.
    Warning: This method will deliver a diagonal matrix with decimal values and the rest at zero!
    :return: A numpy array with its element converted to decimal
    """
    new_sequence = np.zeros(np.shape(sequence), dtype=d)
    __convert_to_decimal(new_sequence, sequence)
    return new_sequence


def __convert_to_decimal(new_sequence, old_sequence):
    for index in range(len(new_sequence)):
        if len(np.shape(new_sequence)) == 1:
            new_sequence[index] = d(str(old_sequence[index]))
        else:
            __convert_to_decimal(new_sequence[index], old_sequence[index])


def generate_train_test_sets(transition_pattern, ndimensions=2, max_values=1E2, min_values=-1E2, size_train_set=50, size_test_set=50):
    """Generates two sets of sequences of observations for training and testing of a HMM model.

    :param transition_pattern: Required sequence of tuples used to determine how to change states. Each tuple correspond to the
        state with the same index and its values determine how to make jumps between states. It's mandatory that the sum
        of the values in a tuple add up to 1. Notice that the last state must include an imaginary state that will serve
        to exit from it and terminate the sequence.
    :param ndimensions: Dimension of the vectors created.
    :param max_values: Optional maximum value a vector's component can have.
    :param min_values: Optional minimum value a vector's component can have.
    :param size_train_set: Optional number of vectors delivered in the set of training sequences.
    :param size_test_set: Optional number of vectors delivered in the set of testing sequences.
    :return: Four objects: the set of training sequences and the same set but specifying the states correspondent to
        every vector. Also the same for the testing set.
    """
    nstates = len(transition_pattern)
    states = [[r.uniform(min_values, max_values), r.uniform(min_values, max_values)] for i in range(nstates)]

    # Making sure the first element of each pair is the smaller one
    for pair in states:
        if pair[0] > pair[1]:
            aux = pair[1]
            pair[1] = pair[0]
            pair[0] = aux

    training = [[] for i in range(size_train_set)]
    training_states = [[] for i in range(size_train_set)]
    testing = [[] for i in range(size_test_set)]
    testing_states = [[] for i in range(size_test_set)]

    for i in range(max(size_train_set, size_test_set)):
        if i < size_train_set:
            curr_state = 0
            # Creating the sequence
            while curr_state < nstates:
                # Creating a single observation
                training[i].append([r.uniform(states[curr_state][0], states[curr_state][1]) for j in range(ndimensions)])
                training_states[i].append(curr_state)
                # Choosing the next state
                rand_value = r.uniform(0, 1)
                for j in range(len(transition_pattern[curr_state])):
                    if rand_value < np.sum(transition_pattern[curr_state][: j + 1]):
                        curr_state = j
                        break

        if i < size_test_set:
            curr_state = 0
            # Creating the sequence
            while curr_state < nstates:
                # Creating a single observation
                testing[i].append([r.uniform(states[curr_state][0], states[curr_state][1]) for j in range(ndimensions)])
                testing_states[i].append(curr_state)
                # Choosing the next state
                rand_value = r.uniform(0, 1)
                for j in range(len(transition_pattern[curr_state])):
                    if rand_value < np.sum(transition_pattern[curr_state][: j + 1]):
                        curr_state = j
                        break

    return training, training_states, testing, testing_states


def prod(sequence, dtype=int):
    """Computes the product of every element in the sequence

    :param sequence: Required sequence to find the product of its elements
    :return: The product of every element
    """
    res = dtype(1)
    for elem in sequence:
        res *= dtype(elem)

    return res


def remove_nan_rows(seq):
    """Remove the rows that contain inf or -inf

    :param seq: Sequence to process
    :return: A sequence without these rows
    """
    res = []
    for elem in seq:
        if float('-inf') in elem or float('inf') in elem:
            continue
        res.append(elem)

    return np.array(res)


def nroot(x, n):
    """Computes the n-root of x

    :param x: Number to find its root
    :param n: Depth of the root
    :return: The n-root of x
    """
    return m.exp(m.log(x) / float(n))


def log10(x):
    """Wrapper for the math.log10 function in order ot return -inf when log10(0) is called instead an exception.

    :param x: Value to find its logarithm
    :return: The logarithm of x.
    """
    try:
        res = m.log10(x)
    except ValueError:
        res = float('-inf')
    finally:
        return res


def mean(sequence):
    """Computes the mean of a sequence of vectors

    :param sequence: Required numpy array containing the vectors.
    :return: A vector containing the mean of the list
    """
    if len(np.shape(sequence)) == 1:
        mean_val = sequence.mean()

    elif len(np.shape(sequence)) == 1:
        mean_val = np.array([nd.mean(sequence[:, i]) for i in range(np.shape(sequence)[1])])
    else:
        raise ValueError('Not implemented for arrays of depth three or more')

    return mean_val


def delete_clusters_less_than(data, means, labels, less_than=1):
    """Delete all the cluster whose size (i.e. number of elements) is less than a value specified

    :param data: Required sequence of vector whose clusters were found
    :param means: Required sequence of means representing the clusters
    :param labels: Required sequence of labels representing where each data belong
    :param less_than: Optional value that implies that every cluster of size less than it will be removed and its data
    will be relocated to the remaining clusters
    :return:
    """
    if less_than < 1 or len(labels) <= less_than:
        return means, labels

    # Deleting clusters of small size. Notice that its elements will be relocated if they exist
    i = 0
    recompute_means = False

    while i < len(means):

        elems_in_cluster_i = np.count_nonzero(labels == i)

        if elems_in_cluster_i < less_than:
            if elems_in_cluster_i > 0:

                recompute_means = True  # Necessary to recompute means because some clusters are changed
                # Relocating the elements of cluster i
                for j in range(len(labels)):
                    if labels[j] != i:
                        continue

                    cluster, min_distance = None, float('inf')
                    for k in range(len(means)):
                        if k == i:
                            continue

                        distance = euclidean_norm(means[k], data[j])
                        if distance < min_distance:
                            min_distance = distance
                            cluster = k

                    labels[j] = cluster

            # Removing the centroid for the small cluster
            means = np.delete(means, i, axis=0)
            # Fix the index that point to a mean with an index bigger than the one deleted because when we removed it its
            # indexes drop by one unit
            labels[labels > i] -= 1
        else:
            i += 1

    if recompute_means:
        means = np.array([mean(data[labels == i]) for i in range(len(means))])

    return means, labels


def euclidean_norm(x1, x2):
    """Computes the euclidean distance between two points

    :param x1: Required first vector
    :param x2: Required second vector
    :return: The euclidean distance between the two points
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    return np.sqrt(((x1 - x2) ** 2).sum())