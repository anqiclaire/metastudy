import math
import os
import errno
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt

def get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]


def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)


def empty_dir(path):
    """
    Delete all files and folders in a directory
    :param path: string, path to directory
    :return: nothing
    """
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Warning: {}'.format(e))


def create_dir(path):
    """
    Creates a directory
    :param path: string
    :return: nothing
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def prepare_dir(path, empty=False):
    """
    Creates a directory if it soes not exist
    :param path: string, path to desired directory
    :param empty: boolean, delete all directory content if it exists
    :return: nothing
    """
    if not os.path.exists(path):
        create_dir(path)

    if empty:
        empty_dir(path)

# copy two functions from conviz.py to here, no need for another file
def plot_conv_weights(weights, name, channels_all=True, PLOT_DIR = './out/plots'):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    # plot_dir = os.path.join(PLOT_DIR, 'conv_w')
    # plot_dir = os.path.join(plot_dir, name)
    plot_dir = PLOT_DIR

    # create directory if does not exist, otherwise empty it
    prepare_dir(plot_dir)
    # prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            np.savetxt(os.path.join(plot_dir, 'conv_w_{}-{}-{}.csv'.format(name, channel, l)), img, delimiter = ',')
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='none', cmap='binary')#'seismic')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, 'conv_w_{}-{}.png'.format(name, channel)), bbox_inches='tight')
    plt.close()

def plot_conv_output(conv_img, name, PLOT_DIR = './out/plots', w_min = None, w_max = None):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    # plot_dir = os.path.join(PLOT_DIR, 'conv_layer')
    # plot_dir = os.path.join(plot_dir, name)
    plot_dir = PLOT_DIR
    
    # create directory if does not exist, otherwise empty it
    prepare_dir(plot_dir)
    # prepare_dir(plot_dir, empty=True)

    if w_min is None:
        w_min = np.min(conv_img)
    if w_max is None:
        w_max = np.max(conv_img)

    # get number of convolutional filters
    num_filters = conv_img.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate filters
    for l, ax in enumerate(axes.flat):
        # get a single image
        img = conv_img[0, :, :,  l]
        np.savetxt(os.path.join(plot_dir, 'conv_layer_{}-{}.csv'.format(name, l)), img, delimiter = ',')
        # put it on the grid
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='none', cmap='Greys')
        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
    # put w_max and w_min in the middle
    plt.text(-20, 50, 'max: {:.4f}\nmin: {:.4f}'.format(w_max, w_min), horizontalalignment='center', verticalalignment='center')
    # save figure
    plt.savefig(os.path.join(plot_dir, 'conv_layer_{}.png'.format(name)), bbox_inches='tight')
    plt.close()