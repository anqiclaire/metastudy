########## balance after train test split FIXED on 0908

import numpy as np
import random
import scipy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

def prepare_struc(dataset, struc = 'Designs_2D_10Cells', mode = 'larger', times = 2):
    unit = 10; # 10*10 unit cell
    Xdata = dataset[struc]
    if mode == 'larger':
        result = np.zeros((Xdata.shape[0], unit*unit*times*times))
    else:
        result = np.zeros((Xdata.shape[0], unit*unit))
    count = 0;
    for vec in Xdata:
        mat = vecToMat(vec)
        if mode == 'larger':
            mat = np.tile(mat,[times,times])
        flat = mat.flatten()
        result[count] = flat
        count += 1
    return result

def vecToMat(vector):
    vlength = len(vector)
    mlength = 2 * findNSum(vlength)
    assert(mlength > 0)
    matrix = np.zeros((mlength, mlength))
    # first change the 1/8 triangle
    for i in range(vlength):
        if vector[i] != 0:
            seq = vlength - (i + 1) # turn 1,2... into 14,13...
            col = seq - findSmallerNSum(seq)
            row = findNSum(findSmallerNSum(seq))
            fillMat(matrix, row, col)
    return matrix

# symmetrically fill in matrix with 1
def fillMat(matrix, row, col):
    mlength = matrix.shape[0]
    matrix[row][col] = 1
    matrix[mlength - 1 - row][col] = 1
    matrix[row][mlength - 1 - col] = 1
    matrix[mlength - 1 - row][mlength - 1 - col] = 1
    matrix[mlength - 1 - col][row] = 1
    matrix[col][mlength - 1 - row] = 1
    matrix[mlength - 1 - col][mlength - 1 - row] = 1
    matrix[col][row] = 1


# find the nearest smaller NSum, e.g. 14 finds 10, 17 finds 15, 15 finds 15
def findSmallerNSum(num):
    assert(num >= 0)
    i = 0
    while findNSum(num - i) < 0:
        i += 1
    return num - i

# 15 in, 5 out; 10 in, 4 out
def findNSum(num):
    mySum = 0
    inc = 1
    while mySum < num:
        mySum += inc
        inc += 1
    if mySum == num:
        return inc - 1
    else:
        return -1 

def prepare_label(dataset, prop = 'PSV_bg', bg = 2):
    label = dataset[prop]
    result = np.zeros(label.shape)
    for r in range(label.shape[0]):
        start = np.zeros(label.shape[1])
        row = label[r]
        for i in range(len(row)):
            # if i > len(start):
            #     continue
            if (row[i] > 0):
                start[i] = 1
        result[r] = start
    open_cls = np.vstack(result[:,bg-1]) 
    close_cls = 1 - open_cls
    # print(np.concatenate((open_cls, close_cls), axis=1))
    return np.concatenate((open_cls, close_cls), axis=1)
    # return 'result' if a 16-long binary label needed; return result[:, bg] if only T/F (1/0) needed, np.vstack for reformatting a row-array to a column array


def load_data(raw = 'data_original.mat', wanted = 'data_original.mat', mode='train', balance='on', times = 2, prop = 'PSV_bg', bg = 2):
    # 'PSV_2_nz_cell.mat', 'data.mat'
    """
    Function to (download and) load the MNIST data
    :param mode: train or test
    :return: images and the corresponding labels
    """
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # if mode == 'train':
    #     x_train, y_train, x_valid, y_valid = mnist.train.images, mnist.train.labels, \
    #                                          mnist.validation.images, mnist.validation.labels
    #     x_train, _ = reformat(x_train, y_train)
    #     x_valid, _ = reformat(x_valid, y_valid)
    #     return x_train, y_train, x_valid, y_valid
    # elif mode == 'test':
    #     x_test, y_test = mnist.test.images, mnist.test.labels
    #     x_test, _ = reformat(x_test, y_test)
    # return x_test, y_test
    rawdata = scipy.io.loadmat(raw)
    wanted_data = scipy.io.loadmat(wanted)

    X_raw = prepare_struc(rawdata, times = times)# 2^15 * 100 unit cell structures
    # prop = 'PSV_bg'
    # y_raw = 1000*rawdata[prop]
    # print(y_raw)
    y_raw = prepare_label(rawdata, prop = prop, bg = bg)# 1*16 one-hot bg class
    # print(y_raw)
    # balance the open and close cases
    x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

    if balance == 'on':
        y_true_index = np.where(y_train_data[:,0] == 1)[0]
        # print(y_true_index.shape)
        y_true_num = y_true_index.shape[0]
        # print("y_true_num: %d" %(y_true_num))
        y_false_num = y_train_data.shape[0] - y_true_num
        # print("y_false_num: %d" %(y_false_num))
        y_insert_num = int(0.2*(y_false_num - y_true_num))
        # y_insert_num = 1
        # print("y_insert_num: %d" %(y_insert_num))
        np.random.seed(6) # set random seed for np
        y_insert_index = np.random.choice(y_true_index, y_insert_num)
        # print("y_insert_index: " + str(y_insert_index))
        y_insert = y_train_data[y_insert_index, :]
        # print("y_insert: " + str(y_insert))
        # print("y_insert.shape: " + str(y_insert.shape))
        # print("y_raw.shape: " + str(y_raw.shape))
        y_raw_new = np.append(y_train_data, y_insert, axis = 0)
        # print("y_raw_new.shape: " + str(y_raw_new.shape))
        # print("y_raw_new: " + str(y_raw_new))
        x_insert = x_train_data[y_insert_index, :]
        # print("x_insert: " + str(x_insert))
        # print("x_insert.shape: " + str(x_insert.shape))
        # print("x_raw.shape: " + str(X_raw.shape))
        x_raw_new = np.append(x_train_data, x_insert, axis = 0)
        # print("x_raw_new.shape: " + str(x_raw_new.shape))
        # print("x_raw_new: " + str(x_raw_new))
        # replace the old X_raw and y_raw
        x_train_data = x_raw_new
        y_train_data = y_raw_new

        ## for test data
        y_true_index_t = np.where(y_test_data[:,0] == 1)[0]
        # print(y_true_index.shape)
        y_true_num_t = y_true_index_t.shape[0]
        # print("y_true_num: %d" %(y_true_num))
        y_false_num_t = y_test_data.shape[0] - y_true_num_t
        # print("y_false_num: %d" %(y_false_num))
        y_insert_num_t = int(0.2*(y_false_num_t - y_true_num_t))
        # y_insert_num = 1
        # print("y_insert_num: %d" %(y_insert_num))
        np.random.seed(6) # set random seed for np
        y_insert_index_t = np.random.choice(y_true_index_t, y_insert_num_t)
        # print("y_insert_index: " + str(y_insert_index))
        y_insert_t = y_test_data[y_insert_index_t, :]
        # print("y_insert: " + str(y_insert))
        # print("y_insert.shape: " + str(y_insert.shape))
        # print("y_raw.shape: " + str(y_raw.shape))
        y_test_new = np.append(y_test_data, y_insert_t, axis = 0)
        # print("y_raw_new.shape: " + str(y_raw_new.shape))
        # print("y_raw_new: " + str(y_raw_new))
        x_insert_t = x_test_data[y_insert_index_t, :]
        # print("x_insert: " + str(x_insert))
        # print("x_insert.shape: " + str(x_insert.shape))
        # print("x_raw.shape: " + str(X_raw.shape))
        x_test_new = np.append(x_test_data, x_insert_t, axis = 0)
        # print("x_raw_new.shape: " + str(x_raw_new.shape))
        # print("x_raw_new: " + str(x_raw_new))
        # replace the old X_raw and y_raw
        x_test_data = x_test_new
        y_test_data = y_test_new
    # x_short_data, x_shooort_data, y_short_data, y_shooort_data = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
    # x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
    
    # for i in range(0, len(y_test_data)):
    #     print(np.where(y_raw[:,0] == 1)[0])
    #     if np.where(y_raw[:,0] == 1)[0] == True:
    #         count += 1
    # print(count)
    # x_train_data = X_raw
    # y_train_data = y_raw
    # x_test_data = X_raw[0:4]
    # y_test_data = y_raw[0:4]
    # print("x_test_data: " + str(x_test_data))
    # print("y_test_data: " + str(y_test_data))
    
    # get latent/prediction for wanted structures
    x_wanted_data = prepare_struc(wanted_data, times = times)
    y_wanted_data = prepare_label(wanted_data, prop = prop, bg = bg)

    if mode == 'train':
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_data, y_train_data, test_size=0.2, random_state=42)
        # x_train = X_raw
        # y_train = y_raw
        # x_valid = X_raw[0:4]
        # y_valid = y_raw[0:4]
        # print("x_train: " + str(x_train))
        # print("y_train: " + str(y_train))
        # print("x_valid: " + str(x_valid))
        # print("y_valid: " + str(y_valid))
        x_train, _ = reformat(x_train, y_train)
        x_valid, _ = reformat(x_valid, y_valid)
        # print("x_train after reformat: " + str(x_train))
        # print("x_valid after reformat: " + str(x_valid))
        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        x_test, _ = reformat(x_test_data, y_test_data)
        return x_test, y_test_data
    elif mode == 'wanted':
        x_wanted, _ = reformat(x_wanted_data, y_wanted_data)
    return x_wanted, y_wanted_data


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def reformat(x, y):
    """
    Reformats the data to the format acceptable for convolutional layers
    :param x: input array
    :param y: corresponding labels
    :return: reshaped input and labels
    """
    img_size, num_ch, num_class = int(np.sqrt(x.shape[-1])), 1, 2
    dataset = x.reshape((-1, img_size, img_size, num_ch)).astype(np.float32)
    # print("y[:,None]")
    # print(y[:,None])
    # print("np.arange(num_class)")
    # print(np.arange(num_class))
    # input("pause")
    # print(np.arange(num_class) == y[:, None])
    # print('----------------------------------------------------------------------')
    # labels = (np.arange(num_class) == y[:, None]).astype(np.float32)
    # print(labels)
    labels = y
    return dataset, labels


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch

def multi_label_hot(prediction, threshold=0.5):
    prediction = tf.cast(prediction, tf.float32)
    threshold = float(threshold)
    return tf.cast(tf.greater(prediction, threshold), tf.int64, name = 'correct_pred')


def plot_images(images, cls_true, cls_pred=None, title=None):
    """
    Create figure with 3x3 sub-plots.
    :param images: array of images to be plotted, (9, img_h*img_w)
    :param cls_true: corresponding true labels (9,)
    :param cls_pred: corresponding true labels (9,)
    """
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(np.squeeze(images[i]), cmap='binary')

        # Show true and predicted classes.
        # if cls_pred is None:
        #     ax_title = "True: {0}".format(cls_true[i])
        # else:
        #     ax_title = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # ax.set_title(ax_title)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        plt.suptitle(title, size=20)
    plt.show(block=False)


def plot_example_errors(images, cls_true, cls_pred, title=None):
    """
    Function for plotting examples of images that have been mis-classified
    :param images: array of all images, (#imgs, img_h*img_w)
    :param cls_true: corresponding true labels, (#imgs,)
    :param cls_pred: corresponding predicted labels, (#imgs,)
    """
    # Negate the boolean array.
    print(images.shape)
    incorrect = np.logical_not(np.equal(cls_pred, cls_true))
    print(incorrect[0])

    # Get the images from the test-set that have been
    # incorrectly classified.
    incorrect_images = images[incorrect]

    # Get the true and predicted classes for those images.
    cls_pred = cls_pred[incorrect]
    cls_true = cls_true[incorrect]

    # Plot the first 9 images.
    plot_images(images=incorrect_images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9],
                title=title)