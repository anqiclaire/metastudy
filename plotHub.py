import tensorflow as tf
import numpy as np
from CNN_bg_utils import *
from CNN_bg_ops import *
from drawMM import drawMM
from conviz_utils import plot_conv_weights, plot_conv_output
import sys
import os
from glob import glob

# two options: 1. given 1*15 matrix; 2. given row number in some csv
def plotByMatOrRows(logs_path = "./logs", prop = "PSV_bg", bg = 7, t = 4):
    print("Initializing...")
    epochs = findEpoch(logs_path)
    # rows starts with 0, max is 32767
    csv_output_path = "%s_%d" %(prop, bg)
    xwantedcsv = csv_output_path + "/x_wanted.csv"
    with open(xwantedcsv, 'r') as f:
        rawxwanted = f.read().split('\n')
        if rawxwanted[-1] == '':
            del rawxwanted[-1]
    # loop through all x_wanted strings for w_max and w_min in the "layer" var
    w_min, w_max, saver, x = global_min_max(rawxwanted, logs_path, prop, bg, epochs, t)
    RorM = input("Input Row number or Matrix string? [R/M] ")
    if RorM.lower() == 'r':
        # ask for rows and call plot by rows
        rowsraw = input("Please input the row number in x_wanted.csv, start with 0, separate by whitespace: ").split()
        rows = []
        for r in rowsraw:
            try:
                rows.append(int(r))
            except:
                print("{} is not a valid row number, skipped.".format(r))
                continue
        plotByRows(rows, rawxwanted, saver, x, w_min, w_max, logs_path, prop, bg, epochs, t)
    elif RorM.lower() == 'm':
        # ask for matrix string and call plot by mat
        mat = input("Please input the 15 digit matrix string: ")
        plotByMat(mat, rawxwanted, saver, x, w_min, w_max, logs_path, prop, bg, epochs, t)

def plotByMat(mat, rawxwanted, saver, x, w_min, w_max, logs_path = "./logs", prop = "PSV_bg", bg = 7, epochs = 30, t = 4):
    if len(mat.strip()) != 15:
        print("The length of matrix string should be 15! Nothing returned!")
        return
    # search inside xwanted csv for the corresponding row number
    matlist = list(mat)
    csvquery = '.0,'.join(matlist) + '.0'
    if csvquery not in rawxwanted:
        print("The mat string provided is not in x_wanted! Nothing returned!")
        return
    row = rawxwanted.index(csvquery)
    print("Now processing matrix {}...".format(mat))
    plotByRows([row], rawxwanted, saver, x, w_min, w_max, logs_path, prop, bg, epochs, t)

def global_min_max(rawxwanted, logs_path = "./logs", prop = "PSV_bg", bg = 7, epochs = 30, t = 4):
    saver, x = getSaver()
    w_min = 1e9 # init value
    w_max = -1e9 # init value
    csv_output_path = "%s_%d" %(prop, bg)
    # generate weight and layers
    with tf.Session() as sess:
        saver.restore(sess, "{}/{}.bg{}.model.epoch{}.ckpt".format(logs_path, prop, bg, epochs - 1))
        x_wanted, y_wanted = load_data(mode='wanted', times=t, prop = prop, bg = bg)
        for row in range(len(rawxwanted)):
            ## generate all raw structures
            matstr = rawxwanted[row].split(',')
            matlist = []
            for n in matstr:
                matlist.append(int(float(n)))
            mat = np.asarray(matlist)
            conv_out = sess.run([tf.get_collection('layer')], feed_dict={x: x_wanted[row:row+1]})
            for c in conv_out[0]:
                w_min = min(np.min(c), w_min)
                w_max = max(np.max(c), w_max)
    return (w_min, w_max, saver, x)


def plotByRows(rows, rawxwanted, saver, x, w_min, w_max, logs_path = "./logs", prop = "PSV_bg", bg = 7, epochs = 30, t = 4):
    csv_output_path = "%s_%d" %(prop, bg)
    # generate weight and layers
    with tf.Session() as sess:
        saver.restore(sess, "{}/{}.bg{}.model.epoch{}.ckpt".format(logs_path, prop, bg, epochs - 1))
        x_wanted, y_wanted = load_data(mode='wanted', times=t, prop = prop, bg = bg)
        for seq,row in enumerate(rows):
            ## generate all raw structures
            matstr = rawxwanted[row].split(',')
            matlist = []
            # if csv_output_path/plots does not exist, mkdir
            if not os.path.isdir('{}/plots'.format(csv_output_path)):
                os.mkdir('{}/plots'.format(csv_output_path)) 
            if not os.path.isdir('{}/plots/{}_{}'.format(csv_output_path, seq, row)):
                os.mkdir('{}/plots/{}_{}'.format(csv_output_path, seq, row))
            for n in matstr:
                matlist.append(int(float(n)))
            mat = np.asarray(matlist)
            drawer = drawMM(mat, imageName = '{}/plots/{}_{}/raw_{}.png'.format(csv_output_path, seq, row, row))
            drawer.dupNbyM(4, 4)
            drawer.save()
            print("Row {} raw data saved as {}/plots/{}_{}/raw_{}.png".format(row, csv_output_path, seq, row, row))
            ## Plot convolutional layers using conviz
            # get weights of all convolutional layers
            # no need for feed dictionary here
            conv_weights = sess.run([tf.get_collection('conv_w')])
            for i, c in enumerate(conv_weights[0]):
                plot_conv_weights(c, '{}-{}'.format(row,i), PLOT_DIR = '{}/plots/{}_{}'.format(csv_output_path, seq, row))
            print("Row {} conv weights saved as {}/plots/{}_{}/conv_w_{}-0.png".format(row, csv_output_path, seq, row, row))
            # get output of all convolutional layers
            # here we need to provide an input image
            conv_out = sess.run([tf.get_collection('layer')], feed_dict={x: x_wanted[row:row+1]})
            for i, c in enumerate(conv_out[0]):
                # print("local max: {}, global max: {}".format(np.max(c), w_max))
                # print("local min: {}, global min: {}".format(np.min(c), w_min))
                plot_conv_output(c, '{}-{}'.format(row,i), PLOT_DIR = '{}/plots/{}_{}'.format(csv_output_path, seq, row), w_min = w_min, w_max = w_max)
            print("Row {} layer before relu saved as {}/plots/{}_{}/conv_layer_{}.png".format(row, csv_output_path, seq, row, row))

def plotByCSV(csvdir, logs_path = "./logs", prop = "PSV_bg", bg = 7, t = 4):
    epochs = findEpoch(logs_path)
    csv_output_path = "%s_%d" %(prop, bg)
    xwantedcsv = csv_output_path + "/x_wanted.csv"
    with open(xwantedcsv, 'r') as f:
        rawxwanted = f.read().split('\n')
        if rawxwanted[-1] == '':
            del rawxwanted[-1]
    # loop through all x_wanted strings for w_max and w_min in the "layer" var
    print("Initializing...")
    w_min, w_max, saver, x = global_min_max(rawxwanted, logs_path, prop, bg, epochs, t)
    with open(csvdir, 'r') as f:
        topRaw = f.read().split('\n')
    rows = []
    for num in topRaw:
        if num.isdigit():
            rows.append(int(num))
    plotByRows(rows, rawxwanted, saver, x, w_min, w_max, logs_path, prop, bg, epochs, t)

def findEpoch(logs_path):
    names = glob(logs_path + '/*.meta')
    epochs = -1
    for name in names:
        epochIndex = name.index('epoch')
        if epochIndex >= 0:
            dotIndex = name.index('.', epochIndex)
            if dotIndex >= 0:
                newEpochs = int(name[epochIndex + len('epoch'):dotIndex]) + 1
                if newEpochs > epochs:
                    epochs = newEpochs
    return epochs

def getSaver():
    bg = 7 # bandgap
    prop = 'PSV_bg' # choose the bandgap type
    t = 4 # times
    img_h = img_w = 10 * t  # images are 10x10
    img_size_flat = img_h * img_w  # 10x10=100, the total number of pixels
    n_classes = 2  # Number of classes
    n_channels = 1
    # Load MNIST data
    # x_train, y_train, x_valid, y_valid = load_data(mode='train', times = t, prop = prop, bg = bg)
    # Hyper-parameters
    logs_path = "./logs"  # path to the folder that we want to save the logs for Tensorboard
    lr = 0.001  # The optimization initial learning rate
    epochs = 50  # Total number of training epochs
    batch_size = 10 # Training batch size
    display_freq = 500  # Frequency of displaying the training results
    # 1st convolutional layer
    stride1 = 1  # The stride of the sliding window
    # conv1-1 size 8
    filter_size1_1 = 8  # Convolution filters are 2 x 2 pixels, smaller fig
    num_filters1_1 = 2  # There are 10 of these filters.
    # conv1-1 size 6
    filter_size1_2 = 6  # Convolution filters are 2 x 2 pixels, smaller fig
    num_filters1_2 = 2  # There are 10 of these filters.
    # conv1-1 size 4
    filter_size1_3 = 4  # Convolution filters are 2 x 2 pixels, smaller fig
    num_filters1_3 = 2  # There are 10 of these filters.
    # conv1-1 size 2
    filter_size1_4 = 2  # Convolution filters are 2 x 2 pixels, smaller fig
    num_filters1_4 = 2  # There are 10 of these filters.
    # 2nd Convolutional Layer
    filter_size2 = 8  # Convolution filters are 2 x 2 pixels.
    num_filters2 = 8  # There are 20 of these filters.
    stride2 = 1  # The stride of the sliding window
    # Fully-connected layer.
    # h1 = 128  # Number of neurons in fully-connected layer.
    h1 = 16  # Number of neurons in fully-connected layer = 64
    h2 = 1
    # Create the network graph
    # Placeholders for inputs (x), outputs(y)
    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, shape=[None, img_h, img_w, n_channels], name='X')
        y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')
    # # -----------------------------original version
    # conv1-1 pool 1-1
    # output size 
    conv1_1 = conv_layer(x, filter_size1_1, num_filters1_1, stride1, name='conv1_1')
    # output size 
    pool1_1 = max_pool(conv1_1, ksize=2, stride=1, name='pool1_1')
    # conv1-1 pool 1-2
    # output size 
    # conv1_2 = conv_layer(x, filter_size1_2, num_filters1_2, stride1, name='conv1_2')
    # # output size 
    # pool1_2 = max_pool(conv1_2, ksize=2, stride=1, name='pool1_2')
    # conv1-1 pool 1-3
    # # output size 
    # conv1_3 = conv_layer(x, filter_size1_3, num_filters1_3, stride1, name='conv1_3')
    # # output size
    # pool1_3 = max_pool(conv1_3, ksize=2, stride=1, name='pool1_3')
    # # conv1-1 pool 1-1
    # # output size
    # conv1_4 = conv_layer(x, filter_size1_4, num_filters1_4, stride1, name='conv1_4')
    # # output size
    # pool1_4 = max_pool(conv1_4, ksize=2, stride=1, name='pool1_4')
    pool1 = pool1_1
    # conv2 = conv_layer(pool1, filter_size2, num_filters2, stride2, name='conv2')
    # pool2 = max_pool(conv2, ksize=2, stride=2, name='pool2')
    layer_flat = flatten_layer(pool1)
    fc1 = fc_layer(layer_flat, h1, 'FC1', use_relu=False)
    fc2 = fc_layer(fc1, h2, 'FC2', use_relu=False)
    output_logits = fc_layer(fc2, n_classes, 'OUT', use_relu=False)
    # Define the loss function, optimizer, and accuracy
    with tf.variable_scope('Train'):
        with tf.variable_scope('Loss'):
            loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output_logits), name='loss')
            # loss = tf.losses.absolute_difference(y,output_logits)
            # loss = tf.reduce_sum(tf.math.abs(output_logits-y), name = 'loss')
        tf.summary.scalar('loss', loss)
        with tf.variable_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='Adam-op').minimize(loss)
            # optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, name='moment-op').minimize(loss)
        with tf.variable_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
            # incorrect_prediction = tf.not_equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='incorrect_pred')
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)
        with tf.variable_scope('Prediction'):
            # Network predictions
            cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')
    # Creating the op for initializing all variables
    init = tf.global_variables_initializer()
    # Merge all summaries
    merged = tf.summary.merge_all()
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    return (saver, x)

if __name__ == '__main__':
    # plotByMatOrRows()
    csvdir = input("Please input the csv dir of the selected row numbers: ")
    plotByCSV(csvdir)