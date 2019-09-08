import tensorflow as tf
import numpy as np
from CNN_bg_utils import *
from CNN_bg_ops import *
import sys
import os

# usage: python CNN_bg.py train/no_train [epoch_for_test]
trainFlag = False
epoch_for_test = None
if len(sys.argv) != 2:
    raise Exception("usage: python CNN_bg.py train/no_train")
elif (sys.argv[1] == 'train'):
    trainFlag = True

# Data Dimensions
# img_h = img_w = 28  # MNIST images are 28x28
# img_size_flat = img_h * img_w  # 28x28=784, the total number of pixels
# n_classes = 10  # Number of classes, one class per digit
# n_channels = 1

bg = 7 # bandgap
prop = 'PSV_bg' # choose the bandgap type
t = 4 # times
img_h = img_w = 10 * t  # images are 10x10
img_size_flat = img_h * img_w  # 10x10=100, the total number of pixels
n_classes = 2  # Number of classes
n_channels = 1

# Load MNIST data
x_train, y_train, x_valid, y_valid = load_data(
    mode='train', times = t, prop = prop, bg = bg)
print("Size of:")
# print(y_train)
print("- Training-set:\t\t{}".format(len(y_train)))
print("- Validation-set:\t{}".format(len(y_valid)))

# Hyper-parameters
logs_path = "./logs"  # path to the folder that we want to save the logs for Tensorboard
lr = 0.001  # The optimization initial learning rate
epochs = 35  # Total number of training epochs
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

# conv1 = conv_layer(x, filter_size1, num_filters1, stride1, name='conv1')
# pool1 = max_pool(conv1, ksize=2, stride=2, name='pool1')
# conv2 = conv_layer(pool1, filter_size2, num_filters2, stride2, name='conv2')
# pool2 = max_pool(conv2, ksize=2, stride=2, name='pool2')

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

# # conv1-1 pool 1-3
# # output size 
# conv1_3 = conv_layer(x, filter_size1_3, num_filters1_3, stride1, name='conv1_3')
# # output size
# pool1_3 = max_pool(conv1_3, ksize=2, stride=1, name='pool1_3')

# # conv1-1 pool 1-1
# # output size
# conv1_4 = conv_layer(x, filter_size1_4, num_filters1_4, stride1, name='conv1_4')
# # output size
# pool1_4 = max_pool(conv1_4, ksize=2, stride=1, name='pool1_4')


pool1 = pool1_1#tf.concat([pool1_2, pool1_3, pool1_4], 1)

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

# Specify desired features
# f = [tf.shape(conv1_1),tf.shape(conv1_2), tf.shape(conv1_3), tf.shape(conv1_4), tf.shape(pool1_1),tf.shape(pool1_2), tf.shape(pool1_3), tf.shape(pool1_4), tf.shape(pool12), tf.shape(pool123), tf.shape(pool1), tf.shape(conv2), tf.shape(pool2), tf.shape(layer_flat), tf.shape(fc1), tf.shape(fc2), tf.shape(output_logits)]
f = [pool1]
desired_features = [correct_prediction, y, fc2]

# # Specify which ckpt to use
# ckpt_path = saver.save(sess, "{}/model.epoch{}.ckpt".format(logs_path, epochs-1))

# Switch, skip training or not
if trainFlag:
    # Launch the graph (session)
    with tf.Session() as sess:
        sess.run(init)
        global_step = 0
        summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
        # Number of training iterations in each epoch
        num_tr_iter = int(len(y_train) / batch_size)
        for epoch in range(epochs):
            print('Training epoch: {}'.format(epoch + 1))
            # x_train, y_train = randomize(x_train, y_train)
            for iteration in range(num_tr_iter):
                global_step += 1
                start = iteration * batch_size
                end = (iteration + 1) * batch_size
                x_batch, y_batch = get_next_batch(x_train, y_train, start, end)

                # Run optimization op (backprop)
                feed_dict_batch = {x: x_batch, y: y_batch}

                # strucf = sess.run([f], feed_dict=feed_dict_batch)
                # print(strucf)

                sess.run(optimizer, feed_dict=feed_dict_batch)

                # features_np = sess.run(desired_features)

                # Calculate and display the batch loss and accuracy
                loss_batch, acc_batch, summary_tr, features_np= sess.run([loss, accuracy, merged, desired_features],
                                                                 feed_dict=feed_dict_batch)
                summary_writer.add_summary(summary_tr, global_step)
                    # print( f"The flat layer is: {sess.run(layer_flat[0], feed_dict=feed_dict_batch)}" )
                    # print( f"The fc1 layer is: {sess.run(fc1[0], feed_dict=feed_dict_batch)}" )
                    # print( f"The fc2 layer is: {sess.run(fc2[0], feed_dict=feed_dict_batch)}" )

                    # print( f"The output_logits is: {sess.run(output_logits, feed_dict=feed_dict_batch)}" )
                    # print( f"The correct_prediction is: {sess.run(y, feed_dict=feed_dict_batch)}" )

                if iteration%display_freq == 0:
                    print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".#,\tR2={1:.2f}".
                              format(iteration, loss_batch, acc_batch))

                    #print(features_np[0].shape)
                    #print(x_batch)
                    #print(x_batch.shape)
                # print(features_np) 

                # if iteration % display_freq == 0:
                #     # Calculate and display the batch loss and accuracy
                #     loss_batch, acc_batch, summary_tr, features_np = sess.run([loss, accuracy, merged, desired_features],
                #                                                  feed_dict=feed_dict_batch)
                #     summary_writer.add_summary(summary_tr, global_step)

                #     # print( f"The flat layer is: {sess.run(layer_flat[0], feed_dict=feed_dict_batch)}" )
                #     # print( f"The fc1 layer is: {sess.run(fc1[0], feed_dict=feed_dict_batch)}" )
                #     # print( f"The fc2 layer is: {sess.run(fc2[0], feed_dict=feed_dict_batch)}" )

                #     # print( f"The output_logits is: {sess.run(output_logits, feed_dict=feed_dict_batch)}" )
                #     # print( f"The correct_prediction is: {sess.run(y, feed_dict=feed_dict_batch)}" )


                #     print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".#,\tR2={1:.2f}".
                #           format(iteration, loss_batch, acc_batch))

                #     #print(features_np[0].shape)
                #     #print(x_batch)
                #     #print(x_batch.shape)
                # print(features_np)
            # Run validation after every epoch
            feed_dict_valid = {x: x_valid, y: y_valid}
            loss_valid, acc_valid, summary_val = sess.run([loss, accuracy, merged], feed_dict=feed_dict_valid)
            summary_writer.add_summary(summary_val, global_step)
            print('---------------------------------------------------------')
            print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
                  format(epoch + 1, loss_valid, acc_valid))
            print('---------------------------------------------------------')

            # Save the variables to disk.
            save_path = saver.save(sess, "{}/{}.bg{}.model.epoch{}.ckpt".format(logs_path, prop, bg, epoch))
            print("Model saved in path: %s" % save_path)

            # Save the x
            filenamex = "x_epoch%d.csv" %(epoch)
            rows = x_batch.shape[0] # number of input images
            sideL = x_batch.shape[1] # .shape[1] == .shape[2]
            extreme = int((sideL - 1)/2) # for example, in 10*10 case it stands for where 1st pixel locates (4, 4)
            # open file and loop
            with open(filenamex, 'w', newline = '') as f:
                for row in range(rows):
                    matrix = x_batch[row] # for example, a 10*10*1 matrix
                    vector = [] # an empty vector, for example, a 1*15 vector
                    r = extreme # moving locator of row number
                    c = extreme # moving locator of col number
                    while r >= 0:
                        while c >= 0:
                            vector.append(matrix[r][c][0])
                            c -= 1
                        # end of the loop, finish a row
                        r -= 1
                        c = r
                    # end of the loop
                    # write the vector to file
                    for v in range(len(vector)):
                        f.write(str(vector[v]))
                        if v < (len(vector) - 1): # skip the last column
                            f.write(",")
                    f.write("\n")
            # Save the features_np
            filenamef = "features_np_epoch%d.csv" %(epoch)
            np.savetxt(filenamef, features_np[0], delimiter=",")

        sess.close()

    # reload session
    with tf.Session() as sess:
        saver.restore(sess, "{}/{}.bg{}.model.epoch{}.ckpt".format(logs_path, prop, bg, epochs - 1))
        # Test the network when training is done
        x_test, y_test = load_data(mode='test', times=t, prop = prop, bg = bg)
        feed_dict_test = {x: x_test, y: y_test}
        loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
        print('---------------------------------------------------------')
        print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
        print('---------------------------------------------------------')
    # sess.close()

# create the folder for the prop-bg pair if not exists
csv_output_path = "%s_%d" %(prop, bg)
if not os.path.isdir(csv_output_path):
    os.mkdir(csv_output_path)
# test
with tf.Session() as sess:
    saver.restore(sess, "{}/{}.bg{}.model.epoch{}.ckpt".format(logs_path, prop, bg, epochs-1))
    # Test the network when training is done
    x_wanted, y_wanted = load_data(mode='wanted', times=t, prop = prop, bg = bg)
    feed_dict_wanted = {x: x_wanted, y: y_wanted}
    loss_wanted, acc_wanted, features_np = sess.run([loss, accuracy, desired_features], feed_dict=feed_dict_wanted)
    print('---------------------------------------------------------')
    print("Wanted loss: {0:.2f}, Wanted accuracy: {1:.01%}".format(loss_wanted, acc_wanted))
    print('---------------------------------------------------------')
    print(features_np)

    # Save the x
    filenamex = "x_wanted.csv"
    rows = x_wanted.shape[0] # number of input images
    sideL = x_wanted.shape[1] # .shape[1] == .shape[2]
    extreme = int((sideL/t - 1)/2) # for example, in 10*10 case it stands for where 1st pixel locates (4, 4)
    # open file and loop
    with open(csv_output_path + '/' + filenamex, 'w', newline = '') as f:
        for row in range(rows):
            matrix = x_wanted[row] # for example, a 10*10*1 matrix
            vector = [] # an empty vector, for example, a 1*15 vector
            r = extreme # moving locator of row number
            c = extreme # moving locator of col number
            while r >= 0:
                while c >= 0:
                    vector.append(matrix[r][c][0])
                    c -= 1
                # end of the loop, finish a row
                r -= 1
                c = r
            # end of the loop
            # write the vector to file
            for v in range(len(vector)):
                f.write(str(vector[v]))
                if v < (len(vector) - 1): # skip the last column
                    f.write(",")
            f.write("\n")
    # Save the features_np
    filenamef1 = "%s/%s_%d_TP.csv" %(csv_output_path, prop, bg) # True Positive, open predicted as open
    filenamef2 = "%s/%s_%d_FN.csv" %(csv_output_path, prop, bg) # False Negative, open predicted as close
    filenamef3 = "%s/%s_%d_FP.csv" %(csv_output_path, prop, bg) # False Positive, close predicted as open
    filenamef4 = "%s/%s_%d_TN.csv" %(csv_output_path, prop, bg) # True Negative, close predicted as close
    filenamef5 = "%s/features_np_wanted_fc.csv" %(csv_output_path)
    # np.savetxt(filenamef, features_np[0], delimiter=",")
    print("=============================usefulINFO=============================")
    # file 1, bg open and prediction correct
    bgopen_rows = np.where(features_np[1][:,0] == 1)[0]
    predcorrect_rows = np.where(features_np[0] == 1)[0]
    bgopen_predopen_rows = np.intersect1d(bgopen_rows, predcorrect_rows)
    # file 2, bg open and predicted as close
    predwrong_rows = np.where(features_np[0] == 0)[0]
    bgopen_predclose_rows = np.intersect1d(bgopen_rows, predwrong_rows)
    # file 3, bg close and predicted as open
    bgclose_rows = np.where(features_np[1][:,0] == 0)[0]
    bgclose_predopen_rows = np.intersect1d(bgclose_rows, predwrong_rows)
    # file 4, bg close and predicted as close
    bgclose_predclose_rows = np.intersect1d(bgclose_rows, predcorrect_rows)
    # save files
    np.savetxt(filenamef1, np.transpose(bgopen_predopen_rows), fmt = '%i', delimiter=",")
    np.savetxt(filenamef2, np.transpose(bgopen_predclose_rows), fmt = '%i', delimiter=",")
    np.savetxt(filenamef3, np.transpose(bgclose_predopen_rows), fmt = '%i', delimiter=",")
    np.savetxt(filenamef4, np.transpose(bgclose_predclose_rows), fmt = '%i', delimiter=",")
    np.savetxt(filenamef5, features_np[2], delimiter=",")
sess.close()



#     # Plot some of the correct and misclassified examples
# cls_pred = sess.run(cls_prediction, feed_dict=feed_dict_test)

# cls_true = y_test#np.argmax(y_test, axis=1)
# plot_images(x_test, cls_true, cls_pred, title='Correct Examples')
# plot_example_errors(x_test, cls_true, cls_pred, title='Misclassified Examples')
# plt.show()