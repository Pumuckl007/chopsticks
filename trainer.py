import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import os
import random
import math
print ("PACKAGES LOADED")



training = []
testing = []

for i in range(1, 6):
    files = os.listdir("/media/pics/%01d/" % i)
    files = sorted(files)
    print (i)
    for file in files:
        img = cv2.imread(("/media/pics/%01d/" % i) + file, cv2.IMREAD_GRAYSCALE)
        # img = cv2.inRange(img, 40, 255)
        np_image_data = np.asarray(img)
        np_image_data=cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        np_final = np.concatenate(np_image_data,axis=0)
        hot = ord(file[0].upper())-65
        label = [0,0,0,0,0]
        label[i-1] = 1
        if random.randint(0, 100) < 1:
            testing.append({'img': np_final, 'label': label})
        else:
            training.append({'img': np_final, 'label': label})

random.shuffle(training)

print ("Training length", len(training))

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 5])  # None is for infinite
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 5])
b_fc2 = bias_variable([5])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# W = tf.Variable(tf.zeros([100, 26]))
# b = tf.Variable(tf.zeros([26]))

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

batch_size      = 30
training_epochs = int(math.floor(len(training)/batch_size))
display_step    = 5
# SESSION
sess = tf.InteractiveSession()
sess.run(init)

def getBatch(start, num):
    data = training[start:(start+num)]
    return data

def getXsAndYs(batch):
    xs = []
    ys = []
    for data in batch:
        xs.append(data['img'])
        ys.append(data['label'])
    return xs, ys

current_pos = 0
# MINI-BATCH LEARNING
for epoch in range(training_epochs):
    batch = getBatch(current_pos, batch_size)
    batch_xs, batch_ys = getXsAndYs(batch)
    current_pos += batch_size
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.6})
    # DISPLAY
    if epoch % display_step == 0:
        feeds_train = {x: batch_xs, y_: batch_ys, keep_prob: 1.0}
        test_xs, test_ys = getXsAndYs(testing)
        feeds_test = {x: test_xs, y_: test_ys, keep_prob: 1.0}
        train_acc = accuracy.eval(feed_dict=feeds_train)
        test_acc = accuracy.eval(feed_dict=feeds_test)
        print ("Epoch: %03d/%03d train_acc: %.3f test_acc: %.3f"
               % (epoch, training_epochs, train_acc, test_acc))
print ("DONE")

saver = tf.train.Saver()
saver.save(sess, "/media/pics/model.ckpt")
