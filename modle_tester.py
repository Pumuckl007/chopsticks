import numpy as np
import cv2
import tensorflow as tf

cap = cv2.VideoCapture(0)

bgrBound = (0, 0, 83);
bgrBoundH = (195, 87, 255);

def callbackB(newVal) :
    global bgrBound;
    bgrBound = (newVal, bgrBound[1], bgrBound[2]);

def callbackG(newVal) :
    global bgrBound;
    bgrBound = (bgrBound[0], newVal, bgrBound[2]);

def callbackR(newVal) :
    global bgrBound;
    bgrBound = (bgrBound[0], bgrBound[1], newVal);

def callbackBH(newVal) :
    global bgrBoundH;
    bgrBoundH = (newVal, bgrBoundH[1], bgrBoundH[2]);

def callbackGH(newVal) :
    global bgrBoundH;
    bgrBoundH = (bgrBoundH[0], newVal, bgrBoundH[2]);

def callbackRH(newVal) :
    global bgrBoundH;
    bgrBoundH = (bgrBoundH[0], bgrBoundH[1], newVal);

cv2.namedWindow('Color Selector')
cv2.createTrackbar('B', 'Color Selector', bgrBound[0], 255, callbackB)
cv2.createTrackbar('G', 'Color Selector', bgrBound[1], 255, callbackG)
cv2.createTrackbar('R', 'Color Selector', bgrBound[2], 255, callbackR)
cv2.createTrackbar('BH', 'Color Selector', bgrBoundH[0], 255, callbackBH)
cv2.createTrackbar('GH', 'Color Selector', bgrBoundH[1], 255, callbackGH)
cv2.createTrackbar('RH', 'Color Selector', bgrBoundH[2], 255, callbackRH)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(1, 0.7)
kernel_hand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
fgbg_hand = cv2.bgsegm.createBackgroundSubtractorGMG(1, 1)
kernel_erode = np.ones((5,5),np.uint8)

hand = ""
old_hist = ["", "", ""]

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

tfx = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 5])  # None is for infinite
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(tfx, [-1, 28, 28, 1])
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

decoder = tf.argmax(y_conv, axis=1)

saver = tf.train.Saver()

sess = tf.Session()
saver.restore(sess, "/media/pics/model.ckpt")

def identify(img):
    np_image_data = np.asarray(img)
    np_image_data=cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
    np_final = np.concatenate(np_image_data,axis=0)
    data = {tfx: [np_final], keep_prob: 1.0}
    decodedResult = sess.run(decoder, data)
    print (decodedResult)

while(True):
    ret, frame = cap.read()

    if(hand == ""):
        size = 28, 28, 1
        hand = np.zeros(size, dtype=np.uint)

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    hsvmask = cv2.inRange(hsv, bgrBound, bgrBoundH)

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    eroded = cv2.erode(fgmask, kernel_erode, iterations = 1)
    dilated = cv2.dilate(eroded, kernel_erode, iterations = 3)

    mask = cv2.bitwise_and(hsvmask, dilated)

    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    biggest = ""
    biggest_area = 0

    if( len(contours) > 0):
        biggest = contours[0]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x,y,w,h = cv2.boundingRect(contour);
            if(w * h > biggest_area):
                biggest = contour;
                biggest_area = w*h

    if not biggest == "":
        x,y,w,h = cv2.boundingRect(biggest);
        cx = x + w/2
        # cy = y + h/2
        h = 200
        w = 200
        x = int(cx - w/2)
        # y = int(cy - h/2)
        if(w > 0 and h > 0 and x > 0):
            maybe_hand = cv2.resize(frame[y:y+h, x:x+w],(28, 28), interpolation = cv2.INTER_CUBIC)
            maybe_hand_mask = cv2.resize(mask[y:y+h, x:x+w],(28, 28), interpolation = cv2.INTER_CUBIC)
            maybe_hand = cv2.cvtColor(maybe_hand, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([maybe_hand_mask],[0],None,[256],[0,256])
            if x < 200:
                maybe_hand = cv2.flip(maybe_hand, 1)
            if( not old_hist[2] == ""):
                correlation1 = cv2.compareHist(hist, old_hist[0], cv2.HISTCMP_CORREL)
                correlation2 = cv2.compareHist(hist, old_hist[1], cv2.HISTCMP_CORREL)
                correlation3 = cv2.compareHist(hist, old_hist[2], cv2.HISTCMP_CORREL)
                correlation = max(correlation1, correlation2, correlation3)
                if(correlation > 0.99985):
                    identify(maybe_hand)
                    hand = maybe_hand
            old_hist[2] = old_hist[1]
            old_hist[1] = old_hist[0]
            old_hist[0] = hist

    cv2.imshow('frame', frame)
    cv2.imshow('hsv', hsvmask)
    cv2.imshow('hand', hand)
    key = cv2.waitKey(1);
    if key & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
