import numpy as np
import cv2

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

i = 3000
written = False

def save_image(img, n):
    global i
    global written
    if written:
        return
    cv2.imwrite('/media/pics/%01d/%04d.png' % (n, i),img)
    i = i + 1;
    written = True

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
                    written = False
                    hand = maybe_hand
            old_hist[2] = old_hist[1]
            old_hist[1] = old_hist[0]
            old_hist[0] = hist

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    cv2.imshow('frame', frame)
    cv2.imshow('hsv', hsvmask)
    cv2.imshow('dialated', mask)
    cv2.imshow('hand', hand)
    key = cv2.waitKey(1);
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('1'):
        save_image(hand, 1)
    if key & 0xFF == ord('2'):
        save_image(hand, 2)
    if key & 0xFF == ord('3'):
        save_image(hand, 3)
    if key & 0xFF == ord('4'):
        save_image(hand, 4)
    if key & 0xFF == ord('5'):
        save_image(hand, 5)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
