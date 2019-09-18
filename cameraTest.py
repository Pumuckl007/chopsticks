import numpy as np
import cv2

cap = cv2.VideoCapture(0)

bgrBound = (65, 63, 102);
bgrBoundH = (132, 188, 255);

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


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);
    thresh = cv2.inRange(frame,bgrBound,bgrBoundH);
    h, s, v = cv2.split(hsv);
    b, g, r = cv2.split(frame);
    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY);
    # ret, thresh = cv2.threshold(s, bgrBound[0], 255, cv2.THRESH_BINARY_INV);
    # canny = cv2.Canny(s, bgrBound[0], bgrBound[1], bgrBound[2]);

    kernel = np.ones((5,5),np.uint8);
    erosion = cv2.erode(thresh,kernel,iterations = 1);
    blur = cv2.blur(erosion, (5,5));
    ret, hands = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY);

    im2, contours, hierarchy = cv2.findContours(hands, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, contours, -1, (0,255,0), 3)

    biggest = "";
    biggest2 = "";

    if (len(contours) >= 2):
        biggest = contours[0];
        biggest2 = contours[1];

    for contour in contours:
        area = cv2.contourArea(contour);
        if(area > 2000):
            cv2.drawContours(frame, [contour], -1, (0, 255, 0));
            x,y,w,h = cv2.boundingRect(contour);
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2);
            cv2.putText(frame,"Area: " + str(w*h), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2);
            if(area > cv2.contourArea(biggest)):
                biggest2 = biggest;
                biggest = contour;
            else :
                if(area > cv2.contourArea(biggest2)):
                    biggest2 = contour;
        else :
            cv2.drawContours(frame, [contour], -1, (0, 0, 255));

    if (len(contours) >= 2):
        cv2.drawContours(frame, [biggest, biggest2], -1, (255, 255, 0));

    # Display the resulting frame
    cv2.imshow('thresh',erosion);
    cv2.imshow('hands', hands);
    cv2.imshow('frame',frame);
    # cv2.imshow('origional',gray);
    cv2.imshow('hsv', hsv);

    # cv2.imshow('h', h);
    # cv2.imshow('s', s);
    # cv2.imshow('v', v);
    # cv2.imshow('b', b);
    # cv2.imshow('g', g);
    # cv2.imshow('r', r);
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
