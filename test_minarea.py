import numpy
import cv2
import random

contour = numpy.array([[[0,20]], [[80,0]], [[100,100]], [[0,100]]]) #make a fake array

triangle = cv2.minEnclosingTriangle(contour)

print (triangle)

# drawing = numpy.zeros([100, 100],numpy.uint8)
# cv2.drawContours(drawing,[triangle],0,(255,255,255),2)

# cv2.imshow('output',drawing)
cv2.waitKey(0)
