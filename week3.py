import numpy as np
from matplotlib import pyplot as plt
import cv2
img = cv2.imread('openCv-logo.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(f'number of coutours: {len(contours)}')

for i in range(len(contours)):
    cv2.drawContours(img, contours, i, (255, 0, 255-10*i), 3)

cv2.imshow('Image', img)
cv2.imshow('Image gray', imgray)

cv2.waitKey(0)
cv2.destroyAllWindows()
