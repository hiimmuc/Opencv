# learning ab openCV applying to data analysis
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1 capture videos
# cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
# out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret == True:
#         print(f'{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} x {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}')
#         out.write(frame)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         cv2.imshow('frame', gray)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()


# 2 draw shape
# img = cv2.imread('husky.jpg', 1)
# # img = np.zeros([512, 512, 3], np.uint8)
# color = (147, 96, 40)
# # methods
# img = cv2.line(img, (0, 0), (305, 305), color, 5)
# img = cv2.line(img, (610, 0), (305, 305), color, 5)
# img = cv2.arrowedLine(img, (305, 0), (305, 305), color, 5)
# img = cv2.rectangle(img, (100, 100), (510, 510), color, 3)
# img = cv2.circle(img, (305, 305), 150, color, 3)
# for i in range(4):
#     img = cv2.ellipse(img, (305, 305), (200, 80), 0 + 45*i, 0, 360, color, 5)
# font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
# img = cv2.putText(img, 'HUSKY', (100, 700), font, 4, (127.5, 127.5, 127.5), 10, cv2.LINE_AA)
# #
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 3 setting camera paraemeters
# import datetime
# cap = cv2.VideoCapture(0)
# # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
# # out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))
# # cap.set(3, 640)
# # cap.set(4, 480)
# # print(f'{cap.get(3)} x {cap.get(4)}')
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret == True:
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         text = 'Width: ' + str(cap.get(3)) + '  Height: ' + str(cap.get(4))
#         date = str(datetime.datetime.now())
#         frame = cv2.putText(frame, date, (10, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
#
# cap.release()
# # out.release()
# cv2.destroyAllWindows()

# 4
# handle mouse events
# events = [i for i in dir(cv2) if 'EVENT' in i]
# print(events)
#
#
# def click_event(event, x, y, flag, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
#         # points.append((x, y))
#         # if len(points) >= 2:
#         #     cv2.line(img, points[-1], points[-2], (0, 0, 255), 2)
#         # blue = img[x, y, 0]
#         # green = img[x, y, 1]
#         # red = img[x, y, 2]
#         # cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
#         # my_color = np.zeros((512, 512, 3), np.uint8)
#         # my_color[:] = [blue, green, red]
#         #
#         print(x, ' , ', y)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         strXY = f'{x} x {y}'
#         cv2.putText(img, strXY, (x, y), font, .5, (255, 255, 0), 1)
#
#         cv2.imshow('image', img)
#
#
# # img = np.zeros((512, 512, 3), np.uint8)
# img = cv2.imread('ronaldo1.jpg')
# cv2.imshow('image', img)
# points = []
# cv2.setMouseCallback('image', click_event)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 5
# img = cv2.imread('ronaldo1.jpg')
# img2 = cv2.imread('ronaldo.jpg')
# print(img.shape)
# print(img.size)
# print(img.dtype)
# b, g, r = cv2.split(img)
# img = cv2.merge((b, g, r))
#
# ball = img[215:280, 365:425]
# img[273:338, 300:360] = ball
#
# img2 = cv2.resize(img2, (600, 445))
# dst = cv2.add(img, img2)
# dst = cv2.addWeighted(img, .8, img2, .2, 0)
#
# cv2.imshow('Image', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 6
# img1 = np.zeros((600, 445, 3), np.uint8)
# img1 = cv2.rectangle(img1, (200, 0), (300, 100), (255, 255, 255), -1)
# img2 = cv2.imread("ronaldo.jpg")
#
# bitAnd = cv2.bitwise_and(img2, img1)
# # bitOr = cv2.bitwise_or(img2, img1)
# # bitXor = cv2.bitwise_xor(img1, img2)
# # bitNot1 = cv2.bitwise_not(img1)
# # bitNot2 = cv2.bitwise_not(img2)
# #
# cv2.imshow("img1", img1)
# cv2.imshow("img2", img2)
# # cv2.imshow('bitAnd', bitAnd)
# # cv2.imshow('bitOr', bitOr)
# # cv2.imshow('bitXor', bitXor)
# # cv2.imshow('bitNot1', bitNot1)
# # cv2.imshow('bitNot2', bitNot2)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 7
# import cv2 as cv
#
#
# def nothing(x):
#     print(x)
#
#
# # Create a black image, a window
# # img = np.zeros((300, 512, 3), np.uint8)
# # cv.namedWindow('image')
# #
# # cv.createTrackbar('B', 'image', 0, 255, nothing)
# # cv.createTrackbar('G', 'image', 0, 255, nothing)
# # cv.createTrackbar('R', 'image', 0, 255, nothing)
# #
# # switch = '0 : OFF\n 1 : ON'
# # cv.createTrackbar(str(switch), 'image', 0, 1, nothing)
# # running = True
# # while (running):
# #     cv.imshow('image', img)
# #
# #     b = cv.getTrackbarPos('B', 'image')
# #     g = cv.getTrackbarPos('G', 'image')
# #     r = cv.getTrackbarPos('R', 'image')
# #     s = cv.getTrackbarPos(switch, 'image')
# #
# #     if s == 0:
# #         img[:] = 0
# #     else:
# #         img[:] = [b, g, r]
# #     k = cv.waitKey(1) & 0xFF
# #     if k == 27:
# #         running = False
# #
# # cv.destroyAllWindows()
#
# def nothing(x):
#     print(x)
#
#
# # Create a black image, a window
# cv.namedWindow('image')
#
# cv.createTrackbar('CP', 'image', 10, 400, nothing)
#
# switch = 'color/gray'
# cv.createTrackbar(switch, 'image', 0, 1, nothing)
#
# while (1):
#     img = cv.imread('husky.jpg')
#     pos = cv.getTrackbarPos('CP', 'image')
#     font = cv.FONT_HERSHEY_SIMPLEX
#     cv.putText(img, str(pos), (50, 150), font, 6, (0, 0, 255), 10)
#
#     k = cv.waitKey(1) & 0xFF
#     if k == 27:
#         break
#
#     s = cv.getTrackbarPos(switch, 'image')
#
#     if s == 0:
#         pass
#     else:
#         img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
#     img = cv.imshow('image', img)
#
# cv.destroyAllWindows()
# 8
# def nothing(x):
#     pass
#
#
# cap = cv2.VideoCapture(0)
#
# cv2.namedWindow("Tracking")
# cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
# cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
# cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
#
# while True:
#     # frame = cv2.imread('smarties.png')
#     _, frame = cap.read()
#
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     l_h = cv2.getTrackbarPos("LH", "Tracking")
#     l_s = cv2.getTrackbarPos("LS", "Tracking")
#     l_v = cv2.getTrackbarPos("LV", "Tracking")
#
#     u_h = cv2.getTrackbarPos("UH", "Tracking")
#     u_s = cv2.getTrackbarPos("US", "Tracking")
#     u_v = cv2.getTrackbarPos("UV", "Tracking")
#
#     l_b = np.array([l_h, l_s, l_v])
#     u_b = np.array([u_h, u_s, u_v])
#
#     mask = cv2.inRange(hsv, l_b, u_b)
#
#     res = cv2.bitwise_and(frame, frame, mask=mask)
#
#     cv2.imshow("frame", frame)
#     cv2.imshow("mask", mask)
#     cv2.imshow("res", res)
#
#     key = cv2.waitKey(1)
#     if key == 27:
#         breakS
#
# cap.release()
# cv2.destroyAllWindows()

# img = cv2.imread('gradients.jpeg', 0)
# _, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# _, th2 = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
# _, th3 = cv2.threshold(img, 128, 255, cv2.THRESH_TRUNC)
# _, th4 = cv2.threshold(img, 128, 255, cv2.THRESH_TOZERO)
#
# cv2.imshow('THRESHOLD1', th1)
# cv2.imshow('THRESHOLD2', th2)
# cv2.imshow('THRESHOLD3', th3)
# cv2.imshow('THRESHOLD4', th4)
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread('sudoku.jpg', 0)
# # _, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# # th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# # th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# #
# # cv2.imshow('Image', img)
# # cv2.imshow('THRESHOLD1', th1)
# # cv2.imshow('THRESHOLD2', th2)
# # cv2.imshow('THRESHOLD3', th3)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# matplotlib

# img = cv2.imread('lily.jpg', -1)
# cv2.imshow('image', img)
#
# img = cv2.imread('lily.jpg', cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# img = cv.imread('gradients.jpeg', 0)
# _, th1 = cv.threshold(img, 50, 255, cv.THRESH_BINARY)
# _, th2 = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)
# _, th3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
# _, th4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
# _, th5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
#
# titles = ['Original Image', 'BINARY',
#           'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
# images = [img, th1, th2, th3, th4, th5]
#
# for i in range(6):
#     plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# #cv.imshow("Image", img)
# #cv.imshow("th1", th1)
# #cv.imshow("th2", th2)
# #cv.imshow("th3", th3)
# #cv.imshow("th4", th4)
# #cv.imshow("th5", th5)
# plt.show()
# # cv.waitKey(0)
# # cv.destroyAllWindows()

img = cv2.imread('ronaldo.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
titles = ['image']
images = [img]
for i in range(len(images)):
    plt.subplot(1, 1, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()


# img = cv2.imread('lily.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# edge detection
# lap = cv2.Laplacian(img, cv2.CV_64F, ksize=1)
# lap = np.uint8(np.absolute(lap))
# # dx, dy can be 1 or 0
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
# sobelx = np.uint8(np.absolute(sobelx))
# sobely = np.uint8(np.absolute(sobely))
# sobecombine = cv2.bitwise_or(sobelx, sobely)
# canny = cv2.Canny(img, 100, 200)

# # list of images and their titles
# title = ['image', 'Laplacian', 'sobelx', 'sobely', 'sobecombine', 'canny']
# images = [img, lap, sobelx, sobely, sobecombine, canny]


# show all images by matplotlib lib
# for i in range(len(images)):
#     row, col = 1, len(images)
#     if col >= row**2:
#         step = int(math.sqrt(col) - row) + 1
#         plt.subplot(row + step, int(math.sqrt(col)), i + 1)
#     else:
#         plt.subplot(row, col, i + 1)

#     plt.imshow(images[i], 'gray')
#     plt.title(title[i])
#     plt.xticks([])
#     plt.yticks([])

# plt.show()
