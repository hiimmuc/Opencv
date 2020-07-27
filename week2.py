import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

# -> step 1
apple, orange, guy = cv2.imread("apple.jpg",), cv2.imread(
    "orange.jpg"), cv2.imread('guy.jpg')
# resize to 512x512 px for laplacian pyramid
apple, orange, guy = cv2.resize(apple, (512, 512)), cv2.resize(
    orange, (512, 512)), cv2.resize(guy, (512, 512))
apple, orange, guy = cv2.cvtColor(apple, cv2.COLOR_BGR2RGB), cv2.cvtColor(
    orange, cv2.COLOR_BGR2RGB), cv2.cvtColor(guy, cv2.COLOR_BGR2RGB)
mid = 256
OrApple = np.hstack((apple[:, :mid], orange[:, mid:]))

# -> step 2
# generate Gaussian pyramid for apple
apple_copy = apple.copy()
gp_apple = [apple_copy]
# find up to 6 level
for i in range(6):
    apple_copy = cv2.pyrDown(apple_copy)
    gp_apple.append(apple_copy)
# generate Gaussian pyramid for orange
orange_copy = orange.copy()
gp_orange = [orange_copy]
# find up to 6 level
for i in range(6):
    orange_copy = cv2.pyrDown(orange_copy)
    gp_orange.append(orange_copy)

# -> step 3
# generate laplacian pyramid for apple and orange
lp_apple = [gp_apple[5]]

for i in range(5, 0, -1):
    gaussian_extended = cv2.pyrUp(gp_apple[i])
    laplacian = cv2.subtract(gp_apple[i - 1], gaussian_extended)
    lp_apple.append(laplacian)
# #
lp_orange = [gp_orange[5]]

for i in range(5, 0, -1):
    gaussian_extended = cv2.pyrUp(gp_orange[i])
    laplacian = cv2.subtract(gp_orange[i - 1], gaussian_extended)
    lp_orange.append(laplacian)

# -> step 4
# join halves of both object together
apple_orange_pyr = []
for apple_lap, orange_lap in zip(lp_apple, lp_orange):
    rows, cols, channels = apple_lap.shape
    k = cols / apple.shape[1]
    diff = abs(mid - apple.shape[1]/2)
    laplacian = np.hstack(
        (apple_lap[:, 0: int(cols / 2 + k*diff)], orange_lap[:, int(cols / 2 + k*diff):]))
    apple_orange_pyr.append(laplacian)
#
# -> step 5
# reconstruct
apple_orange_reconstruct = apple_orange_pyr[0]
for i in range(1, 6):
    apple_orange_reconstruct = cv2.pyrUp(apple_orange_reconstruct)
    apple_orange_reconstruct = cv2.add(
        apple_orange_pyr[i], apple_orange_reconstruct)


#
titles = ['i have an apple', 'i have an orange', 'Ohh!!', 'OrApple !']
images = [apple, orange, guy, OrApple]
# show all images by matplotlib lib
plt.suptitle('Before blending', fontsize=16)
for i in range(len(images)):
    row, col = 1, len(images)
    if col >= row**2:
        step = int(math.sqrt(col) - row) + 1
        plt.subplot(row + step, int(math.sqrt(col)), i + 1)
    else:
        plt.subplot(row, col, i + 1)

    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

# show
apple_orange_reconstruct = cv2.cvtColor(
    apple_orange_reconstruct, cv2.COLOR_RGB2BGR)
cv2.imshow('After blending', apple_orange_reconstruct)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
