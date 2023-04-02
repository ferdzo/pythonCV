import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)

_5bit = np.divide(img, 8).astype('uint8')
plt.imshow(_5bit, cmap='gray')
plt.title("5 bit")
plt.show()

_4bit = np.divide(img, 16).astype('uint8')
plt.imshow(_4bit, cmap='gray')
plt.title("4 bit")
plt.show()

_3bit = np.divide(img, 32).astype('uint8')
plt.imshow(_3bit, cmap='gray')
plt.title("3 bit")
plt.show()

_2bit = np.divide(img, 64).astype('uint8')
plt.imshow(_2bit, cmap='gray')
plt.title("2 bit")
plt.show()

_1bit = np.divide(img, 128).astype('uint8')
plt.imshow(_1bit, cmap='gray')
plt.title("1 bit")
plt.show()
