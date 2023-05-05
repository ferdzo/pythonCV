import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
img = cv2.imread("images/Lenna.jpg", cv2.IMREAD_GRAYSCALE)

# Define the compass kernel matrices
n = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
ne = np.array([[0,1,1],[-1,0,1],[-1,-1,0]])
e = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
se = np.array([[-1,-1,0],[-1,0,1],[0,1,1]])
s = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
sw = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
w = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
nw = np.array([[1,1,0],[1,0,-1],[0,-1,-1]])

# Apply the transformation
n_img = cv2.filter2D(img, -1, n)
ne_img = cv2.filter2D(img, -1, ne)
e_img = cv2.filter2D(img, -1, e)
se_img = cv2.filter2D(img, -1, se)
s_img = cv2.filter2D(img, -1, s)
sw_img = cv2.filter2D(img, -1, sw)
w_img = cv2.filter2D(img, -1, w)
nw_img = cv2.filter2D(img, -1, nw)

# Initial image
plt.imshow(img, cmap = plt.cm.gray)
plt.xticks([]),plt.yticks([])
plt.title('Initial image')
plt.show()

# Create a subplot with two rows and four columns
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))


# Plot the images in the subplots
axs[0, 0].imshow(n_img, cmap=plt.cm.gray)
axs[0, 0].set_title('North')
axs[0, 1].imshow(ne_img, cmap=plt.cm.gray)
axs[0, 1].set_title('North East')
axs[0, 2].imshow(e_img, cmap=plt.cm.gray)
axs[0, 2].set_title('East')
axs[0, 3].imshow(se_img, cmap=plt.cm.gray)
axs[0, 3].set_title('South East')
axs[1, 0].imshow(s_img, cmap=plt.cm.gray)
axs[1, 0].set_title('South')
axs[1, 1].imshow(sw_img, cmap=plt.cm.gray)
axs[1, 1].set_title('South West')
axs[1, 2].imshow(w_img, cmap=plt.cm.gray)
axs[1, 2].set_title('West')
axs[1, 3].imshow(nw_img, cmap=plt.cm.gray)
axs[1, 3].set_title('North West')

# Remove the x and y axis ticks
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

# Set the title of the subplot
fig.suptitle('Compass Filter Results')

# Display the plot
plt.show()

#Merge filters
filters = [n_img,ne_img,e_img,se_img,s_img,sw_img,w_img,nw_img]
all = np.maximum.reduce(filters)
plt.imshow(all, cmap = plt.cm.gray)
plt.xticks([]),plt.yticks([])
plt.title('Convolved filters')
plt.show()