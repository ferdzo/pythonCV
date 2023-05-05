import cv2
import numpy as np
from matplotlib import pyplot as plt



ig= cv2.imread("images/Lenna.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(ig, cv2.COLOR_BGR2GRAY)



plt.imshow(img, cmap = plt.cm.gray)
plt.xticks([]),plt.yticks([])
plt.show()

#Compass kernel matrices
n = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
ne = np.array([[0,1,1],[-1,0,1],[-1,-1,0]])
e = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
se = np.array([[-1,-1,0],[-1,0,1],[0,1,1]])
s = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
sw = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
w = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
nw = np.array([[1,1,0],[1,0,-1],[0,-1,-1]])

fig = plt.figure(figsize=(4, 4))
#North
n_img=cv2.filter2D(img,-1,n)
plt.imshow(n_img, cmap = plt.cm.gray)
plt.xticks([]), plt.yticks([])
plt.show()
#North East
ne_img=cv2.filter2D(img,-1,ne)
plt.imshow(ne_img, cmap = plt.cm.gray)
plt.xticks([]), plt.yticks([])
plt.show()
#South East
se_img=cv2.filter2D(img,-1,se,)
plt.imshow(se_img, cmap = plt.cm.gray)
plt.xticks([]), plt.yticks([])
plt.show()
#South
s_img=cv2.filter2D(img,-1,s)
plt.imshow(s_img, cmap = plt.cm.gray)
plt.xticks([]), plt.yticks([])
plt.show()
#South West
sw_img=cv2.filter2D(img,-1,sw)
plt.imshow(sw_img, cmap = plt.cm.gray)
plt.xticks([]), plt.yticks([])
plt.show()
#West
w_img=cv2.filter2D(img,-1,w)
plt.imshow(w_img, cmap = plt.cm.gray)
plt.xticks([]), plt.yticks([])
plt.show()
#North West
nw_img=cv2.filter2D(img,-1,nw)
plt.imshow(nw_img, cmap = plt.cm.gray)
plt.xticks([]), plt.yticks([])
plt.show()


#Merge filters
filters = [n_img,ne_img,e_img,se_img,s_img,sw_img,w_img,nw_img]
all = np.maximum.reduce(filters)
plt.imshow(all, cmap = plt.cm.gray)
plt.xticks([]),plt.yticks([])
plt.title('Convolved filters')
plt.show()