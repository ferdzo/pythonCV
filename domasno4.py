import cv2
import pprint
import glob
from matplotlib import pyplot as plt

class MyImage:
    def __init__(self, img_name):
        self.img = cv2.imread(img_name,0)
        self.__name = img_name
 
    def __str__(self):
        return self.__name

db_images = [MyImage(file) for file in glob.glob('database/*.jpg') ]
query_images = [MyImage(file) for file in glob.glob('query_images/*.jpg') ]
matched = dict()

for img in query_images:
    ret, thresh = cv2.threshold(img.img, 100, 255, cv2.THRESH_BINARY_INV)
    contours,hierarchy = cv2.findContours(thresh,2,1)
    cnt = contours[0]
    print("Image: ", str(img))
    for img in db_images:
        ret, thresh = cv2.threshold(img.img, 100, 255, cv2.THRESH_BINARY_INV)
        contours,hierarchy = cv2.findContours(thresh,2,1)
        ret = cv2.matchShapes(cnt,contours[0],1,0.0)
        matched[str(img)]=ret
        sorted(matched.items(),key=lambda x: x[1])
    pprint.pprint(matched)
    
    matched.clear()
