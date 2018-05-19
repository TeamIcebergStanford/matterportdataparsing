##Code largely based on openCV developed by Shiva Badruswamy to generate 2-D bounding boxes on 3-D matterport data to validate if matterport data can be input shaped for FAIR's mask r-cnn algorithm
##Sample code inputs matterport RGB images and outputs contour data. Contour data can be used to construct bounding boxees, masks, polygons etc.
## with these generated masks and contours the mask r-cnns utils can be used to generate bounding boxes as well.
##I have generated 2 types of bounding boxes - a stadard one and a close fit one. The close fit BBox can be made to fit objects at any angles. 
##Contours are extracted using Canny algorithm


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# reduce decimal place to 1 while printing
np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

# image directory path
dirpath = "/path/" ##set your image directory's path here
img_name = "imgnamejpg" ##put your image name in here
img = mpimg.imread(dirpath+img_name)

# display image
cv.imshow("Original image:"+img_name, img.astype(np.uint8))
cv.waitKey() ##when cursor on image, press any key to progress in code
cv.destroyAllWindows()

# extract edges
imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imggrayblur = cv.blur(imggray, (3, 3))
imedges = cv.Canny(imggrayblur, 50, 75, apertureSize=3, L2gradient=True)
cv.imshow("Extracted edges", imedges)
cv.waitKey()
cv.destroyAllWindows()

# # applying closing function
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
# closed = cv.morphologyEx(imedges, cv.MORPH_CLOSE, kernel)
# cv.imshow("Closed", closed)
# cv.waitKey(0)
# cv.destroyAllWindows()

# find contours
img2,contours,hierarchy = cv.findContours(
    imedges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
print("contours array shape = ", np.shape(contours))
cv.imshow("Extracted contours", img2)
cv.waitKey()
cv.destroyAllWindows()

# draw contours
# for c in contours:
#     cnt_img = cv.drawContours(img, [c], -1, (0, 255, 0), 2)
cnt_img = cv.drawContours(img, contours, -1, (0, 255, 0), 2)
cv.imshow("Contours applied on original image:"+img_name, cnt_img)
cv.waitKey()
cv.destroyAllWindows()

# # draw masks
# color = (255,255,255)
# for c in contours:
#     cnt_img2 = cv.drawContours(img, c, -1, color, cv.FILLED)
#     cv.imshow("Ground Truth Masks", cnt_img2)
# cv.waitKey()
# cv.destroyAllWindows()

# draw bounding boxes
start = 200
end = 500
step = 1
color = (255,0,0)
area_thresh = 20
for c in contours:
    rect = cv.boundingRect(c)
    area = cv.contourArea(c) ## area under bounding box. useful to screen out bounding boxes that are less than certain sizes
    x, y, w, h = rect ##bounding box params. 
    if  area > area_thresh :
        print("x:",x, " y:", y, " w:",w, " h:",h, " Area:", area)
        cv.rectangle(img, (x, y), (x+w, y+h),color, 2)
        cv.putText(img, 'Area:' + str(area),
                   (x+w+10, y+h), 0, 0.3, color)
cv.imshow("Ground Truth Bounding Boxes", img)
cv.waitKey()
cv.destroyAllWindows()

# Min area rectangle
for c in contours:
    rect = cv.minAreaRect(c)
    center, w_h, aor = rect ##aor = angle of rotation, w_h = (width,height) tuple, center = (x,y) tuple of bounding box center
    area = w_h[0] * w_h[1]
    if area > 20 and aor in range (0,360,90):
        print("center:", center, "width:", w_h[0], "height:", w_h[1], " aor:", aor)
        box = np.int0(cv.boxPoints(rect))
        imbox = cv.drawContours(img,[box],0,color,2)
cv.imshow("Ground Truth Close-fit Bounding Boxes", imbox)
cv.waitKey()
cv.destroyAllWindows()

