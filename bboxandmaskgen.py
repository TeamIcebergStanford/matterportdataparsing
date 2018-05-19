##code developed by shiva badruswamy to generate 2-D bounding boxes on 3-D matterport data to validate if matterport data can be utilized in mask r-cnn tasks
##Sample code inputs matterport RGB images and outputs contour data. Contour data can be used to construct bounding boxes and masks.
##if at all we use matterport data for mask r-cnn analysis this code needs to be executed to generate 2-D bounding boxes.
## matterport data is compute intensive so the team will be using NYU data to do initial demonstrations

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# reduce decimal place to 1 while printing
np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

# image directory path
dirpath = "/Users/admin/Desktop/EducationandProjects/MatlabWorkspace/OpenCVcodetesting/undistorted_color_images/"
img_name = "0f37bd0737e349de9d536263a4bdd60d_i1_3.jpg"
img = mpimg.imread(dirpath+img_name)

# display image
cv.imshow("Original image:"+img_name, img.astype(np.uint8))
cv.waitKey()
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
for c in contours:
    rect = cv.boundingRect(c)
    area = cv.contourArea(c)
    x, y, w, h = rect
    if  area > 20 :
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
    center, w_h, aor = rect
    area = w_h[0] * w_h[1]
    if area > 20 and aor in range (0,360,90):
        print("center:", center, "width:", w_h[0], "height:", w_h[1], " aor:", aor)
        box = np.int0(cv.boxPoints(rect))
        imbox = cv.drawContours(img,[box],0,color,2)
cv.imshow("Ground Truth Close-fit Bounding Boxes", imbox)
cv.waitKey()
cv.destroyAllWindows()

