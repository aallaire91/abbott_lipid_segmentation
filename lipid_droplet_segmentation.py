import cv2
import numpy as np

img_name = "data/test_images/MAX_5_1.tif"
img_gt = cv2.imread(img_name[:-4]+"_gt.tif")
img = cv2.imread(img_name)

cells = img[:,:,-1]
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# gblur = cv2.equalizeHist(cells)
gblur = cv2.GaussianBlur(cells,(11,11),0)

v = np.median(gblur)
# apply automatic Canny edge detection using the computed median
lower = int(max(0, (1.0 - 0.66) * v))
upper = int(min(255, (1.0 + 0.66) * v))
edges = cv2.Canny(gblur, lower, upper)
edges = cv2.dilate(edges,kernel5,iterations = 1)
edges = cv2.erode(edges,kernel5,iterations = 1)

# cv2.imwrite(f"gblur_{img_name}",gblur)


ret, thresh = cv2.threshold(cells,40,255,cv2.THRESH_BINARY)
erode = cv2.erode(thresh,kernel5,iterations = 1)
dilate = cv2.dilate(erode,kernel5,iterations = 1)

gthresh = cv2.adaptiveThreshold(cells,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,25,2)
gthresh = cv2.erode(gthresh,kernel5,iterations = 1)
gthresh = cv2.dilate(gthresh,kernel5,iterations = 1)





and_img = cv2.bitwise_and(dilate,gthresh)
# and_img = cv2.bitwise_or(and_img,edges)


im_contour = img.copy()
im_contour[:,:,-1] = cells.copy()
im_contour[:,:,0] = 0

im_contour_all = img.copy()
im_contour_all[:,:,-1] = cells.copy()
im_contour_all[:,:,0] = 0

lipid_contours = []
areas = []
mean_vals = []
# ret,contours,hierarchy = cv2.findContours(edges,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
# mask = np.zeros(cells.shape,np.uint8)

# hierarchy = hierarchy[0]
# for contour,descr in zip(contours,hierarchy):
#     print(descr)
#     # if not ((descr[-1]<0) and (descr[2]<0) ):
#     if (descr[-1] < 0):
#         if cv2.contourArea(contour) < 450 and cv2.contourArea(contour) > 15:
#             pos, size,th = cv2.minAreaRect(contour)
#             aspect_ratio = float(size[0]) / size[1]
#             if aspect_ratio > 0.3 and aspect_ratio < 3:
#                 mask = np.zeros(cells.shape, np.uint8)
#
#                 cv2.drawContours(mask, [contour], 0, 255, -1)
#                 mean_val = cv2.mean(cells,mask = mask)[0]
#                 # if mean_val > 40:
#
#                     # areas.append(cv2.contourArea(contour))
#                     # lipid_contours.append(contour)
ret,contours,hierarchy = cv2.findContours(and_img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

hierarchy = hierarchy[0]

for contour,descr in zip(contours,hierarchy):
    cv2.drawContours(im_contour_all,contour, -1, (0, 255, 0), 1)
    if (descr[-1]<0):
        if cv2.contourArea(contour) < 450 and cv2.contourArea(contour) > 15:
            pos, size,th = cv2.minAreaRect(contour)
            aspect_ratio = float(size[0]) / size[1]
            if aspect_ratio > 0.3 and aspect_ratio < 3:
                mask = np.zeros(cells.shape, np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                mean_val = cv2.mean(cells, mask=mask)[0]
                if mean_val > 40:

                    areas.append(cv2.contourArea(contour))
                    lipid_contours.append(contour)

for contour in lipid_contours:
    cv2.drawContours(im_contour, contour, -1, (0, 255, 0), 1)


# markers = cv2.watershed(img,markers)
# img[markers == -1] = [255,0,0]

save_img = np.concatenate((img_gt,im_contour),axis=1)
output_prefix = img_name.split("\")[:-4]
cv2.imwrite(f"{output_prefix}_lipid_droplets.tif",save_img)
cv2.imwrite(f"{output_prefix}_contours_all.tif",im_contour_all)
cv2.imshow("input",cells)
# cv2.imshow("edges",edges)
# cv2.imshow("gblur",gblur)
cv2.imshow("thresh",thresh)
cv2.imshow("adaptive filter",gthresh)
cv2.imshow("and",and_img)
cv2.imshow("lipids",save_img)
cv2.imshow("contours_all",im_contour_all)
cv2.waitKey(0)
