import cv2
import numpy as np
import Tracker_HSV as tr
from PIL import Image
import os

cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
cv2.resizeWindow('mask',(500,500))
filename= 'flower.jpg'
cwd = os.getcwd()
name_file=os.path.splitext(filename)[0]

path_save_temp=os.path.join(cwd,'Data')
path_save_folder=os.path.join(path_save_temp,f'{name_file}_blur_data')
if not os.path.exists(path_save_folder):
    os.makedirs(path_save_folder)


img=cv2.imread(filename)
img = cv2.GaussianBlur(img,(5,5),0)
img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

file_save_blur= os.path.join(path_save_folder,'blur.png')
im_blur = cv2.GaussianBlur(img,(81,81),0)
cv2.imwrite(file_save_blur,im_blur)

xs,ys,w,h = cv2.selectROI('mask',img)
crop_img=crop_img_true=crop_img_contour=img[ys:ys+h, xs:xs+w]

if not crop_img_true.shape[0]> 1:
    crop_img_true=img

x,y,z,a,b,c=(tr.tracker(crop_img_true))

crop_img_true=cv2.cvtColor(crop_img_true,cv2.COLOR_BGR2HSV)

file_save_mask_inrange= os.path.join(path_save_folder,'mask inRange.png')
mask_inRange=cv2.inRange(crop_img_true,(x,y,z),(a,b,c))
cv2.imwrite(file_save_mask_inrange,mask_inRange)


_, threshold = cv2.threshold(mask_inRange, 250, 255, cv2.THRESH_BINARY)
Gauss_threshold =cv2.adaptiveThreshold(threshold,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,101,10)


blank_space_black= np.zeros((crop_img_true.shape[0],crop_img_true.shape[1]),np.uint8)
blank_space_black[:]=(0)

_,contours,_ = cv2.findContours(Gauss_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


maxi=cv2.contourArea(contours[0])
c=[]

for cnt in contours:
    if cv2.contourArea(cnt)>=maxi:
        maxi=cv2.contourArea(cnt)
##        print(cv2.contourArea(cnt))
        c= cnt

file_save_contour= os.path.join(path_save_folder,'Contour.png')
cv2.drawContours(crop_img_contour, c, -1, (0, 255, 0), 5)
cv2.imwrite(file_save_contour,crop_img_contour)


file_save_poly= os.path.join(path_save_folder,'mask fill poly.png')
mask_poly=cv2.fillConvexPoly(blank_space_black,c,(255,255,255))
cv2.imwrite(file_save_poly,mask_poly)

crop_img_true=cv2.cvtColor(crop_img_true,cv2.COLOR_HSV2BGR)

file_save_mask_bitwise= os.path.join(path_save_folder,'mask bitwise and.png')
mask_bitwise_and = cv2.bitwise_and(crop_img_true,crop_img_true,mask=mask_poly)
cv2.imwrite(file_save_mask_bitwise,mask_bitwise_and)

im2= Image.open(file_save_mask_bitwise)
im2=im2.convert('RGBA')

datas=im2.getdata()
newdata=[]

for data in datas:
    if data[0]== 0 and data[1]== 0 and data[2]== 0:
        newdata.append((255,255,255,0))
    else:
        newdata.append(data)

file_save_transparent= os.path.join(path_save_folder,'transparent.png')
im2.putdata(newdata)
im2.save(file_save_transparent)

im_blur= Image.open(file_save_blur)

file_save_final= os.path.join(path_save_folder,'final.png')
im_blur.paste(im2,(xs,ys),im2)
im_blur.save(file_save_final)

im_final= Image.open(file_save_final)
im_final.show('Final Result')
cv2.waitKey(0)
cv2.destroyAllWindows()    
    















