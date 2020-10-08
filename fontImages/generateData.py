import cv2
import numpy as np
import os
import random

#https://theailearner.com/2019/05/07/add-different-noise-to-an-image/
kernel = np.ones((3,3),np.uint8)
def add_Gauss_Noise(img):
    # Generate Gaussian noise
    gauss = np.random.normal(0, random.random()*0.75, img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1]).astype('uint8')
    # Add the Gaussian noise to the image
    img_gauss = cv2.add(img, gauss)

    return img_gauss

def add_s_p_Noise(img):
    gauss = np.random.normal(0,random.random()*0.75 , img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1]).astype('uint8')
    noise = img + img * gauss
    return noise

def dilate(img,name):
    img = cv2.bitwise_not(img)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.bitwise_not(img)
    cv2.imshow("dilated"+str(name),img)

    return img

def erode(img,name):
    img = cv2.bitwise_not(img)
    img = cv2.erode(img,kernel,iterations=1)
    img = cv2.bitwise_not(img)
    cv2.imshow("eroded"+str(name), img)
    return img

def randomWarp(img):
    rows, cols = img.shape
    M = np.float32([[1, 0, int(random.random()*10)-5], [0, 1, int(random.random()*6)-3]])
    dst = cv2.warpAffine(img, M, (cols, rows), borderValue=(255,255,255))
    return dst

# image = cv2.imread("./4/4.png", cv2.IMREAD_GRAYSCALE)
# dilate(image)
# erode(image)
#
# image = cv2.imread("./1/1-2843.jpg", cv2.IMREAD_GRAYSCALE)
# dilate(image)
# erode(image)
print(os.getcwd())
#
# number = 0
# for i in range(1,10):
#     files = os.listdir("./" + str(i) + "/")
#     for j in files:
#         print(j)
#         if j==".DS_Store":
#             continue
#         img = cv2.imread("./" + str(i) + "/"+j)
#
#
#         dilated = dilate(img,number)
#         eroded = erode(img,number)
#
#         if number==0:
#             cv2.imwrite("./" + str(i) + "/eroded0.jpg",eroded)
#             cv2.imwrite("./" + str(i) + "/dilated0.jpg", dilated)
#
#         if number == 4:
#             cv2.imwrite("./" + str(i) + "/dilated4.jpg", dilated)
#
#         if number == 3:
#             cv2.imwrite("./" + str(i) + "/eroded3.jpg", eroded)
#
#         if number == 7:
#             cv2.imwrite("./" + str(i) + "/eroded7.jpg", eroded)
#             cv2.imwrite("./" + str(i) + "/dilated7.jpg", dilated)
#
#         if number == 9:
#             cv2.imwrite("./" + str(i) + "/dilated9.jpg", dilated)
#         if number == 42:
#             cv2.imwrite("./" + str(i) + "/eroded42.jpg", eroded)
#             cv2.imwrite("./" + str(i) + "/dilated42.jpg", dilated)
#
#         if number == 41:
#             cv2.imwrite("./" + str(i) + "/dilated41.jpg", dilated)
#
#         if number == 38:
#             cv2.imwrite("./" + str(i) + "/dilated38.jpg", dilated)
#
#         if number == 36:
#             cv2.imwrite("./" + str(i) + "/eroded36.jpg", eroded)
#
#         if number == 34:
#             cv2.imwrite("./" + str(i) + "/eroded34.jpg", eroded)
#             cv2.imwrite("./" + str(i) + "/dilated34.jpg", dilated)
#
#         if number == 33:
#             cv2.imwrite("./" + str(i) + "/dilated33.jpg", dilated)
#
#         if number == 25:
#             cv2.imwrite("./" + str(i) + "/eroded34.jpg", eroded)
#             cv2.imwrite("./" + str(i) + "/dilated34.jpg", dilated)
#
#         if number == 29:
#             cv2.imwrite("./" + str(i) + "/dilated33.jpg", dilated)
#
#         if number == 21:
#             cv2.imwrite("./" + str(i) + "/eroded34.jpg", eroded)
#             cv2.imwrite("./" + str(i) + "/dilated34.jpg", dilated)
#
#         if number == 23:
#             cv2.imwrite("./" + str(i) + "/dilated33.jpg", dilated)
#
#         if number == 19:
#             cv2.imwrite("./" + str(i) + "/eroded34.jpg", eroded)
#             cv2.imwrite("./" + str(i) + "/dilated34.jpg", dilated)
#
#         if number == 18:
#             cv2.imwrite("./" + str(i) + "/dilated33.jpg", dilated)
#
#         if number == 12:
#             cv2.imwrite("./" + str(i) + "/eroded34.jpg", eroded)
#             cv2.imwrite("./" + str(i) + "/dilated34.jpg", dilated)
#
#         if number == 13:
#             cv2.imwrite("./" + str(i) + "/dilated33.jpg", dilated)
#
#
#
#
#         number+=1


for i in range (1,10):

    try:
        files = os.listdir("./"+str(i)+"/")
        files.remove(".DS_Store")
    except ValueError:
        pass
    # files = ["7-1.jpg","7-2.jpg","7-3.jpg","7-4.jpg","7-5.jpg","7-6.jpg","7-7.jpg","7-8.jpg","7-9.jpg","7-10.jpg",
    #          "7-22.jpg"]
    images = []
    totalImages = 0
    for k in range(50):
        for j in range(len(files)):
            if files[j][0:9] != "Generated":

                temp = cv2.imread("./"+str(i)+"/"+str(files[j]), cv2.IMREAD_GRAYSCALE)

                images.append(temp)
                try:
                    pass

                    cv2.imwrite("./"+str(i)+"/Generated-" + str(totalImages+1) + "-Warped-sp.jpg", add_s_p_Noise(
                        randomWarp(temp)))


                except AttributeError:

                    pass


                totalImages += 2

    for j in range(len(files)):
        temp = cv2.imread("./"+str(i)+"/"+str(files[j]), cv2.IMREAD_GRAYSCALE)
        if temp is None:
            print(files[j])



for i in range (1,10):
    files = os.listdir("./"+str(i)+"/")
    print(i,len(files))



cv2.waitKey(0)
cv2.destroyAllWindows()