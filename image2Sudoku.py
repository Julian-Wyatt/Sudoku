from typing import Any, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.training.tracking.tracking import AutoTrackable

class imagingInformation:
    def __init__(self, img):
        self.img = img
        initialImaging(self.img)

        lineDetection()

        # contours, hierarchy = cv2.findContours(thresholdGray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # img = cv2.drawContours(img, contours, -1, (0,255,0), 3)

        vertical, horizontal = drawLines(lines)

        Ydiffs, Xdiffs = findDifferences(vertical, horizontal)

        # print("Y Mean:",np.mean(Ydiffs),"Y Median",np.median(Ydiffs))
        # print("X Mean:",np.mean(Xdiffs),"X Median",np.median(Xdiffs))


        means, finalGrid = findBoxes(Ydiffs, Xdiffs)




    def initialImaging(self, src):
        if (src.shape[0] > 750):
            src = cv2.resize(src, (0, 0), fx=0.4, fy=0.4)
        else:
            # Initialize arguments for the filter
            top = int(0.01 * src.shape[0])  # shape[0] = rows
            bottom = top
            left = int(0.01 * src.shape[1])  # shape[1] = cols
            right = left
            src = cv2.copyMakeBorder(src, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)

        self.gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        self.gray = cv2.GaussianBlur(self.gray, (1, 1), cv2.BORDER_DEFAULT)

        # thresholdGray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #             cv2.THRESH_BINARY,33,14)

        blah, self.thresholdGray = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
        blah, self.hardThreshold = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)



class Error(Exception):
    """Base class for exceptions"""
    pass

class missedLinesException(Error):
    """When analysing """


def initialImaging(src):
    if (src.shape[0] > 750):
        src = cv2.resize(src, (0, 0), fx=0.4, fy=0.4)
    else:
        # Initialize arguments for the filter
        top = int(0.01 * src.shape[0])  # shape[0] = rows
        bottom = top
        left = int(0.01 * src.shape[1])  # shape[1] = cols
        right = left
        src = cv2.copyMakeBorder(src, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (1, 1), cv2.BORDER_DEFAULT)


    # thresholdGray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #             cv2.THRESH_BINARY,33,14)

    blah, thresholdGray = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    blah, hardThreshold = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)
    return src, thresholdGray, hardThreshold, gray

# autocanny: https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/

def auto_canny(image, sigma=0.33):

    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

def lineDetection(thresholdGray):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html

    # edges = cv2.Canny(thresholdGray,100,200, apertureSize=3)
    edges = auto_canny(thresholdGray)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 145)
    return lines, edges

def drawLines(lines):
    horizontal = []
    vertical = []

    for line in lines:
        rho, theta = line[0]

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        if 1.55 < theta < 1.58:
            horizontal.append(line[0])
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif theta == 0:
            vertical.append(line[0])

            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    vertical = sorted([i[0] for i in vertical])
    horizontal = sorted(i[0] for i in horizontal)
    return vertical,horizontal
    # rho = x Cos(theta) + y Sin(theta)
    # cos (0) = 1; cos(1.57...) = 0


def findDifferences(vertical, horizontal):
    largestXDiff = 0
    Xdiffs = []
    for i in range(len(vertical) - 1, 0, -1):
        diff = vertical[i] - vertical[i - 1]
        # print(vertical[i],vertical[i-1],diff)
        if diff > largestXDiff:
            largestXDiff = diff
        Xdiffs.append((int(vertical[i - 1]), int(vertical[i]), diff))
        # Xdiffs.append(diff)
    Xdiffs.sort()
    # print("\n\n\n\nYDiffs")
    largestYDiff = 0
    Ydiffs = []
    for i in range(len(horizontal) - 1, 0, -1):
        diff = horizontal[i] - horizontal[i - 1]
        # print(horizontal[i],horizontal[i-1],diff)
        if diff > largestYDiff:
            largestYDiff = diff
        Ydiffs.append((int(horizontal[i - 1]), int(horizontal[i]), diff))
        # Ydiffs.append(diff)
    Ydiffs.sort()
    # print(Ydiffs)

    return Ydiffs, Xdiffs



def findBoxes(Ydiffs, Xdiffs):
    only_diffs = [i[2] for i in Ydiffs if i[2] > 5]
    only_diffs2 = [i[2] for i in Xdiffs if i[2] > 5]
    boxMin = int(np.ceil((np.mean([np.mean(only_diffs), np.median(only_diffs), np.mean(only_diffs2), np.median(
        only_diffs2)])) * 0.7))
    # boxMax = int((np.mean([np.mean(only_diffs),np.median(only_diffs),np.mean(only_diffs2),np.median(only_diffs2)]))*1.5)
    boxMax = int(np.ceil((np.mean([np.mean(only_diffs), np.median(only_diffs), np.mean(only_diffs2), np.median(
        only_diffs2)])) * 1.3))
    print(boxMin, boxMax)
    print(Ydiffs)
    print(Xdiffs)
    image = 1
    finalGrid = []
    means = []
    try:
        for i in range(len(Xdiffs)):
            regions = []
            if len(finalGrid) == 9:
                break
            if (Xdiffs[i][2]) > boxMin and Xdiffs[i][2] < boxMax:

                for j in range(len(Ydiffs)):
                    if len(regions) == 9:
                        break
                    if (Ydiffs[j][2]) > boxMin and Ydiffs[j][2] < boxMax:


                        roi = hardThreshold[Ydiffs[j][0]:Ydiffs[j][1], Xdiffs[i][0]:Xdiffs[i][1]]

                        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

                        if np.mean(roi) < 100:
                            roi = cv2.bitwise_not(roi)

                        roi = roi[2:28, 4:28]
                        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                        # cv2.imshow("roi"+str(image),roi)
                        # print(image,np.mean(roi))
                        means.append(np.mean(roi))
                        regions.append(roi)
                        image += 1

            if len(regions) == 9:
                finalGrid.append(regions)
            else:
                if len(regions) == 0:
                    continue
                print("region length", len(regions))
                if len(regions) < 9:
                    for i in range(len(regions)):
                        cv2.imshow(str(i), regions[i])
                    raise missedLinesException

    except missedLinesException:
        img, thresholdGray, hardThreshold, gray = initialImaging(img)

        lines, edges = lineDetection(thresholdGray)

        # contours, hierarchy = cv2.findContours(thresholdGray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # img = cv2.drawContours(img, contours, -1, (0,255,0), 3)

        vertical, horizontal = drawLines(lines)

        Ydiffs, Xdiffs = findDifferences(vertical, horizontal)


    return means, finalGrid

# img = cv2.imread("./images/IMG_2592.png")
# img = cv2.imread("./images/IMG_2833.jpeg")
img = cv2.imread("./images/10.png")

img, thresholdGray,hardThreshold, gray = initialImaging(img)

lines, edges = lineDetection(thresholdGray)

# contours, hierarchy = cv2.findContours(thresholdGray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# img = cv2.drawContours(img, contours, -1, (0,255,0), 3)

vertical, horizontal = drawLines(lines)

Ydiffs, Xdiffs = findDifferences(vertical,horizontal)


#print("Y Mean:",np.mean(Ydiffs),"Y Median",np.median(Ydiffs))
#print("X Mean:",np.mean(Xdiffs),"X Median",np.median(Xdiffs))


means, finalGrid = findBoxes(Ydiffs,Xdiffs)

finalGrid = np.array(finalGrid)
print(finalGrid.shape)
print(np.mean(means),len(means))


predictor: tf.keras.models = tf.keras.models.load_model("./Model/saved28x28NumberPredictor")
# predictor.summary()

#TODO: Look at doing if mean is greater than 150, then the numbers are 0.95*mean, if mean is less than 150,
# numbers are 1.05*mean
# https://discussions.apple.com/thread/250905135
totalNums = 0
totalBlanks =0

values = []
for i in range(len(means)):
    if means[i] < 245:
    # if means[i] > 30:
        print(i+1, means[i])
        totalNums+=1

        cv2.imshow("roiNumber"+str(i+1),finalGrid[int(i/9),i%9])
        # cv2.imwrite("./fontImages/"+str(i+1)+".jpg",finalGrid[int(i/9),i%9])
        prediction = tf.argmax(predictor.predict(finalGrid[int(i/9),i%9].reshape(1,28,28,1)),1)

        prediction = prediction.numpy()
        values.append(prediction[0]+1)
        # print(i+1, prediction[0]+1)

    else:
        print(i+1,means[i])
        values.append(0)
        totalBlanks+=1
        cv2.imshow("roiBlank" + str(i + 1), finalGrid[int(i / 9), i % 9])

values = np.array(values).reshape((9,9)).transpose()

print(values)
print(totalBlanks,totalNums)
# print(finalGrid[0])
# print(finalGrid)

cv2.imshow("image",img)
cv2.imshow("edges",edges)
cv2.imshow("thresh",thresholdGray)
cv2.imshow("original gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()