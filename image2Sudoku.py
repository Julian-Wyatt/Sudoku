from typing import Any, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.training.tracking.tracking import AutoTrackable

class imagingInformation:
    def __init__(self, img, threshValue=130):
        self.img = cv2.imread(img)
        self.imgName = img
        self.initialImaging(threshValue)

        self.lineDetection()

        # contours, hierarchy = cv2.findContours(thresholdGray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # img = cv2.drawContours(img, contours, -1, (0,255,0), 3)

        self.drawLines()

        self.findDifferences()

        # print("Y Mean:",np.mean(Ydiffs),"Y Median",np.median(Ydiffs))
        # print("X Mean:",np.mean(Xdiffs),"X Median",np.median(Xdiffs))

        try:
            self.findBoxes()


            self.finalGrid = np.array(self.finalGrid)
            # means, finalGrid = findBoxes(Ydiffs, Xdiffs)

            self.predictor: tf.keras.models = tf.keras.models.load_model("./Model/saved28x28NumberPredictor")
            self.feedDict = {0: 1,1: 2,2: 3,3: 4,4: 5,5: 6,6: 7,7: 8,8: 9}
            self.predict()

            self.show()
        except missedLinesException:
            self.__init__(self.imgName, 225)





    def initialImaging(self, threshValue):
        print(self.img.shape)
        if (self.img.shape[0] > 750):
            self.img = cv2.resize(self.img, (0, 0), fx=0.4, fy=0.4)
        else:
            # Initialize arguments for the filter
            top = int(0.01 * self.img.shape[0])  # shape[0] = rows
            bottom = top
            left = int(0.01 * self.img.shape[1])  # shape[1] = cols
            right = left
            self.img = cv2.copyMakeBorder(self.img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)

        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.gray = cv2.GaussianBlur(self.gray, (1, 1), cv2.BORDER_DEFAULT)

        # thresholdGray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #             cv2.THRESH_BINARY,33,14)

        blah, self.thresholdGray = cv2.threshold(self.gray, threshValue, 255, cv2.THRESH_BINARY_INV)
        # 130 or 225
        blah, self.hardThreshold = cv2.threshold(self.gray, 200, 255, cv2.THRESH_BINARY_INV)


    def lineDetection(self):
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html

        # edges = cv2.Canny(thresholdGray,100,200, apertureSize=3)
        self.edges = auto_canny(self.thresholdGray)

        self.lines = cv2.HoughLines(self.edges, 1, np.pi / 180, 145)



    def drawLines(self):
        self.horizontal = []
        self.vertical = []

        for line in self.lines:
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
                self.horizontal.append(line[0])
                cv2.line(self.img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif theta == 0:
                self.vertical.append(line[0])

                cv2.line(self.img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.vertical = sorted([i[0] for i in self.vertical])
        self.horizontal = sorted([i[0] for i in self.horizontal])
        # rho = x Cos(theta) + y Sin(theta)
        # cos (0) = 1; cos(1.57...) = 0


    def findDifferences(self):
        largestXDiff = 0
        self.Xdiffs = []
        for i in range(len(self.vertical) - 1, 0, -1):
            diff = self.vertical[i] - self.vertical[i - 1]
            # print(self.vertical[i],self.vertical[i-1],diff)
            if diff > largestXDiff:
                largestXDiff = diff
            self.Xdiffs.append((int(self.vertical[i - 1]), int(self.vertical[i]), diff))
            # Xdiffs.append(diff)
        self.Xdiffs.sort()
        # print("\n\n\n\nYDiffs")
        largestYDiff = 0
        self.Ydiffs = []
        for i in range(len(self.horizontal) - 1, 0, -1):
            diff = self.horizontal[i] - self.horizontal[i - 1]
            # print(self.horizontal[i],self.horizontal[i-1],diff)
            if diff > largestYDiff:
                largestYDiff = diff
            self.Ydiffs.append((int(self.horizontal[i - 1]), int(self.horizontal[i]), diff))
            # Ydiffs.append(diff)
        self.Ydiffs.sort()
        # print(Ydiffs)




    def findBoxes(self):
        only_diffs = [i[2] for i in self.Ydiffs if i[2] > 5]
        only_diffs2 = [i[2] for i in self.Xdiffs if i[2] > 5]

        if self.img.shape[0] < 400:
            boxMin = int(np.ceil((np.mean([np.mean(only_diffs), np.median(only_diffs), np.mean(only_diffs2), np.median(
                only_diffs2)])) * 0.6))
            # boxMax = int((np.mean([np.mean(only_diffs),np.median(only_diffs),np.mean(only_diffs2),np.median(only_diffs2)]))*1.5)
            boxMax = int(np.ceil((np.mean([np.mean(only_diffs), np.median(only_diffs), np.mean(only_diffs2), np.median(
                only_diffs2)])) * 1.7))
        else:
            boxMin = int(np.ceil((np.mean([np.mean(only_diffs), np.median(only_diffs), np.mean(only_diffs2), np.median(
                only_diffs2)])) * 0.7))
            # boxMax = int((np.mean([np.mean(only_diffs),np.median(only_diffs),np.mean(only_diffs2),np.median(only_diffs2)]))*1.5)
            boxMax = int(np.ceil((np.mean([np.mean(only_diffs), np.median(only_diffs), np.mean(only_diffs2), np.median(
                only_diffs2)])) * 1.3))
        #
        # print(boxMin, boxMax)
        # print(self.Ydiffs)
        # print(self.Xdiffs)
        image = 1
        self.finalGrid = []
        self.means = []

        for i in range(len(self.Xdiffs)):
            regions = []
            if len(self.finalGrid) == 9:
                break
            if (self.Xdiffs[i][2]) > boxMin and self.Xdiffs[i][2] < boxMax:

                for j in range(len(self.Ydiffs)):
                    if len(regions) == 9:
                        break
                    if (self.Ydiffs[j][2]) > boxMin and self.Ydiffs[j][2] < boxMax:


                        roi = self.hardThreshold[self.Ydiffs[j][0]:self.Ydiffs[j][1], self.Xdiffs[i][0]:self.Xdiffs[
                            i][1]]

                        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

                        if np.mean(roi) < 100:
                            roi = cv2.bitwise_not(roi)

                        roi = roi[2:28, 4:28]

                        # print(image,np.mean(roi[0,0:-1]))
                        while np.mean(roi[0,0:-1]) < 100:
                            roi = roi[1:-1,:]
                        while np.mean(roi[0:-1,0]) < 100:
                            roi = roi[:,1:-1]


                        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                        # cv2.imshow("roi"+str(image),roi)
                        # print(image,np.mean(roi))
                        self.means.append(np.mean(roi[4:-4,4:-4]))
                        regions.append(roi)
                        image += 1

            if len(regions) == 9:
                self.finalGrid.append(regions)
            else:
                if len(regions) == 0:
                    continue
                print("region length", len(regions))
                if len(regions) < 9:
                    # for i in range(len(regions)):
                    #     cv2.imshow(str(i), regions[i])
                    raise missedLinesException



    def predict(self):
        totalNums = 0
        totalBlanks = 0
        print(len(self.means))
        values = []
        for i in range(len(self.means)):
            if self.means[i] < 245:
                # if self.means[i] > 30:
                # print(i + 1, self.means[i])
                totalNums += 1

                # cv2.imshow("roiNumber" + str(i + 1), self.finalGrid[int(i / 9), i % 9])

                # cv2.imwrite("./fontImages/"+str(i+1)+".jpg",self.finalGrid[int(i/9),i%9])
                prediction = tf.argmax(self.predictor.predict(self.finalGrid[int(i / 9), i % 9].reshape(1, 28, 28,
                                                                                                       1)), 1)

                prediction = prediction.numpy()
                values.append(self.feedDict[prediction[0]])
                # print(i+1, prediction[0]+1)

            else:
                # print(i + 1, self.means[i])
                values.append(0)
                totalBlanks += 1
                # cv2.imshow("roiBlank" + str(i + 1), self.finalGrid[int(i / 9), i % 9])

        self.finalGrid = np.array(values).reshape((9, 9)).transpose()

        print(self.finalGrid)
        # print(totalBlanks, totalNums)

    def show(self):
        cv2.imshow("image", self.img)
        cv2.imshow("edges", self.edges)
        cv2.imshow("thresh", self.thresholdGray)
        cv2.imshow("hardThres",self.hardThreshold)
        cv2.imshow("original gray", self.gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class Solver:
    def __init__(self,mat):
        self.initial = mat


    # https://www.youtube.com/watch?v=G_UYXzGuqvM
    def possible(self,x,y,n,grid):
        """Is n possible in position x,y"""
        for i in range(9):
            if grid[x,i] == n:
                return False
            if grid[i,y] == n:
                return False

        x0 = (x//3)*3
        y0 = (y//3)*3

        for i in range(3):
            for j in range(3):
                if grid[x0+i,y0+j] == n:
                    return False
        return True

    def backtrack(self,grid):
        for i in range(9):
            for j in range(9):
                if grid[i,j] == 0:
                    for k in range(1,10):
                        if self.possible(i,j,k,grid):
                            grid[i,j] = k
                            self.backtrack(grid)
                            grid[i, j] = 0
                    return

        print(grid)






class Error(Exception):
    """Base class for exceptions"""
    pass

class missedLinesException(Error):
    """When analysing """


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


newImage = imagingInformation("./images/IMG_2840.png")
# newImage = imagingInformation("./images/10.png")
print("\n\n\n")
newSolver = Solver(newImage.finalGrid)
newSolver.backtrack(newSolver.initial)

# newImage = imagingInformation("./images/1.png")
# newImage = imagingInformation("./images/8.jpg")


# predictor.summary()

#TODO: Look at doing if mean is greater than 150, then the numbers are 0.95*mean, if mean is less than 150,
# numbers are 1.05*mean
# https://discussions.apple.com/thread/250905135
