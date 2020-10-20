import numpy as np
import cv2
from abc import ABCMeta
import os
import imutils
from imutils import contours
from scipy.spatial import distance as dist
from imutils import perspective
from abc import ABCMeta,abstractmethod

def midpoint(ptA, ptB):
    return ((ptA[0]+ptB[0])*0.5, (ptA[1]+ptB[1])*0.5)

class Rdp(metaclass=ABCMeta):
    def __init__(self,image,width):
        try:
            assert isinstance(image,str)
        except AssertionError:
            print("Error: Please enter the correct file name and make sure it is in parentheses.")
        else:
            try:
                if not os.path.exists(image):
                    raise FileNotFoundError()
            except FileNotFoundError:
                print("Error: Can't found the image!")
            else:
                print("OK! We,found the image!")

                self.originImage = cv2.imread(image)
                self.grayImage = None
                self.blur_grayImage = None
                self.edged = None
                self.cnts = None
                self.box = []
                self.orig = None
                self.width = width
                self.pixelsPerMetric = None

    @ abstractmethod
    def init(self,blur=(9,9),r1=50,r2=100):
        pass

    def foundCounters(self,path = "./",size = 120):
        print("共发现%d个“点”" %self.cnts.shape[0])
        count = 0
        self.orig = self.originImage.copy()
        print("=======================")
        print("过滤开始：")
        print("=======================")
        for c in self.cnts:
            if cv2.contourArea(c) > size:
                count += 1
                print("过滤后点%d的面积为(%.3f)" %(count,cv2.contourArea(c)))
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box,dtype="int")
                box = perspective.order_points(box)
                cv2.drawContours(self.orig,[box.astype("int")],-1,(0,0,255),3)
                for (x, y) in box:
                    cv2.circle(self.orig, (int(x), int(y)), 8, (0,255,0), -1)
                self.box.append(box)
                
        # cv2.imshow("Image",self.orig)
        # cv2.waitKey(0)
        cv2.imwrite(path+"endding.jpg",self.orig)

    def saveGrayAndbw(self,path="./"):
        try:
            assert isinstance(path,str)
        except AssertionError:
            print("Error: Please enter the correct path and make sure it is in parentheses.")
        else:
            cv2.imwrite(path+"gray.jpg",self.grayImage)
            cv2.imwrite(path+"blur.jpg",self.blur_grayImage)
            cv2.imwrite(path+"bw.jpg",self.edged)


    def inch(self,path='./'):
        for b in self.box:
            (tl, tr, br, bl) = b
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            # 在图中画出中点
            cv2.circle(self.orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(self.orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(self.orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(self.orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
            cv2.line(self.orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 3)
            cv2.line(self.orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 3)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            if self.pixelsPerMetric is None:
                self.pixelsPerMetric = dB / self.width
            dimA = dA / self.pixelsPerMetric
            dimB = dB / self.pixelsPerMetric
 
            cv2.putText(self.orig, "{:.1f}mm".format(dimA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 0), 2)
            cv2.putText(self.orig, "{:.1f}mm".format(dimB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 0), 2)
            cv2.imwrite(path+"Endding_d.jpg",self.orig)

class PRdp(Rdp):
    def __init__(self,image,width):
        super(PRdp,self).__init__(image,width)
    def init(self,blur=(9,9),r1=50,r2=100):
        self.orig = self.originImage.copy()
        self.grayImage = cv2.cvtColor(self.originImage,cv2.COLOR_BGR2GRAY)
        self.blur_grayImage = cv2.GaussianBlur(self.grayImage,blur,0)
        self.edged = cv2.Canny(self.blur_grayImage,r1,r2)
        self.edged = cv2.dilate(self.edged, None, iterations=1)
        self.edged = cv2.erode(self.edged, None, iterations=1)
        self.cnts = cv2.findContours(self.edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        print(self.cnts[1])
        self.cnts = np.array(imutils.grab_contours(self.cnts))
        cv2.drawContours(self.orig,self.cnts,-1,(0,255,0),3)
        cv2.imshow("image",self.orig)
        cv2.waitKey(0)
    def foundCounters(self):
        print("yes")


class ORdp(Rdp):
    
    def __init__(self,image,width):
        super(ORdp,self).__init__(image,width)
    
    def init(self,blur=(9,9),r1=50,r2=100):
        self.orig = self.originImage.copy()
        self.grayImage = cv2.cvtColor(self.originImage,cv2.COLOR_BGR2GRAY)
        self.blur_grayImage = cv2.GaussianBlur(self.grayImage,blur,2)
        self.edged = cv2.Canny(self.blur_grayImage,r1,r2)
        self.edged = cv2.dilate(self.edged, None, iterations=1)
        self.edged = cv2.erode(self.edged, None, iterations=1)
        self.cnts = cv2.findContours(self.edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        self.cnts = np.array(imutils.grab_contours(self.cnts))