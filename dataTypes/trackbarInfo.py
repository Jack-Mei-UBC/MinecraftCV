import cv2 as cv


class TrackBarInfo:
    lHue = None
    lSat = None
    lVal = None
    hHue = None
    hSat = None
    hVal = None
    lGray = None
    hGray = None

    @staticmethod
    def createTrackBar():
        cv.namedWindow("Parameters")
        cv.resizeWindow("Parameters", 640, 340)
        cv.createTrackbar("lHue", "Parameters", 0, 255, lambda a: None)
        cv.createTrackbar("hHue", "Parameters", 255, 255, lambda a: None)
        cv.createTrackbar("lSat", "Parameters", 0, 255, lambda a: None)
        cv.createTrackbar("hSat", "Parameters", 5, 255, lambda a: None)
        cv.createTrackbar("lVal", "Parameters", 70, 255, lambda a: None)
        cv.createTrackbar("hVal", "Parameters", 140, 255, lambda a: None)
        cv.createTrackbar("lGray", "Parameters", 0, 255, lambda a: None)
        cv.createTrackbar("hGray", "Parameters", 255, 255, lambda a: None)

    def grabTrackBarInfo(self):
        self.lHue = cv.getTrackbarPos("lHue", "Parameters")
        self.lSat = cv.getTrackbarPos("lSat", "Parameters")
        self.lVal = cv.getTrackbarPos("lVal", "Parameters")
        self.hHue = cv.getTrackbarPos("hHue", "Parameters")
        self.hSat = cv.getTrackbarPos("hSat", "Parameters")
        self.hVal = cv.getTrackbarPos("hVal", "Parameters")
        self.lGray = cv.getTrackbarPos("lGray", "Parameters")
        self.hGray = cv.getTrackbarPos("hGray", "Parameters")

    def getInfo(self):
        return (self.lHue,
                self.lSat,
                self.lVal,
                self.hHue,
                self.hSat,
                self.hVal,
                self.lGray,
                self.hGray)
