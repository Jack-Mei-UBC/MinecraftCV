import cv2 as cv
import numpy as np
from dataTypes.trackbarInfo import *


class Vision:

    @staticmethod
    def DetectMap(image: np.ndarray) -> (np.ndarray, int, int, int, int):
        trackbar = TrackBarInfo()
        trackbar.grabTrackBarInfo()
        lHue, lSat, lVal, hHue, hSat, hVal, lGray, hGray = trackbar.getInfo()

        # Define the lower and upper thresholds for the desired color range
        transformed = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        transformed_blurred = cv.GaussianBlur(transformed, (3, 3), cv.IMREAD_UNCHANGED)

        mask = cv.inRange(transformed_blurred, np.array([lHue, lSat, lVal])
                          , np.array([hHue, hSat, hVal]))
        # Apply the mask to the original image

        thresholded = cv.bitwise_and(image, image, mask=mask)
        grayscale = cv.cvtColor(thresholded, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(grayscale, (3, 3), cv.IMREAD_UNCHANGED)
        # Apply Canny edge detection
        edges = cv.Canny(blurred, lGray, hGray)
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        copy = image.copy()
        largestContour = (0, None)
        for i, c in enumerate(contours):
            epsilon = 0.02 * cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, epsilon, True)  # get the bounding rect
            x, y, w, h = cv.boundingRect(c)
            # Check if the polygon has four vertices (a rectangle)
            if len(approx) == 4 and cv.contourArea(c):
                cv.drawContours(copy, [approx], 0, (36, 255, 12), 2)
                if w * h > largestContour[0]:
                    largestContour = (cv.contourArea(c), c)
            # else:
                cv.drawContours(copy, [approx], 0, (255, 255, 0), 2)
        x, y, w, h = cv.boundingRect(largestContour[1])
        cv.rectangle(copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # vis = np.concatenate((image, copy), axis=1)
        # copy = cv.resize(copy, (copy.shape[1] // 2, copy.shape[0] // 2))
        # cv.drawContours(copy,  contours, -1, (0,255,0), 3)
        cv.imshow("Detected Map", copy)
        cv.imshow('Detected Edges', edges)

        subImage = copy[y:y + h, x:x + w]
        return subImage, x, y, w, h


def trimImage(image: np.ndarray, w, h):
    # Trim center triangle
    image = image[22:h - 22, 22:w - 22]
    h, w, _ = image.shape
    image[h // 2 - 16:h // 2 + 16, w // 2 - 16:w // 2 + 16] = np.array([0, 0, 0])
    # Trim border + NWES icons
    return image


if __name__ == '__main__':
    images = []
    for i in range(4):
        images += [cv.imread(f'needles/vault{i}.png')]
    TrackBarInfo.createTrackBar()
    maps = []
    for i in range(len(images)):
        map, x, y, w, h = Vision.DetectMap(images[i])
        map = trimImage(map, w, h)
        maps += [map]
    maps.reverse()
    stitcher = cv.Stitcher.create(cv.STITCHER_SCANS)
    stitcher.setPanoConfidenceThresh(1.0)
    status, result = stitcher.stitch(maps)
    print(status)
    cv.imshow("map", result)
    if cv.waitKey(0) == ord('q'):
        cv.destroyAllWindows()
    # while True:
    #     map, x, y, w, h = Vision.DetectMap(image)
    #     map = trimImage(map,x,y,w,h)
    #     # cv.imshow("map", map)
    #     cv.imwrite("Screencapture/map1.png", map)
    #     if cv.waitKey(1) == ord('q'):
    #         cv.destroyAllWindows()
