import cv2 as cv
import os
import time

from dataTypes.trackbarInfo import TrackBarInfo
from windowcapture import WindowCapture
from vision import Vision
import pyautogui

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# initialize the WindowCapture class
# wincap = WindowCapture('Minecraft* 1.18.2 - Multiplayer (3rd-party Server)')
def record():
    # Change the working directory to the folder this script is in.
    # Doing this because I'll be putting the files from each video in their own folder on GitHub
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # initialize the WindowCapture class
    firefox = None
    for x in pyautogui.getAllWindows():
        if "Minecraft" in x.title:
            firefox = x.title
            print("Found " + firefox)
    wincap = WindowCapture(firefox)
    WindowCapture.list_window_names()
    # initialize the Vision class
    # vision_limestone = Vision('')

    '''
    # https://www.crazygames.com/game/guns-and-bottle
    wincap = WindowCapture()
    vision_gunsnbottle = Vision('gunsnbottle.jpg')
    '''

    loop_time = time.time()
    Max_FPS = 15

    TrackBarInfo.createTrackBar()
    while True:

        # get an updated image of the game
        screenshot = wincap.get_screenshot()

        # display the processed image
        # points = vision_limestone.find(screenshot, 0.5, 'rectangles')
        # points = vision_gunsnbottle.find(screenshot, 0.7, 'points')
        Vision.DetectMap(screenshot)
        # debug the loop rate
        # screenshot = cv.putText(screenshot, 'FPS {}'.format(1 // (time.time() - loop_time)), (0,14), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        # loop_time = time.time()
        # cv.imshow("Capture", screenshot)

        # press 'q' with the output window focused to exit.
        # waits 1 ms every loop to process key presses
        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break
        sleepTime = 1/Max_FPS - (time.time() - loop_time)
        if sleepTime > 0:
            time.sleep(sleepTime)

    print('Done.')
