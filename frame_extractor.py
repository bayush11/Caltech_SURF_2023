#This code is used to extract frames from a given video. Down below you can see the 
#video being accessed as well as the nth frame of all of the frames being extracted. 
#You have the ability to change this value to whatever you want.

import cv2

capture = cv2.VideoCapture(r"C:\Users\ayush\Downloads\beaker_good_fast_sands_trimmed.mp4") #reads the video file

currentframe = 0
counter = 10 #captures every nth image

while (True):
    counter -= 1
    # reading from frame
    ret, frame = capture.read()

    if ret and counter == 0:
        name = r"C:\Users\ayush\Documents\IMG_FROM_VIDEO\frame" + str(currentframe) + '.jpg' #this is the name of the frame
        print('Creating...' + name)

        cv2.imwrite(name, frame)

        currentframe += 1
        counter = 10 #nth fame is taken and this number can be changed to any.
    else:
        break

capture.release()
cv2.destroyAllWindows()
