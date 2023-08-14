import cv2

capture = cv2.VideoCapture(r"C:\Users\ayush\Downloads\beaker_good_fast_sands_trimmed.mp4")

currentframe = 0
counter = 10 #captures every nth image

while (True):
    counter -= 1
    # reading from frame
    ret, frame = capture.read()

    if ret and counter == 0:
        name = r"C:\Users\ayush\Documents\IMG_FROM_VIDEO\frame" + str(currentframe) + '.jpg'
        print('Creating...' + name)

        cv2.imwrite(name, frame)

        currentframe += 1
        counter = 10
    else:
        break

capture.release()
cv2.destroyAllWindows()
