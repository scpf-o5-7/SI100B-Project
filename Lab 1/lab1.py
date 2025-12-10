import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Generate an empty image with 480 height and 640 width
    blanks = np.zeros((480, 640, 3), dtype=np.uint8)  # Create a black image

    ### FILL red when x in range [0, 160] and y in range [0, 120]
    blanks[0:120, 0:160] = [0, 0, 255]  # BGR red

    ### FILL GREEN when x in range [160, 320] and y in range [120, 240]
    blanks[120:240, 160:320] = [0, 255, 0]  # BGR green

    ### FILL GREEN when x in range [320, 480] and y in range [240, 360]
    blanks[240:360, 320:480] = [0, 255, 0]  # BGR green

    ### FILL GRAY[128, 128, 128] when x in range [480, 640] and y in range [360, 480]
    blanks[360:480, 480:640] = [128, 128, 128]  # BGR gray

    cv2.imwrite("create.png", blanks)
    cv2.imshow("src", blanks)
    cv2.waitKey(0)
