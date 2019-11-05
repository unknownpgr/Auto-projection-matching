import cv2
import numpy as np
import time
import math

WINDOW_TITLE = 'semtle(computer)_conference'

# The frame size must be proportional to the camera stream resolution.
# Or the image will be distorted.
FRAME_W = 1280  # Frame width
FRAME_H = 960  # Frame hiehgt

# Camera stream
CV_CAMERA = cv2.VideoCapture(0)


# Camera test function. display camera input until the Q key is pressed.
def camTest():
    while True:
        _, img = CV_CAMERA.read()
        img = cv2.resize(img, (FRAME_W, FRAME_H))
        cv2.imshow(WINDOW_TITLE, img)
        key = cv2.waitKey(1)
        if key is ord('q'):
            break


# Automatic align function.
def align():

    # Display a red dot on screen and get projected position.
    def projA(x, y):
        img = np.ndarray([FRAME_H, FRAME_W, 3], np.float32)
        cv2.circle(img, (x, y), 32, (0, 0, 255), -1)
        cv2.imshow(WINDOW_TITLE, img)
        cv2.waitKey(200)
        _, img = CV_CAMERA.read()
        img = cv2.resize(img, (FRAME_W, FRAME_H))
        imgR = (img[:, :, 0] < 100) & (
            img[:, :, 1] < 100) & (img[:, :, 2] > 150)

        # Calculate center of red dot.
        xt = 0
        yt = 0
        c = 0
        for i in range(FRAME_H):
            for j in range(FRAME_W):
                if imgR[i, j]:
                    xt += j
                    yt += i
                    c += 1

        # Exception.
        if c == 0:
            print('No red dot detected.')
            cv2.imshow(WINDOW_TITLE, img)
            cv2.waitKey()
            exit()

        return (int(xt/c), int(yt/c))

    # Project the red circle placed on the four corners of the frame.
    pointsDst = [(32, 32), (FRAME_W - 32, 32),
                 (32, FRAME_H-32), (FRAME_W-32, FRAME_H-32)]
    pointsSrc = []
    cx = 0
    cy = 0

    # Get the pointSrc and center of projected points.
    for p in pointsDst:
        proj = projA(p[0], p[1])
        pointsSrc.append(proj)
        cx += proj[0]
        cy += proj[1]

    # Type convert
    pointsDst = np.float32(pointsDst)
    pointsSrc = np.float32(pointsSrc)

    # Return the transfrom matrix and center of projected dots.
    return cv2.getPerspectiveTransform(pointsSrc, pointsDst), int(cx/4), int(cy/4)


# Start
videoSource = 'sample.mp4'
scalar = 0.5

# Camera test and get the projection matrix
camTest()
perspectiveMatrix, cx, cy = align()

# Get the video stream
video = cv2.VideoCapture(videoSource)

# Get the video size and some constants
ret, frame = video.read()
h, w, _ = frame.shape
h = int(h*scalar)
w = int(w*scalar)
h_ = int(h/2)
w_ = int(w/2)

# Frame position-related constants
xs = max(0, cx-w_)
xf = min(cx+w_, FRAME_W)
ys = max(cy-h_, 0)
yf = min(cy+h_, FRAME_H)

fw = xf-xs
fh = yf-ys

while True:
    # Get the default image(background)
    img = np.ndarray([FRAME_H, FRAME_W, 3], np.uint8)

    # Read the frame from video
    ret, frame = video.read()

    # Check frame
    if not ret:
        break

    # Resize the frame with scalar
    frame = cv2.resize(frame, (w, h))

    # Paste frame on default image
    img[ys:yf, xs:xf] = frame[0:fh, 0:fw]

    # Projection
    img = cv2.warpPerspective(img, perspectiveMatrix, (FRAME_W, FRAME_H))

    # Display
    cv2.imshow(WINDOW_TITLE, img)

    key = cv2.waitKey(30)
    if key is ord('q'):
        break

# Relase camera
CV_CAMERA.release()
