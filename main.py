from yoloUtil import *
printAmit()

# Usage example:  python main.py --video=test.mp4
#                 python main.py --image=<ImageFilePath>

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

# Default Parameters
inpWidth = 416       #Width of Darknet input image
inpHeight = 416      #Height of Darknet input image

parser = argparse.ArgumentParser(description='YOLO Object Detection using OpenCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# Process inputs
winName = 'YOLO(Darknet) Object Detection'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_output.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_output.avi'
else:
    # Webcam input
    outputFile = "yolo_output.avi"
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 24, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

# Give the configuration and weight files for the model and load the network using them.
net = loadDarknetModel()

while cv.waitKey(1) < 0:
    
    # get frame from the video
    hasMoreFrame, frame = cap.read()
    
    # Stop the program if reached end of video
    if not hasMoreFrame:
        print("Done processing !!! No further frames to process.")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    output = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocessing(frame, output)

    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8));
    else:
        vid_writer.write(frame.astype(np.uint8))

    cv.imshow(winName, frame)

