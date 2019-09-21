from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *
base = BaseOverlay("base.bit")

# Initialize Webcam and HDMI Out
# Monitor configuration: 640*480 @ 60Hz

Mode = VideoMode(640,480,24)
hdmi_out = base.video.hdmi_out
hdmi_out.configure(Mode,PIXEL_BGR)
hdmi_out.start()

# Initialize Webcam and HDMI Out
# Monitor (output) frame buffer size
frame_out_w = 1920
frame_out_h = 1080
# Camera (input) configuration
frame_in_w = 640
frame_in_h = 480


# Initialize camera from OpenCV
import cv2

videoIn = cv2.VideoCapture(0)
videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);

print("Capture device is open: " + str(videoIn.isOpened()))

#Apply the face detection to the input

import cv2

# Capture webcam image
import numpy as np

# Init frame variables
first_frame = None
next_frame = None

font = cv2.FONT_HERSHEY_SIMPLEX
delay_counter = 0
movement_persistent_counter = 0
motionless_counter = 0
isFall = False
isAlarm = False
text = ""

# Number of frames to pass before changing the frame to compare the current
# frame against
FRAMES_TO_PERSIST = 10

# Minimum boxed area for a detected motion to count as actual motion
# Use to filter out noise or small objects
MIN_SIZE_FOR_MOVEMENT = 500

# Minimum length of time where no motion is detected it should take
#(in program cycles) for the program to declare that there is no movement
MOVEMENT_DETECTED_PERSISTENCE = 5 #100

MOTIONLESS_HELP = 10 #30 for 10 secs

lastH = [0]*100
lastW = [0]*100
i = 0

while (True):
    frame_delta = None
    # Set transient motion detected as false
    transient_movement_flag = False

    ret, frame_vga = videoIn.read()

    # Display webcam image via HDMI Out
    if (ret):      
        outframe = hdmi_out.newframe()
        outframe[0:480,0:640,:] = frame_vga[0:480,0:640,:]
        hdmi_out.writeframe(outframe)
    else:
        raise RuntimeError("Failed to read from camera.")

    np_frame = frame_vga

    # face_cascade = cv2.CascadeClassifier(
    #     '/home/xilinx/jupyter_notebooks/base/video/data/'
    #     'haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier(
    #     '/home/xilinx/jupyter_notebooks/base/video/data/'
    #     'haarcascade_eye.xml')

    gray = cv2.cvtColor(np_frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Blur it to remove camera noise (reducing false positives)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # If the first frame is nothing, initialise it
    if first_frame is None: first_frame = gray    

    delay_counter += 1

    # Otherwise, set the first frame to compare as the previous frame
    # But only if the counter reaches the appriopriate value
    # The delay is to allow relatively slow motions to be counted as large
    # motions if they're spread out far enough
    if delay_counter > FRAMES_TO_PERSIST:
        delay_counter = 0
        first_frame = next_frame

   # Set the next frame to compare (the current frame)
    next_frame = gray

    # Compare the two frames, find the difference
    frame_delta = cv2.absdiff(first_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 100, 255, cv2.THRESH_BINARY)[1]

    # Fill in holes via dilate(), and find contours of the thesholds
    thresh = cv2.dilate(thresh, None, iterations = 2)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in cnts:

        # Save the coordinates of all found contours
        (x, y, w, h) = cv2.boundingRect(c)

        # If the contour is too small, ignore it, otherwise, there's transient
        # movement
        if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
            transient_movement_flag = True
            
            # Draw a rectangle around big enough movements
            cv2.rectangle(np_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if w > lastW[i]*1.40:
                isFall = True
                
        
            lastW[i] = w
            lastH[i] = h


    # The moment something moves momentarily, reset the persistent
    # movement timer.
    if transient_movement_flag == True:
        movement_persistent_flag = True
        movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE

    # As long as there was a recent transient movement, say a movement
    # was detected    
    if movement_persistent_counter > 0:
        text = "Movement Detected " + str(movement_persistent_counter)
        cv2.putText(np_frame, str(text), (10,35), font, 0.75, (255,255,255), 2, cv2.LINE_AA)
        movement_persistent_counter -= 1
    else:
        if isFall:
                motionless_counter += 1
                text = "Alarm in {}".format(MOTIONLESS_HELP - motionless_counter)
                if motionless_counter >= MOTIONLESS_HELP:
                    isAlarm = True
                    print("Alarm!")
        else:
            text = ""
        if isAlarm:
            text = "SOS EMERGENCY"
            cv2.putText(np_frame, str(text), (10,35), font, 1.25, (0,0,225), 2, cv2.LINE_AA)
        else:
            
            text = text + " No Movement Detected "
            cv2.putText(np_frame, str(text), (10,35), font, 0.75, (255,255,255), 2, cv2.LINE_AA)


    # Output OpenCV results via HDMI
    outframe[0:480,0:640,:] = frame_vga[0:480,0:640,:]
    hdmi_out.writeframe(outframe)
	
    # Display image in the terminal output window - It is commented at the moment but could be used if HDMI cable is not available
	
    %matplotlib inline 
    from matplotlib import pyplot as plt
    import numpy as np
    plt.imshow(np_frame[:,:,[2,1,0]])
    plt.show()
    cv2.waitKey(200)