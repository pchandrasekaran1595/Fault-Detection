import os
import cv2
import platform
import shutil
import numpy as np

import utils as u

# ******************************************************************************************************************** #

def capture_snapshot(device_id=0, part_name=None, roi_extractor=None):
    path = os.path.join(os.path.join(u.DATASET_PATH, part_name), "Positive")

    if not os.path.exists(path):
        os.makedirs(path)
        file = open(os.path.join(os.path.join(u.DATASET_PATH, part_name), "Box.txt"), "w")
    else:
        shutil.rmtree(os.path.join(u.DATASET_PATH, part_name))
        os.makedirs(path)
        file = open(os.path.join(os.path.join(u.DATASET_PATH, part_name), "Box.txt"), "w")

    # Initialize the capture object
    if platform.system() != "Windows":
        cap = cv2.VideoCapture(device_id)
    else:
        cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, u.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, u.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FPS, u.FPS)
    
    count = 1

    # Read data from capture object
    while cap.isOpened():
        _, frame = cap.read()
        frame = u.clahe_equ(frame)
        d_frame = frame.copy()
        
        x1, y1, x2, y2 = u.get_box_coordinates(roi_extractor, u.ROI_TRANSFORM, frame)
        d_frame = u.process(d_frame, x1, y1, x2, y2)
        d_frame = np.hstack((frame, d_frame))

        # Display the frame
        cv2.imshow("Feed", d_frame)

        # Press 'c' to Capture frame
        if cv2.waitKey(u.DELAY) == ord("c"):
            print("")
            cv2.imwrite(os.path.join(path, "Snapshot_{}.png".format(count)), frame)
            file.write(repr(x1) + "," + repr(y1) + "," +repr(x2) + "," + repr(y2))
            print("Captured Snapshot - {}".format(count))
            count += 1

        # Press 'q' to Quit
        if cv2.waitKey(u.DELAY) == ord("q"):
            break
    
    file.close()

    # Release the capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

# ******************************************************************************************************************** #
