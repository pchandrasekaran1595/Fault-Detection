import os
import platform
import cv2
import torch
import numpy as np

import utils as u

# ******************************************************************************************************************** #

# Inference Helper
def __help__(frame=None, model=None, fea_extractor=None, show_prob=True, pt1=None, pt2=None):
    disp_frame = frame.copy()

    # Resize + Center Crop (256x256 ---> 224x224)
    frame = u.preprocess(frame, change_color_space=False)

    # Perform Inference on current frame
    with torch.no_grad():
        features = u.normalize(fea_extractor(u.FEA_TRANSFORM(frame).to(u.DEVICE).unsqueeze(dim=0)))
        y_pred = torch.sigmoid(model(features))[0][0].item()

    # Prediction > Upper Bound                 -----> Match
    # Lower Bound <= Prediction <= Upper Bound -----> Possible Match
    # Prediction < Lower Bound                 -----> No Match
    if show_prob:
        if y_pred >= u.upper_bound_confidence:
            cv2.putText(img=disp_frame, text="Match, {:.5f}".format(y_pred), org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.CLI_GREEN, thickness=2)
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                          color=u.CLI_GREEN, thickness=2)

        elif u.lower_bound_confidence <= y_pred <= u.upper_bound_confidence:
            cv2.putText(img=disp_frame, text="Possible Match, {:.5f}".format(y_pred), org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.CLI_ORANGE, thickness=2)
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                          color=u.CLI_ORANGE, thickness=2)

        else:
            cv2.putText(img=disp_frame, text="No Match, {:.5f}".format(y_pred), org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.CLI_RED, thickness=2)
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                          color=u.CLI_RED, thickness=2)
    else:
        if y_pred >= u.upper_bound_confidence:
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                          color=u.CLI_GREEN, thickness=2)
        elif u.lower_bound_confidence <= y_pred <= u.upper_bound_confidence:
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                          color=u.CLI_ORANGE, thickness=2)
        else:
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                          color=u.CLI_RED, thickness=2)
    return disp_frame

# ******************************************************************************************************************** #

# Realtime Inference
def realtime(device_id=None, part_name=None, model=None, save=False, fea_extractor=None, show_prob=False):
    base_path = os.path.join(u.DATASET_PATH, part_name)

    # Read the anchor image
    disp_anchor_image = cv2.imread(os.path.join(os.path.join(base_path, "Positive"), "Snapshot_1.png"), cv2.IMREAD_COLOR)

    # Load the model
    path = os.path.join(os.path.join(base_path, "Checkpoints"), "State.pt")
    model.load_state_dict(torch.load(path, map_location=u.DEVICE)["model_state_dict"])
    model.eval()
    model.to(u.DEVICE)

    # Initialize the capture object
    if platform.system() != "Windows":
        cap = cv2.VideoCapture(device_id)
    else:
        cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, u.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, u.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FPS, u.FPS)

    # Save a video file if flag is set
    if save:
        filename = os.path.join(base_path, "{}.mp4".format(part_name))
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(filename, codec, 30.01, (2*u.camWidth, u.camHeight))

    # Open the file containing reference box coordinates
    file = open(os.path.join(base_path, "Box.txt"), "r")
    data = file.read().split(",")
    file.close()

    countp, countn = len(os.listdir(os.path.join(base_path, "Positive"))), len(os.listdir(os.path.join(base_path, "Negative"))) + 1
    if countn == 0:
        countn = 1
    
    # Read data from capture object
    while cap.isOpened():
        _, frame = cap.read()
        frame = u.clahe_equ(frame)
        disp_frame = __help__(frame=frame, model=model, 
                              fea_extractor=fea_extractor,
                              show_prob=show_prob, pt1=(data[0], data[1]), pt2=(data[2], data[3]))
        
        # ********************************************************************* #

        # Press 'p' if the object detected is a False Negative
        if cv2.waitKey(u.DELAY) == ord("p"):
            print("")
            cv2.imwrite(os.path.join(os.path.join(base_path, "Positive"), "Extra_{}.png".format(countp)), frame)
            print("Captured Snapshot - {} and save to Positive Directory".format(countp))
            countp += 1
        
        # Press 'n' if the object detected is a False Positive
        if cv2.waitKey(u.DELAY) == ord("n"):
            print("")
            cv2.imwrite(os.path.join(os.path.join(base_path, "Negative"), "Extra_{}.png".format(countn)), frame)
            print("Captured Snapshot - {} and save to Negative Directory".format(countn))
            countn += 1
        
        # ********************************************************************* #

        disp_frame = np.hstack((disp_anchor_image, disp_frame))
        if save:
            out.write(disp_frame)
        
        # Display the frame
        cv2.imshow("Feed", disp_frame)

        # Press 'q' to Quit
        if cv2.waitKey(u.DELAY) == ord("q"):
            break

    # Release capture object and destory all windows
    cap.release()
    cv2.destroyAllWindows()

# ******************************************************************************************************************** #

# Inference performed on video file
def video(filename=None, part_name=None, model=None, save=False, fea_extractor=None, show_prob=True):
    base_path = os.path.join(u.DATASET_PATH, part_name)

    # Read the anchor image
    disp_anchor_image = cv2.imread(os.path.join(os.path.join(base_path, "Positive"), "Snapshot_1.png"), cv2.IMREAD_COLOR)

    # Load the model
    path = os.path.join(os.path.join(base_path, "Checkpoints"), "State.pt")
    model.load_state_dict(torch.load(path, map_location=u.DEVICE)["model_state_dict"])
    model.eval()
    model.to(u.DEVICE)

    # Initialize the capture object
    cap = cv2.VideoCapture(os.path.join(os.path.join(base_path, "Video"), "FILENAME.mp4"))

    # Save a video file if flag is set
    if save:
        filename = os.path.join(base_path, "{}.mp4".format(part_name))
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(filename, codec, 30.01, (2*u.camWidth, u.camHeight))
    
    # Open the file containing reference box coordinates
    file = open(os.path.join(base_path, "Box.txt"), "r")
    data = file.read().split(",")
    file.close()

    countp, countn = len(os.listdir(os.path.join(base_path, "Positive"))), len(os.listdir(os.path.join(base_path, "Negative"))) + 1
    if countn == 0:
        countn = 1

    # Read data from capture object
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # Perform Inference
            disp_frame = __help__(frame=frame, model=model, 
                                  fea_extractor=fea_extractor, show_prob=show_prob, 
                                  pt1=(data[0], data[1]), pt2=(data[2], data[3]))
            
            # ********************************************************************* #

            # Press 'p' if the object detected is a False Negative
            if cv2.waitKey(u.DELAY) == ord("p"):
                print("")
                cv2.imwrite(os.path.join(os.path.join(base_path, "Positive"), "Extra_{}.png".format(countp)), frame)
                print("Captured Snapshot - {} and save to Positive Directory".format(countp))
                countp += 1
            
            # Press 'n' if the object detected is a False Positive
            if cv2.waitKey(u.DELAY) == ord("n"):
                print("")
                cv2.imwrite(os.path.join(os.path.join(base_path, "Negative"), "Extra_{}.png".format(countn)), frame)
                print("Captured Snapshot - {} and save to Negative Directory".format(countn))
                countn += 1
        
            # ********************************************************************* #

            disp_frame = np.hstack((disp_anchor_image, disp_frame))
            if save:
                out.write(disp_frame)
            
            # Display the frame
            cv2.imshow("Feed", disp_frame)

            # Press 'q' to Quit
            if cv2.waitKey(u.DELAY) == ord("q"):
                break
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Release capture object and destory all windows
    cap.release()
    cv2.destroyAllWindows()

# ******************************************************************************************************************** #