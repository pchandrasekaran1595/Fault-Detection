"""
    Constants and Utility Functions
"""

import os
import cv2
import torch
from torchvision import transforms, ops
from termcolor import colored
os.system("color")

# Self Aware Dataset Directory
DATASET_PATH = os.path.join(os.getcwd(), "Datasets")
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)
# DATASET_PATH = os.path.join(os.path.dirname(__file__), "Datasets")

# Capture object Attributes
CAM_WIDTH, CAM_HEIGHT, FPS, DELAY = 640, 360, 30, 5

# DL Model Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ROI_TRANSFORM = transforms.Compose([transforms.ToTensor(), ])
FEA_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIZE = 224
SEED = 0
RELIEF = 25
FEATURE_VECTOR_LENGTH = 2048

# CLI and GUI Color Schemes
GUI_ORANGE = (255, 165, 0)
GUI_RED    = (255, 0, 0)
GUI_GREEN  = (0, 255, 0)

CLI_ORANGE = (0, 165, 255)
CLI_RED    = (0, 0, 255)
CLI_GREEN  = (0, 255, 0)

# ****************************************** Default CLI Arguments *************************************************** #
embed_layer_size = 2048
num_samples = 2500
epochs = 1000
lower_bound_confidence = 0.95
upper_bound_confidence = 0.99
device_id = 0
early_stopping_step = 50
# ******************************************************************************************************************** #

# LineBreaker
def breaker(num=50, char="*"):
    print(colored("\n" + num*char + "\n", color="magenta"))


# Custom Print Function
def myprint(text, color, on_color=None):
    print(colored(text, color=color, on_color=on_color))


# CLAHE Preprocessing (Cliplimit: 2.0, TileGridSize: (2, 2))
def clahe_equ(image):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
    for i in range(3):
        image[:, :, i] = clahe.apply(image[:, :, i])
    return image


# Center Crop (Resize to 256x256, then center crop the 224x224 region)
def preprocess(image, change_color_space=True):
    if change_color_space:
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    image = cv2.resize(src=image, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    return image[cy - 112:cy + 112, cx - 112:cx + 112, :]

# ******************************************************************************************************************** #

# Function to return the bounding box coordinates of a SINGLE image
# Returns the coordinates relative to the original image size
def get_box_coordinates(model, transform, image):
    """
        model     : Pretrained Deep Learning Detector Model (Pytorch)
        transform : Transform expected to be performed on the input
        image     : Image File
    """
    x1, y1, x2, y2 = None, None, None, None

    h, w, _ = image.shape
    temp_image = image.copy()
    temp_image = preprocess(temp_image, change_color_space=False)

    with torch.no_grad():
        output = model(transform(temp_image).to(DEVICE).unsqueeze(dim=0))[0]
    cnts, scrs = output["boxes"], output["scores"]
    if len(cnts) != 0:
        cnts = ops.clip_boxes_to_image(cnts, (SIZE, SIZE))
        best_index = ops.nms(cnts, scrs, 0.1)[0]
        x1, y1, x2, y2 = int(cnts[best_index][0] * (w / SIZE)), \
                         int(cnts[best_index][1] * (h / SIZE)), \
                         int(cnts[best_index][2] * (w / SIZE)), \
                         int(cnts[best_index][3] * (h / SIZE))
    return x1, y1, x2, y2

# ******************************************************************************************************************** #

# Function to return the bounding box coordinates of a SINGLE image
# Returns the coordinates relative to the resized image (224 x 224)
# This is used in MakeData during the first run of the App to create the negative image
def get_box_coordinates_make_data(model, transform, image):
    """
        model     : Pretrained Deep Learning Detector Model (Pytorch)
        transform : Transform expected to be performed on the input
        image     : Image File
    """

    x1, y1, x2, y2 = None, None, None, None
    temp_image = image.copy()
    with torch.no_grad():
        output = model(transform(temp_image).to(DEVICE).unsqueeze(dim=0))[0]
    cnts, scrs = output["boxes"], output["scores"]
    if len(cnts) != 0:
        cnts = ops.clip_boxes_to_image(cnts, (SIZE, SIZE))
        best_index = ops.nms(cnts, scrs, 0.1)[0]
        x1, y1, x2, y2 = int(cnts[best_index][0]), \
                         int(cnts[best_index][1]), \
                         int(cnts[best_index][2]), \
                         int(cnts[best_index][3])
    return x1, y1, x2, y2

# ******************************************************************************************************************** #

# Function to draw the bounding boxes on an image
def process(image, x1, y1, x2, y2):
    """
        image : Image File
        x1    : Initial X-Coordinate
        y1    : Initial X-Coordinate
        x2    : Final X-Coordinate
        x2    : Final X-Coordinate
    """
    if x1 is None:
        cv2.putText(img=image, text=" --- No Objects Detected ---", org=(50, 50),
                    fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), thickness=2)
    else:
        cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 255, 255), thickness=2)
    return image

# ******************************************************************************************************************** #

# Normalize the vector to a min-max of [0, 1]
def normalize(x):
    for i in range(x.shape[0]):
        x[i] = (x[i] - torch.min(x[i])) / (torch.max(x[i]) - torch.min(x[i]))
    return x

# ******************************************************************************************************************** #

# Function to extract the features from a SINGLE image
def get_single_image_features(model=None, transform=None, image=None):
    """
        model     : Pretrained Deep Learning Feature Extractor Model (Pytorch)
        transform : Transform expected to be performed on the input
        image     : Image File
    """
    with torch.no_grad():
        features = model(transform(image).to(DEVICE).unsqueeze(dim=0))
    return normalize(features).detach().cpu().numpy()

# ******************************************************************************************************************** #
