import os
import cv2
import numpy as np
import torch
from torchvision import transforms, ops
from termcolor import colored
os.system("color")

DATASET_PATH = os.path.join(os.getcwd(), "Datasets")
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)
# DATASET_PATH = os.path.join(os.path.dirname(__file__), "Datasets")

CAM_WIDTH, CAM_HEIGHT, FPS, DELAY = 640, 360, 30, 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ROI_TRANSFORM = transforms.Compose([transforms.ToTensor(), ])
FEA_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIZE = 224
SEED = 0
RELIEF = 25
FEATURE_VECTOR_LENGTH = 2048

# GUI Color Schemes
GUI_ORANGE = (255, 165, 0)
GUI_RED    = (255, 0, 0)
GUI_GREEN  = (0, 255, 0)

# ****************************************** Default CLI Arguments *************************************************** #
embed_layer_size = 2048
num_samples = 1000
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

# Normalize the vector to a min-max of [0, 1]
def normalize(x):
    for i in range(x.shape[0]):
        x[i] = (x[i] - torch.min(x[i])) / (torch.max(x[i]) - torch.min(x[i]))
    return x

# ******************************************************************************************************************** #

# Extract the feature vector from a single image
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