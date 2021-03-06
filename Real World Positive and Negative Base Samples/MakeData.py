import os
import re
import cv2
import torch
import imgaug
import numpy as np
import random as r
from imgaug import augmenters
from torch.utils.data import DataLoader as DL

from DatasetTemplates import FEDS
import utils as u

# ******************************************************************************************************************** #

# Augmentation Pipeline
def get_augments(augment_seed=None): 
    imgaug.seed(entropy=augment_seed)
    dataset_augment = augmenters.Sequential([
        augmenters.HorizontalFlip(p=0.25),
        augmenters.VerticalFlip(p=0.25),
        augmenters.SomeOf(5, [
            augmenters.blur.GaussianBlur(sigma=(0, 2), seed=augment_seed),
            augmenters.blur.MedianBlur(k=(1, 3), seed=augment_seed),
            augmenters.size.Crop(percent=(0, 0.10), seed=augment_seed),
            augmenters.geometric.Affine(rotate=(-45, 45), scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), seed=augment_seed, cval=100, mode="symmetric"),
            augmenters.geometric.Rot90(k=(1, 3), seed=augment_seed),
            augmenters.arithmetic.Dropout(p=(0, 0.075), seed=augment_seed),
            augmenters.arithmetic.SaltAndPepper(p=(0, 0.075), seed=augment_seed),
            augmenters.color.MultiplyBrightness(mul=(0.5, 1.5)),
            augmenters.color.MultiplySaturation(mul=(0, 5), seed=augment_seed),
            augmenters.iaa_convolutional.Sharpen(alpha=(0.75, 1), lightness=(0.75, 1.25), seed=augment_seed),
            augmenters.iaa_convolutional.Emboss(alpha=(0.75, 1), strength=(0.75, 1.25), seed=augment_seed),
            augmenters.contrast.CLAHE(seed=augment_seed),
            augmenters.contrast.GammaContrast(gamma=(0.2, 5), seed=augment_seed),
        ])
    ])

    imgaug.seed(entropy=augment_seed)
    roi_augment = augmenters.Sequential([augmenters.imgcorruptlike.GlassBlur(severity=5, seed=augment_seed),])

    return dataset_augment, roi_augment

# ******************************************************************************************************************** #

def make_data(part_name=None, cls="Positive", num_samples=None, batch_size=48, fea_extractor=None):
    """
        part_name     : Part name
        cls           : Class of the image (Either Negative or Positive)
        num_samples   : Number of Samples to be included in the Dataset
        batch_size    : Batch Size used by feature extracting dataloader
        fea_extractor : Feature Extraction Model
    """

    base_path = os.path.join(u.DATASET_PATH, part_name)
    cls_path = os.path.join(base_path, cls)
    if not os.path.exists(cls_path):
        os.makedirs(cls_path)
    f_names = os.listdir(cls_path)

    r.seed(u.SEED)

    # Calculate the number of samples needed for each image in the directory
    num_samples_per_image = int(num_samples/len(f_names))

    # Preallocate memory to hold features for each image in the directory
    mini_features = torch.zeros(num_samples_per_image, u.FEATURE_VECTOR_LENGTH).to(u.DEVICE)
    features = torch.zeros(1, u.FEATURE_VECTOR_LENGTH).to(u.DEVICE)
    for name in f_names:

        # Get the augmentation pipeline
        augment_seed = r.randint(0, 99)
        dataset_augment, _ = get_augments(augment_seed)

        # Read the image
        image = u.preprocess(cv2.imread(os.path.join(cls_path, name), cv2.IMREAD_COLOR))

        # Augment the entire dataset using the dataset_augment pipeline
        images = np.array(dataset_augment(images=[image for _ in range(num_samples_per_image)]))

        # Setup the feature extraction dataloader
        feature_data_setup = FEDS(X=images, transform=u.FEA_TRANSFORM)
        feature_data = DL(feature_data_setup, batch_size=batch_size, shuffle=False)

        # Extract Features
        for i, X in enumerate(feature_data):
            X = X.to(u.DEVICE)
            with torch.no_grad():
                output = fea_extractor(X)
            mini_features[i * batch_size: (i * batch_size) + output.shape[0], :] = output
        
        features = torch.cat((features, mini_features), dim=0)
    
    # Save the normalized Feature Vectors as numpy arrays
    np.save(os.path.join(base_path, "{}_Features.npy".format(cls)), u.normalize(features[1:]).detach().cpu().numpy())

    # Clean up CUDA device
    del output, features, mini_features, fea_extractor
    torch.cuda.empty_cache()

# ******************************************************************************************************************** #
