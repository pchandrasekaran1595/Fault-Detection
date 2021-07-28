import numpy as np
import torch
from torch.utils.data import Dataset
import utils as u

# ******************************************************************************************************************** #

# Dataset Template used in Feature Extraction
class FEDS(Dataset):
    def __init__(self, X=None, transform=None):
        self.transform = transform
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.transform(self.X[idx])

# ******************************************************************************************************************** #

# Dataset Template used to generate data that can be passed to the Siamese Network
class SiameseDS(Dataset):
    def __init__(self, anchors=None, p_vector=None, n_vector=None):
        self.anchors = anchors
        self.p_vector = p_vector
        self.n_vector = n_vector
        self.fullX = np.zeros((1, 2, u.FEATURE_VECTOR_LENGTH))
        self.fully = np.zeros((1, 1))

        if self.p_vector.shape[0] != self.n_vector.shape[0]:
            for anchor in self.anchors:
                min_val = min(self.p_vector.shape[0], self.n_vector.shape[0])
                self.anchor = np.array([anchor for _ in range(min_val)])

                self.pX = np.concatenate((self.anchor, np.expand_dims(self.p_vector[:min_val], axis=1)), axis=1)
                self.nX = np.concatenate((self.anchor, np.expand_dims(self.n_vector[:min_val], axis=1)), axis=1)
                self.py = np.ones((self.pX.shape[0], 1))
                self.ny = np.zeros((self.nX.shape[0], 1))

                self.X = np.concatenate((self.pX, self.nX), axis=0)
                self.y = np.concatenate((self.py, self.ny), axis=0)

                self.fullX = np.concatenate((self.fullX, self.X), axis=0)
                self.fully = np.concatenate((self.fully, self.y), axis=0)

            self.fullX = self.fullX[1:]
            self.fully = self.fully[1:]
        else:
            for anchor in self.anchors:
                self.anchor = np.array([anchor for _ in range(self.p_vector.shape[0])])

                self.pX = np.concatenate((self.anchor, np.expand_dims(self.p_vector, axis=1)), axis=1)
                self.nX = np.concatenate((self.anchor, np.expand_dims(self.n_vector, axis=1)), axis=1)
                self.py = np.ones((self.pX.shape[0], 1))
                self.ny = np.zeros((self.nX.shape[0], 1))

                self.X = np.concatenate((self.pX, self.nX), axis=0)
                self.y = np.concatenate((self.py, self.ny), axis=0)

                self.fullX = np.concatenate((self.fullX, self.X), axis=0)
                self.fully = np.concatenate((self.fully, self.y), axis=0)
                
            self.fullX = self.fullX[1:]
            self.fully = self.fully[1:]

            print(self.fullX.shape)

    def __len__(self):
        return self.fullX.shape[0]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.fullX[idx]), torch.FloatTensor(self.fully[idx])

# ******************************************************************************************************************** #
