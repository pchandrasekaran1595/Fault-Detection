"""
    Models
"""

import torch
from torchvision import models
from torch import nn, optim
import utils as u

# ******************************************************************************************************************** #

# VGG16 Model; Slice out the final 2 blocks and Average Pool the 512x7x7 features down to 512x2x2 and then Flatten
class FeatureExtractor(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        
        self.model = models.vgg16_bn(pretrained=True, progress=True)
        self.model = nn.Sequential(*[*self.model.children()][:2])
        self.model.add_module("Adaptive Avg Pool", nn.AdaptiveAvgPool2d(output_size=(2, 2)))
        self.model.add_module("Flatten", nn.Flatten())

    def forward(self, x):
        return self.model(x)

# ******************************************************************************************************************** #

"""
    - Siamese Network Architecture (Input Layer --> Embedding Layer --> Similarity Predictor)
    - Expects a pair of inputs during the training phase
"""
class SiameseNetwork(nn.Module):
    def __init__(self, IL=u.FEATURE_VECTOR_LENGTH, embed=None):
        super(SiameseNetwork, self).__init__()

        # Input --> Embedding
        self.embedder = nn.Sequential()
        self.embedder.add_module("BN", nn.BatchNorm1d(num_features=IL, eps=1e-5))
        self.embedder.add_module("FC", nn.Linear(in_features=IL, out_features=embed))
        self.embedder.add_module("AN", nn.ReLU())

        # Embedding -> Similarity Predictor
        self.classifier = nn.Sequential()
        self.classifier.add_module("BN", nn.BatchNorm1d(num_features=embed, eps=1e-5))
        self.classifier.add_module("FC", nn.Linear(in_features=embed, out_features=1))

    # Setup Adam Optimizer
    def getOptimizer(self, lr=1e-3, wd=0):
        return optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    # Setup Dynamic Learn Rate Scheduler
    def getScheduler(self, optimizer=None, patience=5, eps=1e-8):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=patience, eps=eps, verbose=True)

    def forward(self, x1, x2=None):
        if x2 is not None:
            x1 = self.embedder(x1)
            x2 = self.embedder(x2)
            x = self.classifier(torch.abs(x1 - x2))
            return x
        else:
            x = self.classifier(self.embedder(x1))
            return x
        
# ******************************************************************************************************************** #

fea_extractor = FeatureExtractor()
fea_extractor.to(u.DEVICE)
fea_extractor.eval()

# ******************************************************************************************************************** #

# Setup the Siamese Netowrk
def build_siamese_model(embed=None):
    if embed is not None:
        torch.manual_seed(u.SEED)
        model = SiameseNetwork(embed=embed)
    else:
        model = None

    lr = 1e-6
    wd = 1e-6
    batch_size = 512

    return model, batch_size, lr, wd

# ******************************************************************************************************************** #
