### Working

- Contains a Siamese Architecture Implementation of the Application.

- Feature Extraction is performed as part of the data preprocessing pipeline using one of many pretrained models. End model used for training is merely a Multi-Layered Perceptron. This method is used so that the illusion of Siamese Architecture can be given without needing to run the data through the entire Deep Network every epoch.

- Architecture of MLP used is [input_layer --> embed_layer --> output]
    1. input_layer - Size of the feature vector obtained from the pretrained network.
    2. embed_layer - New size of the feature embeddings. Can be provided by use via the command line.
    3. output - Single neuron which after passing through a sigmoid gives a percent prediction.

- Operations taking place during an application run:
    1. Capture of an image from video a feed at the discretion of the user. An Object Detector is used to extract the approximate bounding box of the component within the frame. This bounding box is used in the Realtime Application to set a predefined area within the frame where the object can be placed. (Bounding Box Detection is not applicable for Real World Positive and Negative Base Samples)
    2. Creation of the Feature Vector Dataset.
        1. Define a pretrained model to use and cut off the final classification layer. (FeatureExtractors in Templates/Pytorch/VisionModels.py)
        2. Artificially increase dataset size by augmentation. 
        3. Apply necessary transforms and pass images through the network to obtain features. Concatenate and save.
    3. Load features saved in Step 2 and pass it through a Dataset creation pipeline making the data suitable for usage with a Siamese Network. Since pytorch is being used, also build the dataloaders.
    4. Train the model. At present, only limited hyperparameters are present and can be changed only from the source. However, end user can set:
        1. Number of Epochs (Default: 1000)
        2. Number of Samples (Default: 1000)
        3. Size of New Embeddings (Default: 2048)
        4. Lower Confidence Bound (Default: 0.95)
        5. Upper Confidence Bound (Default: 0.99)
        6. Device ID of capture Device (Default: 0)
        7. Early Stopping Epoch (Default: 50)

- Realtime Application of the designed network.
    1. Extract features of every frame.
    2. Pass these features into the trained MLP to obtain a prediction.
    3. Based on a predefined confidence, classify the image as Faulty, Possible match or Defective.
    4. If it is a False Positive or False Negative, press the appropriate button to add image captured from feed into the correct directory.


### Notes:
- Cannot detect minute changes. (Issue ..??)
- Changing background affects the system, unless image with that bakcground is included in the dataset.
- Retraining may cause upper boud confidence to be reduced, but the system is able to reliably split into Faulty or Defective.
- To install the development version of Pytorch, use:
`pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html`
