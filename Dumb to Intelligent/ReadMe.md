- Uses a VGG16 (with Batch Normalization) as the backbone pretrained network

- This is the main application

- Assumes that the user can provide only a single positive sample for a single component. System must be subsequently trained for each new component added, or if and when more samples become available for a pre-existing component

- Does not contain a CLI version