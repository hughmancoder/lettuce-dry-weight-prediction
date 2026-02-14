# Technical Report

Note: Legacy 

## Resources

[GitHub Repo](https://github.com/hughmancoder/lettuce-dry-weight-prediction)

## Problem

Goal: Predict the dry shoot weight (g/plant) of lettuce from RGB and depth images.

Metric: The model should minimise the Mean Absolute Error (MAE) between the predicted and actual [dry shoot weights](https://www.wisdomlib.org/concept/dry-shoot-weight#:~:text=The%20concept%20of%20Dry%20shoot%20weight%20in%20scientific%20sources&text=Dry%20shoot%20weight%20is%20the%20measured%20mass%20of%20a%20plant's,effectiveness%20of%20different%20growth%20strategies.&text=(1)%20Dry%20shoot%20weight%20refers,increased%20with%20bio%2Dstimulant%20treatments.) 

Further information about the context can be found [here](documentation/Notes.md)

## Preprocessing

The [preprocessing pipeline](src/preprocessing.py) conforms to the following specifications:

- Crops RGB and depth images to a standard region: A crop scale of 0.7 was found to be appropriate

- Normalizes depth values to [0,1]. (Depth image is normalised based on 16-bit depth range (2^16 - 1 = 65535)
RGB pixel values originally 0-255 are normalised to the range [0, 1] within the src/dataset.py __getitem__ method.

- Saves processed images into structured directories ()

- Supports both full dataset and train/test subset processing.

An image size of 224 was chosen as a standard input resolution for many well-known image classification models and is useful for transfer learning

## Model designs, and results

### Residual nets

- ResNet was the suggested baseline model primarily because it solves the core problem of training very deep neural networks and is effective for Transfer Learning in computer vision.
- Given the small size of the dataset N = 232 training samples, this approach is highly useful as features from existing weights can be reused and training a complex model from scratch could lead to  overfitting


### [RGBDResNet](src/models/RGBDResNet.py)

- Pre-trained Weights: ResNet models (ResNet-18, ResNet-34) come pre-trained on the massive ImageNet dataset
- Training parameters: these were chosen as a sensible baseline, we will later experiment with others.
- L1Loss optimises for mean absolute error target

**Modifications**

- The model is a 4-channel [ResNet_18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) model with image channels using 3 channels and depth being the 4th channel.
- The weights for the  4th  channel were initialised by taking the mean of the existing 3 RGB pre-trained weights. This ensures the new channel starts with sensible weights, preserving the benefit of pre-training.
- The final classification layer was replaced with a new regression head consisting of an nn.Linear layer, a ReLU activation, a Dropout layer, and a final nn.Linear layer outputting to output a single Dry Shoot Weight prediction
  
```
Validation Split = 0.2
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.0001
DROPOUT = 0.3
```

**Benchmarks**

Train MAE: 0.6770 | Val MAE: 0.6581

