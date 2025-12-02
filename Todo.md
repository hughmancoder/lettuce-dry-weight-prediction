# TODO

Some ideas
- Retrain ResNet using different hyperparameters
- Try different backbones: ResNet-34, ResNet-50.
- Data augmentation (Interface an augmentation pipeline)
  - Rotation/Flips
  - Shifting lettuce center
  - Colour augmentation
- Try a Multi-Task Learning (MTL) approach, to combine auxiliary features (only useful for training) (loss function becomes a weighted sum of primary task and auxiliary task)
  
## IRHAS
Ideas
- 3D reconstruction for depth map
- TABpfn


## Fixes
- Update bit depth format (Depth format: 16-bit PNG (1 channel, values 0-28,535))