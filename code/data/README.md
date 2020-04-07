
## Dataloader

- use opencv (`cv2`) to read and process images.

- read from **image** files OR from **.lmdb** for fast IO speed.
    - How to create .lmdb file? Please see [`preprocess`](https://github.com/Maclory/SPSR/tree/master/preprocess).
    
- can downsample images using `matlab bicubic` function. However, the speed is a bit slow. Implemented in [`util.py`](https://github.com/Maclory/SPSR/tree/master/code/data/util.py). More about [`matlab bicubic`](https://github.com/xinntao/BasicSR/wiki/Matlab-bicubic-imresize) function.


## Contents

- `LR_dataset`: only reads LR images in the testing phase where there is no GT image.
- `LRHR_dataset`: reads LR and HR pairs from image folders or lmdb files. If only HR images are provided, downsample the images on-the-fly. Used in training and validation phases.


## How To Prepare Data
Please refer to [Dataset Preparation](https://github.com/Maclory/SPSR/tree/master).
