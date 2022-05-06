1. This repository is used to provide a code base for PyTorch-based training framework. One can easily develop their own model on top of it.
2. Some dirty codes are still there without debugging. 
3. UNet-based denoising/segmentation examples are provided based on two open-access datasets
   1. Denoising: **[AAPM LowDose CT Challenge dataset](https://www.aapm.org/grandchallenge/lowdosect/)**
   2. Segmentation: **[StructSeg segmentation dataset](https://structseg2019.grand-challenge.org/)**. 

## step by step:
1. customize your own dataset within lib/dataset
2. customize your own specific training workflow within lib/model
3. customize your own architecture within lib/model/module
