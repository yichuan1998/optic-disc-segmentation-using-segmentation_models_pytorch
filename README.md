# optic-disc-segmentation-using-segmentation_models_pytorch 🚀
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-brightgreen)](https://python.org)

**One-line Tagline**：This is the repository for retinal iamges optic disc segmentation using segmentation_models_pytorch baesd on IDRID dataset

## Introduction
- ​**Problem Statement**​
  Providing a repository for retinal iamges optic disc segmentation using segmentation_models_pytorch.
  
- ​**Key Features**​  
  ✅ function 1：free to choose backbone and encoder
  
  ✅ function 2：visualize metrics's history
  
  ✅ function 3：inference on single image

- ​**Technology Stack**​
  ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch)

## Getting Started
### Training
Edit the config.py and run the od-seg.py.

### Inference on single image
run the od-seg-infer.py.

## Results
visualize the metrics during trainging
![metrics](training_metrics.png)
visualize the inference results, including mask image and overlay image
![overlay](overlay.png)
![mask](mask.png)

### Prerequisites
```bash
Python 3.9+  
pip install albumentations segmentation_models_pytorch tqdm scikit-learn scikit-image
