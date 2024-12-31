# Application of Improved Light-weight Feedback Attention Networks for Multi-Class Segmentation in Urban Scenes

## Abstract

Image segmentation is one of the most important techniques for computer vision. It is not only the foundation for biomedical imaging, but also essential in urban planning and autonomous driving. In this project, we extend the Feedback Attention Network (FANet), originally designed for precise biomedical segmentation, to address the complex demands of urban scenes through the Cityscapes dataset. The original FANet was designed for binary segmentation and applying it to multi-class segmentation is a non-trivial task. To address this challenge, we propose a new competitive layer for inference. This enhancement, coupled with model light-weighting and the incorporation of spatial and channel squeeze and excitation (scSE) modules, not only reduces the computational complexity but also maintains or even surpasses the original model performance. Through rigorous testing and evaluation, the modified FANet demonstrates robust adaptability and strong performance across diverse urban environments. The results highlight the model’s versatility, confirming its extended applicability beyond its original medical domain and its potential as a powerful tool for a broad range of image segmentation tasks.

## Project Overview

This project aims to develop an improved Light-weight Feedback Attention Networks (FANet) for multi-class segmentation in urban scenes. The project involves the following key components:

1. **Data Preparation**: The [Cityscapes dataset](https://www.kaggle.com/datasets/jiaxiyang116/cityscape) is used for training and evaluation. The dataset is preprocessed to extract relevant features and labels for segmentation. We also provide a script to preprocess it. `Test_file_preprocess.py` is used to preprocess the test data.
2. **Model Training**: The FANet model is trained on the preprocessed Cityscapes dataset. We train with different categorized datasets to generate weight files for different classifications. `train.py` is used to train the model.
3. **Model Validation**: The trained model is validated on the Cityscapes dataset to evaluate its performance. `test_Multy_Class.py` is used to test the model for multi_classification, `test.py` is used to test the model for binary_classification.