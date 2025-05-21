# CNN_Waste_Segregation_LJMU_Masters

## Project Overview
This project implements a Convolutional Neural Network (CNN) for classifying waste materials into seven categories. The goal is to create an automated waste classification system that can enhance recycling efficiency and streamline waste management practices through image recognition.

## Business Objective
Improper waste disposal contributes to environmental degradation, increased landfill waste, and inefficient recycling processes. This AI-powered waste classification system addresses these challenges by:
- Automating waste sorting processes
- Reducing operational costs associated with manual sorting
- Improving recycling rates through accurate classification
- Providing a foundation for smart recycling bins and automated waste facilities

## Dataset
The dataset contains approximately 7,000 images categorized into seven waste material types:
- Cardboard
- Food Waste
- Glass
- Metal
- Other
- Paper
- Plastic

Each category contains RGB images of various waste items. All images were originally 256×256 pixels and were resized to 224×224 pixels for processing.

## Key Features
- Multi-class image classification using deep learning
- Data preprocessing pipeline for handling large image datasets
- Implementation of data augmentation techniques
- Class imbalance handling through weighted loss functions
- Comprehensive model evaluation metrics

## Technical Implementation

### Data Preparation
- Image loading and batch processing to manage memory efficiently
- Normalization of pixel values to [0,1] range
- One-hot encoding of categorical labels
- Stratified train-validation split (80-20%)
- Class weighting to address imbalance

### Model Architecture
The CNN model consists of:
- Three convolutional blocks, each with:
  - Convolutional layer (64, 128, 256 filters with 3×3 kernels)
  - Batch normalization
  - ReLU activation
  - Max pooling
  - Dropout (0.25, 0.3, 0.4)
- Flatten layer
- Fully connected layer (512 units)
- Final dense layer with softmax activation (7 units)

### Training Process
- Optimizer: Adam with learning rate of 0.001
- Loss function: Categorical cross-entropy
- Metrics: Accuracy
- Callbacks:
  - Early stopping
  - Learning rate reduction on plateau
  - Model checkpointing

### Data Augmentation
For improving model robustness, the following augmentation techniques were implemented:
- Horizontal flipping
- Random rotation (±10 degrees)
- Brightness adjustments (0.8 to 1.2 factor)
- Contrast adjustments (0.8 to 1.2 factor)

## Results

### Overall Performance
The base model achieved:
- Training accuracy: 99.6%
- Validation accuracy: 68.6%
- Macro-average F1-score: 0.68

The model with data augmentation achieved:
- Training accuracy: 94.3%
- Validation accuracy: 68.0%
- Macro-average F1-score: 0.67

### Per-Class Performance
Performance varied significantly across waste categories:
- Best performing: Cardboard (F1-score: 0.77), Metal (Precision: 0.87)
- Most challenging: Other (Recall: 0.51), Glass (Precision: 0.59)

## Conclusions and Insights
- The model demonstrated reasonable performance considering the challenging nature of waste classification
- Significant overfitting was observed (training-validation accuracy gap of 31%)
- Data augmentation helped reduce overfitting slightly but didn't improve overall performance
- Category-specific improvements were observed with augmentation (Cardboard recall +5%, Paper precision +9%)
- The model is most reliable for Cardboard, Food Waste, Metal, and Plastic classification

## Future Improvements
- Experiment with pre-trained models (ResNet, EfficientNet)
- Implement class-specific augmentation techniques
- Deploy stronger regularization to address overfitting
- Collect more diverse samples for challenging categories
- Adjust confidence thresholds per class for deployment

## Setup and Usage

### Prerequisites
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PIL

### Installation
```bash
pip install numpy==1.26.4 pandas==2.2.2 seaborn==0.13.2 matplotlib==3.10.0 
pip install tensorflow==2.18.0 keras==3.8.0 scikit-learn==1.6.1 Pillow
```

## Acknowledgments
This project was completed as part of Upgrad's deep learning assignment

## Author
Saurabh Tayde
saurabhtayde2810@gmail.com
