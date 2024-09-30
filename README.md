# Introduction

## `AEMLO_train.py`
This script is the main training file for the AEMLO model. It handles the model initialization, training process, parameter tuning, and the final model saving.

## `Generate_Instance.py`
This file is responsible for generating training instances for the AEMLO, including tasks such as data preprocessing, sample generation, and augmentation, providing input data for the model.

## `Main.ipynb`
This is a Jupyter Notebook file used for interactively running the key steps of the AEMLO project. It allows step-by-step code execution, making it easier for debugging and experimentation.

## `Model.py`
This file contains the core model definition for the AEMLO project. It implements the model architecture, forward propagation, and backpropagation for deep learning-related functions.

## `Pre_Function.py`
This file includes preprocessing functions, such as data cleaning, feature extraction, and transformation. These functions prepare the input data for the model.

## `Train_Classifier.py`
This file is a multi-label experimental module that integrates AEMLO-sampled instances with the original training set. It then applies well-known multi-label classifiers such as BR, MLkNN, and RAkEL to evaluate the model's performance. The results are displayed using various metrics, including Macro-level evaluations, providing a comprehensive comparison of the classifiers' effectiveness.

