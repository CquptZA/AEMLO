## Introduction

### `AEMLO_train.py`
This script is the main training file for the AEMLO model. It handles the model initialization, training process, parameter tuning, and the final model saving.

### `Generate_Instance.py`
This file is responsible for generating training instances for the AEMLO, including tasks such as data preprocessing, sample generation, and augmentation, providing input data for the model.

### `Model.py`
This file contains the core model definition for the AEMLO project. It implements the model architecture, forward propagation, and backpropagation for deep learning-related functions.

### `Pre_Function.py`
This file includes preprocessing functions, such as data cleaning, feature extraction, and transformation. These functions prepare the input data for the model.

### `Train_Classifier.py`
This file is a multi-label experimental module that integrates AEMLO-sampled instances with the original training set. It then applies well-known multi-label classifiers such as BR, MLkNN, and RAkEL to evaluate the model's performance. The results are displayed using various metrics, including Macro-level evaluations, providing a comprehensive comparison of the classifiers' effectiveness.

### `Main.ipynb`
This is a Jupyter Notebook file used for interactively running the AEMLO with different base Multi-label Classifier.



## Requirements

Please, install the following packages

    numpy 1.21.6
    
    pytorch 1.11.0
    
    scikit-learn 1.0.2
    
    scikit-multilearn 0.2.0
    
    and python==3.7

About scikit-multilearn reference [scikit-multilearn GitHub Repository](https://github.com/scikit-multilearn/scikit-multilearn).

This package has many errors, such as iterative_train_test_split, you can refer to its issues for resolution.





## Usage

run `Main.ipynb`

Regarding Parameters

```python
parameters = {
    "dataname": dataname,
    "feat_dim": X[train].shape[1],
    "num_labels": y[train].shape[1],

// The core dimension parameters can be understood by referring to the paper
    "latent_dim": params["latent_dim"],
    "fx_h_dim": params["fx_h_dim"],
    "fe_h_dim": params["fe_h_dim"],
    "fd_h_dim": params["fd_h_dim"],

    "X_train": X[train],
    "y_train": y[train],
    "epoch": params["epoch"],
    "batch_size": params["batch_size"],
    "learningrate": params["learningrate"]
}
```

## Citing this repository
If you find this code useful in your research, please consider citing us:

```python
@inproceedings{zhou2024aemlo,
  title={AEMLO: AutoEncoder-Guided Multi-label Oversampling},
  author={Zhou, Ao and Liu, Bin and Wang, Jin and Sun, Kaiwei and Liu, Kelin},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={107--124},
  year={2024}
}
 ```

## Tips
If you have any specific questions, please contact zacqupt@gmail.com
