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


## Usage

run `Main.ipynb`

Regarding Parameters

```python
parameters = {
    "dataname": dataname,
    "feat_dim": X[train].shape[1],
    "num_labels": y[train].shape[1],
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


## Citing this repository
If you find this code useful in your research, please consider citing us:
 
