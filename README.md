---
# StrokePrediction_Model
## About

This project
* [Google Colab Tutorial Notebook](https://colab.research.google.com/github/jlee92603/StrokePrediction_Model/blob/main/StrokePredictionModel.ipynb)

## Introduction

---
## Table of Contents
- [Getting Started](#Getting-Started)
    - [Installations](#Installations)
    - [Downloading Dataset](#Downloading-Dataset)
    - [Connecting Drive and GPU](#Connecting-Drive-and-GPU)
    - [Importing Libraries](#Importing-Libraries)
 
---
## Getting Started
This convolutional neural network is coded with the Python's Tensorflow Keras package on Google Colab. 

### Installations
Important libraries to install are:
* pytorch_tabnet

### Downloading Dataset
The brain tumor dataset is acquired from Kaggle [Kaggle Stroke Prediction Tabular Dataset](https://www.kaggle.com/competitions/playground-series-s3e2/overview). This dataset ...
This dataset contains a set of testing data files and training data files. 

### Connecting Drive and GPU
The dataset is downloaded and uploaded on to Google Drive, which is connected to the Colab notebook. Additionally, Google Colab's T4 GPU is connected for faster model fitting. Google Colab's GPU can be connected by changing runtime type. 
```
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# find GPU
from tensorflow import test
device_name = test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
```

### Importing Libraries
The following packages are imported: 
```
import numpy as np
import pandas as pd # data processing
import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
```


