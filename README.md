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
 - [Explore Data](#Explore-Data)
    - [General Tabular Data](#General-Tabular-Data)
    - [Distribution by Gender](#Distribution-by-Gender)
    - [Kernel Density Estimates](#Kernel-Density-Estimates)
    - [Distribution by Train vs Test](#Distribution-by-Train-vs-Test)
    - [Distribution by Stroke](#Distribution-by-Stroke)
- [Data Preprocessing](#Data-Preprocessing)
    - [Data Encoding](#Data-Encoding)
    - [Feature Engineering](#Feature-Engineering)
    - [Data Split](#Data-Split)
- [Necessary Functions](#Necessary-Functions)
- [Build, Test, Evaluate Models](#Build,-Test,-Evaluate-Models)
    - [Cohort A](#Cohort-A)
    - [Cohort B](#Cohort-A)
    - [Cohort C](#Cohort-A)
    - [Cohort D](#Cohort-A)
    - [Cohort E](#Cohort-A)
    - [Model Comparisons](#Model-Comparisons)
- [Using Tabnet Architecture](#Tabnet-Architecture)
    - [Cohort 0](#Cohort-0)
    - [Cohort 1](#Cohort-1)
    - [Cohort 2](#Cohort-2)
    - [Cohort 3](#Cohort-3)
    - [Tabnet Comparisons](#Tabnet-Comparisons)
- [Conclusions](#Conclusions)


---
## Getting Started
This convolutional neural network is coded with the Python's Tensorflow Keras package on Google Colab. 

### Installations
Important libraries to install are:
* pytorch_tabnet

### Downloading Dataset
The brain tumor dataset is acquired from Kaggle [Kaggle Stroke Prediction Tabular Dataset](https://www.kaggle.com/competitions/playground-series-s3e2/overview). This dataset includes 2 csv files, a training data set where stroke is the binary target and a testing data set where the objective is to predict the probability of positive stroke. The training data csv file consists of tabular data with 10 features for each of the 15304 patients: gender, age, hypertension, heart disease, marriage status, work type, residence type, average glucose level, BMI, and smoking status, and whether the patient has stroke or not. 

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

## Explore Data

```
# get data from drive
train = pd.read_csv(filepath + 'train.csv')
test = pd.read_csv(filepath + 'test.csv')
```

### General Tabular Data

```
# display first 5 patients for training data
train.head()

# get training dataframe information
train.info()

# get statistics for training data
train.describe()
```

<img width="914" alt="Screen Shot 2024-05-27 at 4 51 07 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/fefa3286-09bd-4158-9d67-6302dea6971f">

<img width="334" alt="Screen Shot 2024-05-27 at 4 51 18 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/51f6a00b-ce7b-4c7c-a47a-28da2048b983">

<img width="695" alt="Screen Shot 2024-05-27 at 4 51 27 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/c5fbbe54-3d83-4cda-9c2a-5709442be3b4">

### Distribution by Gender

```
# barplot statistics for male, female, and nonbinary patients
fig = plt.figure(figsize = (15, 5))
plt.subplot(2,2,1)
sns.barplot(x=train['gender'], y=train['age'])

plt.subplot(2,2,2)
sns.barplot(x=train['gender'], y=train['hypertension'])

plt.subplot(2,2,3)
sns.barplot(x=train['gender'], y=train['heart_disease'])

plt.subplot(2,2,4)
sns.countplot(x=train['stroke'])
```



### Kernel Density Estimates

### Distribution by Train vs Test

### Distribution by Stroke

## Data Preprocessing
### Data Encoding
### Feature Engineering
### Data Split
## Necessary Functions
## Build, Test, Evaluate Models
### Cohort A
### Cohort B
### Cohort C
### Cohort D
### Cohort E
### Model Comparisons
## Using Tabnet Architecture
### Cohort 0
### Cohort 1
### Cohort 2
### Cohort 3
### Tabnet Comparisons
## Conclusions
