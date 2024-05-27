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

<br>

<img width="312" alt="Screen Shot 2024-05-27 at 6 28 11 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/48e0ce35-02f4-487a-951d-94bc4038f41f">

<br>

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

<img width="687" alt="Screen Shot 2024-05-27 at 5 57 38 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/894bfe2f-45c8-4d1e-b0b5-50b24adb031e">

### Kernel Density Estimates

```
# get numerial features of data (id, age, avg glucose level, bmi)
num_cols = test.select_dtypes(include=np.number).columns.tolist()
num_cols.remove('hypertension')
num_cols.remove('heart_disease')

# display distribution of numerical features for training and testing datasets
fig = plt.figure(figsize = (15, 5))

# for each numerical feature
for i, col in enumerate(num_cols):
  if i==0: # patient id
    continue

  plt.subplot(1,3,i)
  plt.title(col)

  # plot distribution for each feature
  a = sns.kdeplot(train[col], color='#9c2f3b',
                  fill=True, label='train',
                  alpha=0.6, edgecolor='black')
  sns.kdeplot(test[col], color='#72bfd6',
              fill=True, label='test',
              alpha=0.6, edgecolor='black')
  plt.xticks()
  plt.yticks([])

  # hide spines
  for s in ['right', 'top', 'left', 'bottom']:
      a.spines[s].set_visible(False)
  fig.tight_layout(pad=3)

fig.legend(['train ',  'test'], bbox_to_anchor=(0.65, 1.07),
           ncol=3, borderpad=0.5, frameon=True, fontsize=11, title_fontsize=12)
plt.figtext(0.5, 1.1, 'Numerical features distribution',
            fontweight='bold',size=22, color='#444444', ha='center')
plt.show()
```

<img width="671" alt="Screen Shot 2024-05-27 at 5 58 17 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/5c062b0a-fc24-4985-8e67-f1914cae1f13">

### Distribution by Train vs Test

```
# make copy of testing and training datasets for further data analysis
train1 = train.copy()
test1 = test.copy()

# get categorical features
cat_cols = list(set(test.columns) - set(num_cols))

# concatenate the 2 datasets
train1['data'] = 'train'
test1['data'] = 'test'
df = pd.concat([train1, test1])

l = {'train': len(train1),  'test': len(test1)} # length of train and test data

# plot distribution for each feature for testing and training data sets
# grey: training data, red: testing data

# function to change width of patch
def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value
        patch.set_width(new_value)
        patch.set_x(patch.get_x() + diff * .5)

fig = plt.figure(figsize=(15, 15))

# for each categorical feature
for i, col in enumerate(cat_cols):

    # regroups the data by type of 'data' (train vs test) and 'col'
    df_plot = df.groupby(['data', col], as_index=False) \
                  ['age'].count().rename({'age': 'count'}, axis=1).reset_index()

    # retrieves and assigns length of data
    df_plot['len_data'] = df_plot['data'].apply(lambda x: l[x])

    # calculate percentages of total length of data
    df_plot['count'] = round(df_plot['count'] / df_plot['len_data'] * 100, 2)

    # plot barplot
    plt.subplot(4,2,i+1)
    plt.title(col)
    a = sns.barplot(data=df_plot, x=col, y='count', hue='data',
                    palette=['#d1d1d1', '#9c2f3b'],
                    linestyle="-", linewidth=1, edgecolor="black",
                    saturation=0.9)
    plt.ylabel('')
    plt.xlabel('')
    plt.xticks()
    a.set_yticks([0, 100])
    plt.yticks([])

    # label each bar by percent
    for p in a.patches:
        height = p.get_height()
        if (height>0): # if value in barplot exists, add length percentage on top
          a.annotate(f'{height:g}%', (p.get_x() + p.get_width() / 2, p.get_height()+2),
                   ha='center', va='center', size=8,xytext=(0, 5),
                   textcoords='offset points')

    # hide spines
    for s in ['right', 'top', 'left']:
        a.spines[s].set_visible(False)

    # hide legend and change patch width
    a.legend().set_visible(False)
    change_width(a, 0.35)

fig.tight_layout(pad=6)
plt.figtext(0.5, 0.98, 'Categorical features distribution',
            fontweight='bold',size=22, color='#444444', ha='center')
plt.show()
```

<img width="681" alt="Screen Shot 2024-05-27 at 6 24 11 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/64ad23dd-ebde-4d21-b0a2-f72403e0baea">

<br>

<img width="674" alt="Screen Shot 2024-05-27 at 5 59 39 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/d19e5b95-8254-4b37-91c6-f5af13d8249d">

### Distribution by Stroke

```
fig = plt.figure(figsize = (15, 5))

# for each numerical feature
for i, col in enumerate(num_cols):
  if i==0: # skip patient id
    continue
  plt.subplot(1,3,i)
  plt.title(col)

  # plot KDE for training data based on stroke
  a = sns.kdeplot(train1[train1['stroke']==0][col],
                  color='#72bfd6', label='No stroke',
                  fill=True, alpha=0.8, edgecolor='black')
  sns.kdeplot(train1[train1['stroke']==1][col], color='#9c2f3b',
              fill=True, label='Stroke',
              alpha=0.8, edgecolor='black')
  plt.ylabel('')
  plt.xlabel('')
  plt.xticks()
  plt.yticks([])

  # hide spines
  for s in ['right', 'top', 'left', 'bottom']:
      a.spines[s].set_visible(False)

fig.tight_layout(pad=3)
fig.legend(['No stroke', 'Stroke'], bbox_to_anchor=(0.58, 1.07),
           ncol=2, borderpad=0.5, frameon=True, fontsize=11, title_fontsize=12)
plt.figtext(0.5, 1.1, 'Distribution of stroke by numerical features',
            fontweight='bold', size=22, color='#444444', ha='center')
plt.show()

print()
print()

fig = plt.figure(figsize=(15, 15))
for i, col in enumerate(cat_cols):

    # regroups the data by 'stroke' and 'col'
    df_plot = df.groupby(['stroke', col], as_index=False) \
                  ['age'].count().rename({'age': 'count'}, axis=1)

    # calculate percentages of total length of data
    df_plot['count'] = round(df_plot['count'] / len(train) * 100, 2)

    # plot barplot
    plt.subplot(4,2,i+1)
    plt.title(col)
    a = sns.barplot(data=df_plot, x=col, y='count', hue='stroke',
                    palette=['#72bfd6', '#9c2f3b'],
                    linestyle="-", linewidth=1, edgecolor="black",
                    saturation=0.9)
    plt.ylabel('')
    plt.xlabel('')
    plt.xticks()
    a.set_yticks([0, 50])
    plt.yticks([])

    # assign percentages to each bar
    for p in a.patches:
        height = p.get_height()
        if (height>0):
          a.annotate(f'{height:g}%', (p.get_x() + p.get_width() / 2, p.get_height()+1),
                   ha='center', va='center', size=12, xytext=(0, 5),
                   textcoords='offset points')

    # hide spines, legends, change width
    for s in ['right', 'top', 'left']:
        a.spines[s].set_visible(False)
    a.legend().set_visible(False)
    change_width(a, 0.35)

fig.tight_layout(pad=6)
plt.figtext(0.5, 0.98, 'Distribution of stroke by categorical features',
            fontweight='bold',size=22, color='#444444', ha='center')
plt.show()
```

<img width="715" alt="Screen Shot 2024-05-27 at 6 00 19 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/79ce2238-f342-4854-a285-1a1e47bd3bae">

<br>

<img width="715" alt="Screen Shot 2024-05-27 at 6 00 49 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/ccd3066e-f014-44d2-b591-d5b03a879381">

## Data Preprocessing

```
# drop id number from data because id number does not impact prediction
train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)

# for each feature, return the number of values and the types of unique values
def show_values(df):
    all_cols = df.columns
    feature_name = []
    num_unique_val = []
    name_unique_val = []

    for col in all_cols:
        feature_name.append(col)
        num_unique_val.append(df[col].nunique()) # nunique counts number of unique entries
        name_unique_val.append(df[col].unique())
    return pd.DataFrame({'Feature Name' : feature_name, 'Number of Unique Value': num_unique_val, 'Name of Unique Values': name_unique_val})

show_values(train)
```

<img width="355" alt="Screen Shot 2024-05-27 at 6 01 36 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/92bd98de-b0c8-48ca-9f8d-d38f2979dcae">

### Data Encoding

```
# one hot encoding for gender
enc = OneHotEncoder()
enc_data1 = pd.DataFrame(enc.fit_transform(train[['gender']]).toarray())
enc_data2 = pd.DataFrame(enc.fit_transform(test[['gender']]).toarray())

# join encoded data to train/test
train = train.join(enc_data1[0])
train = train.join(enc_data1[1])
test = test.join(enc_data2[0])
test = test.join(enc_data2[1])

# rename the columns; 0 is female, 1 is male
train = train.rename(columns={0:"female",1:"male"})
test = test.rename(columns={0:"female",1:"male"})

# drop gender
train.drop('gender',axis=1,inplace=True)
test.drop('gender',axis=1,inplace=True)

# replace categorial data into numbers

# married status is 1 if yes, 0 if no
train['ever_married'].replace(to_replace='Yes', value=1, inplace=True)
train['ever_married'].replace(to_replace='No', value=0, inplace=True)
test['ever_married'].replace(to_replace='Yes', value=1, inplace=True)
test['ever_married'].replace(to_replace='No', value=0, inplace=True)

# residence type is 0 if urban, 1 if rural
train['Residence_type'].replace(to_replace='Urban', value=0, inplace=True)
train['Residence_type'].replace(to_replace='Rural', value=1, inplace=True)
test['Residence_type'].replace(to_replace='Urban', value=0, inplace=True)
test['Residence_type'].replace(to_replace='Rural', value=1, inplace=True)

# scaling numerical; standardize the data for age, glucose levels, and bmi
numeric_cols = ['age', 'avg_glucose_level','bmi']
for col in numeric_cols:
    train[col] = StandardScaler().fit_transform(train[[col]])
    test[col] = StandardScaler().fit_transform(test[[col]])

# one hot encoding for smoking status and work type
enc_data3 = pd.DataFrame(enc.fit_transform(train[['smoking_status', 'work_type']]).toarray())
enc_data4 = pd.DataFrame(enc.fit_transform(test[['smoking_status', 'work_type']]).toarray())

# add new encoded columns/data to train
train = train.join(enc_data3[1])
train = train.join(enc_data3[2])
train = train.join(enc_data3[3])

train = train.join(enc_data3[4])
train = train.join(enc_data3[5])
train = train.join(enc_data3[6])
train = train.join(enc_data3[7])

test = test.join(enc_data4[1])
test = test.join(enc_data4[2])
test = test.join(enc_data4[3])

test = test.join(enc_data4[4])
test = test.join(enc_data4[5])
test = test.join(enc_data4[6])
test = test.join(enc_data4[7])

# rename the columns
# 0: unknown smoking status, 8: children
train = train.rename(columns={1:"formerly_smoked",2:"never_smoked",
                              3:"smokes", 4:"govt_job", 5:"never_worked",
                              6:"private_job", 7:"self_employed"})
test = test.rename(columns={1:"formerly_smoked",2:"never_smoked",
                              3:"smokes", 4:"govt_job", 5:"never_worked",
                              6:"private_job", 7:"self_employed"})

# drop smoking status and work type (drop orig, replace with encoded)
train.drop('smoking_status',axis=1,inplace=True)
train.drop('work_type',axis=1,inplace=True)
test.drop('smoking_status',axis=1,inplace=True)
test.drop('work_type',axis=1,inplace=True)
```

```
train
```

<img width="891" alt="Screen Shot 2024-05-27 at 6 03 25 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/9d8516e8-7ab4-457e-b77b-f49f8ef0b26b">

```
test
```

<img width="864" alt="Screen Shot 2024-05-27 at 6 03 44 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/9a132b02-3c31-4dc8-adf2-ae82c607676a">

### Feature Engineering

```
# create copies of training data for feature engineering
# add respective feature engineered data sets to the copies

# data for epoch A
trainA = train.copy()

# data for epoch B: add age*bmi
trainB = train.copy()
trainB["age*bmi"] = trainB["age"] * trainB["bmi"]

# data for epoch C: add age*avg_glucose_level
trainC = train.copy()
trainC["age*glucose"] = trainC["age"] * trainC["avg_glucose_level"]

# data for epoch D: add bmi*avg_glucose_level
trainD = train.copy()
trainD["bmi*glucose"] = trainD["bmi"] * trainD["avg_glucose_level"]

# data for epoch E: add age*bmi, age*avg_glucose_level, bmi*avg_glucose_level
trainE = train.copy()
trainE["age*bmi"] = trainE["age"] * trainE["bmi"]
trainE["age*glucose"] = trainE["age"] * trainE["avg_glucose_level"]
trainE["bmi*glucose"] = trainE["bmi"] * trainE["avg_glucose_level"]
```

### Data Split

```
# split training, validation, testing data
def split_data(train):
  # save stroke values into y, drop stroke from x
  y = train['stroke']
  x = train.drop('stroke', axis=1)

  # split training and valid/test data
  x_train, X, y_train, Y = train_test_split(x,y, test_size=0.5, random_state=0, shuffle=True)

  # split validation and test data
  x_valid, x_test, y_valid, y_test = train_test_split(x,y, test_size=0.5, random_state=0, shuffle=True)

  return x_train, y_train, x_valid, y_valid, x_test, y_test
```

## Necessary Functions

```
# recall metrics for TabNet classifier
class myRecall(Metric):
    def __init__(self):
        self._name = "recall"
        self._maximize = True

    def __call__(self, y_true, y_score):

        np.array(y_true)
        y_score_np = np.array(y_score[:, 1])

        # make score binary (2 classes)
        y_score_np[y_score_np < 0.5] = 0
        y_score_np[y_score_np >= 0.5] = 1

        # get recall score
        rs = recall_score(y_true, y_score_np)
        return rs

# function to compile tabnet model
# inputs: training data, training labels, validation data, validation labels,
# optimizer type step size, gamma, evaluation metrics, patience, batch size
# output: fitted tabnet model

def compile_TabNet(x_train, y_train, x_test, y_test, optimizer=torch.optim.Adam,
                   step_size=10, gamma=0.9, eval_met='accuracy',
                   patience=60, batch_size=512):

  # define model
  tbnt = TabNetClassifier(optimizer_fn=optimizer,
                          scheduler_params={"step_size":step_size, "gamma":gamma},
                          scheduler_fn=torch.optim.lr_scheduler.StepLR,)

  # fit the model
  tbnt.fit(
      x_train.values,y_train.values,
      eval_set=[(x_train.values, y_train.values), (x_test.values, y_test.values)],
      eval_name=['train', 'test'],
      eval_metric=['auc', eval_met],
      max_epochs=200, patience=60,
      batch_size=batch_size, virtual_batch_size=512,
      num_workers=0,
      weights=1,
      drop_last=False
  )

  return tbnt

# print evaluation scores for each type of classifier
# inputs: testing data, testing labels, evaluation metric name, eval metric
def print_score(clf, x_test, y_test, eval_name='Accuracy', metric=accuracy_score):
  pred = clf.predict(x_test)
  clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
  print("================================================")
  print(eval_name + f"Score: {metric(y_test, pred) * 100:.2f}%")
  print("_______________________________________________")
  print(f"CLASSIFICATION REPORT:\n{clf_report}")
  print("_______________________________________________")
  print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

score=[]

# function to compare logistic regression, random forest, xgb accuracy
def compare_scores(x_test, y_test, rf, xgb, lgr, metric=accuracy_score):

  # random forest score
  rf_test_score = round(metric(y_test, rf.predict(x_test)) * 100,2)
  rf_score = cross_val_score(estimator = rf, X = x_train, y = y_train, cv = 10)
  rf_train_score=round(rf_score.mean()*100,2)

  # xgboost score
  xgb_test_score = round(metric(y_test, xgb.predict(x_test)) * 100,2)
  xgb_score = cross_val_score(estimator = xgb, X = x_train, y = y_train, cv = 10)
  xgb_train_score = round(xgb_score.mean()*100,2)

  # logistic regression score
  log_test_score = round(metric(y_test, lgr.predict(x_test)) * 100,2)
  log_score = cross_val_score(estimator = lgr, X = x_train, y = y_train, cv = 10)
  log_train_score=round(log_score.mean()*100,2)

  test_scores = [rf_test_score, xgb_test_score, log_test_score]
  return test_scores

# function to build, test, and compare different classifier models
def build_test_models(x_train, y_train, x_valid, y_valid, x_test, y_test, metric=accuracy_score, optimizer=torch.optim.Adam,
                   step_size=10, gamma=0.9, eval_met='accuracy', eval_name = "Accuracy",
                   patience=60, batch_size=512):

  # build and compile tabnet model
  TabNetModel = compile_TabNet(x_train, y_train, x_valid, y_valid,
                               optimizer=optimizer, step_size=step_size,
                               gamma=gamma, eval_met=eval_met,
                               patience=patience, batch_size=batch_size)

  # test tabnet model to predict values
  pred=TabNetModel.predict(x_test.values)

  # print tabnet model report and score
  report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
  score = metric(y_test, pred)
  print("\n\nTest Result for TabNet Classifier: ")
  print("================================================")
  print(eval_name + f"Score: {score * 100:.2f}%")
  print("_______________________________________________")
  print(f"CLASSIFICATION REPORT:\n{report}")
  print("_______________________________________________")
  print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

  # random forest classifier
  rf= RandomForestClassifier() # model
  rf.fit(x_train, y_train) # fit
  print("\n\nTest Result for Random Forest Classifier: ") # print score
  print_score(rf, x_test, y_test, eval_name=eval_name, metric=metric)

  # XGB classifier
  xgb= XGBClassifier() # model
  xgb.fit(x_train, y_train) # fit
  print("\nTest Result for XGB Classifier: ") # print score
  print_score(xgb, x_test, y_test, eval_name=eval_name, metric=metric)

  # logistic regression classifier
  lgr = LogisticRegression() # model
  lgr.fit(x_train, y_train) # fit
  print("\nTest Result for Logistic Regression Classifier: ") # print score
  print_score(lgr, x_test, y_test, eval_name=eval_name, metric=metric)

  # compare the classifiers
  test_scores = compare_scores(x_test, y_test, rf, xgb, lgr, metric=metric)

  # add tabnet scores
  test_scores.append(score*100)

  # scores for each model as data frame
  results_df = pd.DataFrame(data=[['Random Forrest',test_scores[0]],
                                  ['XGB',test_scores[1]],
                                  ["Logistic Regression", test_scores[2]],
                                  ['TabNet', test_scores[3]]],
                            columns=['Model', 'Testing '+ eval_name +' %'])
  results_df.index += 1

  # keep models for later feature importance analysis
  models = [TabNetModel, rf, xgb, lgr]

  print(results_df)

  return test_scores, models

# plot feature importances
def feature_importances(models, labels, x, y):

  # tabnet architecture feature importance
  ft = pd.DataFrame()
  ft["feature"] = labels
  ft["importance"] = models[0].feature_importances_

  # tabnet sort features
  ft.sort_values(
      by = "importance",
      ascending = True,
      inplace = True
  )
  tbntFT = pd.Series(list(ft["importance"]),list(ft["feature"]))

  # random forest feature importance
  rfFI = pd.Series(models[1].feature_importances_, index=labels)

  # xgboost feature importance
  xgbFitted = models[2].fit(x, y)

  # xgb scale data
  xgbData =[]
  sum_ = sum(xgbFitted.get_booster().get_fscore().values())
  for i in xgbFitted.get_booster().get_fscore().values():
    xgbData.append(i/sum_)
  xgbFI = pd.Series(xgbData, index=xgbFitted.get_booster().get_fscore().keys())

  # logistic regression feature importance
  logFitted = models[3].fit(x, y)
  logFI = pd.Series(abs(logFitted.coef_[0]), index=labels)

  # plot all feature importance plots
  fig, ax = plt.subplots(2,2)
  plt.title("Feature Importances")
  tbntFT.nlargest(20).plot(ax=ax[0,0],kind='barh')
  ax[0,0].set_title("Tabnet")
  rfFI.nlargest(20).plot(ax=ax[0,1],kind='barh')
  ax[0,1].set_title("Random Forest")
  xgbFI.nlargest(20).plot(ax=ax[1,0],kind='barh')
  ax[1,0].set_title("XGBoost")
  logFI.nlargest(20).plot(ax=ax[1,1],kind='barh')
  ax[1,1].set_title("Logistic Regression")

  fig.set_figheight(7)
  fig.set_figwidth(10)
  fig.tight_layout()

  fig.show()
```

## Build, Test, Evaluate Models
### Cohort A

```
# split train, valid, and test data
x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(trainA)

# build and test model for each type
test_scores_A, Amodels = build_test_models(x_train, y_train, x_valid, y_valid, x_test, y_test)

# visualize feature importances for each model
feature_importances(Amodels, x_train.columns, x_train, y_train)
```

<img width="314" alt="Screen Shot 2024-05-27 at 6 09 24 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/29d257c1-d213-4d55-a41b-3430fec7b918">

<br>

<img width="314" alt="Screen Shot 2024-05-27 at 6 10 04 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/e324adec-2672-4560-8d6f-e9f9997c14ab">

<br>

<img width="498" alt="Screen Shot 2024-05-27 at 6 06 48 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/2edc2d1d-3057-4c2e-a5c7-fd22c3c29dcc">

### Cohort B

```
# split train, valid, and test data
x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(trainB)

# build and test model for each type
test_scores_B, Bmodels = build_test_models(x_train, y_train, x_valid, y_valid, x_test, y_test)

# visualize feature importances for each model
feature_importances(Bmodels, x_train.columns, x_train, y_train)
```

<img width="314" alt="Screen Shot 2024-05-27 at 6 10 25 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/f99e1a44-a3f0-4f83-bf1d-33d6510c5d00">

<br>

<img width="314" alt="Screen Shot 2024-05-27 at 6 10 42 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/beb1838e-8807-4926-95d1-c1006ddfe7ea">

<br>

<img width="504" alt="Screen Shot 2024-05-27 at 6 08 31 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/b2bc2a40-02fd-4d06-90d4-d685822f7e65">

### Cohort C

```
# split train, valid, and test data
x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(trainC)

# build and test model for each type
test_scores_C, Cmodels = build_test_models(x_train, y_train, x_valid, y_valid, x_test, y_test)

# visualize feature importances for each model
feature_importances(Cmodels, x_train.columns, x_train, y_train)
```

<img width="314" alt="Screen Shot 2024-05-27 at 6 11 30 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/a1c60f20-1446-4775-bab0-0010631a6b11">

<br>

<img width="314" alt="Screen Shot 2024-05-27 at 6 11 50 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/b52c0238-8c19-45ed-82f5-0be5f972e328">

<br>

<img width="314" alt="Screen Shot 2024-05-27 at 6 12 20 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/3675c0e2-13b4-4f1e-bb72-40c2094223fe">

<br>

<img width="501" alt="Screen Shot 2024-05-27 at 6 08 49 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/ef489cce-9c37-4274-82f9-93c4486cb36c">

### Cohort D

```
# split train, valid, and test data
x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(trainD)

# build and test model for each type
test_scores_D, Dmodels = build_test_models(x_train, y_train, x_valid, y_valid, x_test, y_test)

# visualize feature importances for each model
feature_importances(Dmodels, x_train.columns, x_train, y_train)
```

<img width="314" alt="Screen Shot 2024-05-27 at 6 13 06 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/96e8a287-c0c8-41c1-894d-a3d5e460a358">

<br>

<img width="314" alt="Screen Shot 2024-05-27 at 6 13 26 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/e7c6af49-2a76-4be8-bb09-fedae3a59f46">

<br>

<img width="498" alt="Screen Shot 2024-05-27 at 6 12 41 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/99c44dcf-98c6-4d90-9760-0abadaeab0ef">

### Cohort E

```
# split train, valid, and test data
x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(trainE)

# build and test model for each type
test_scores_E, Emodels = build_test_models(x_train, y_train, x_valid, y_valid, x_test, y_test)

# visualize feature importances for each model
feature_importances(Emodels, x_train.columns, x_train, y_train)
```

<img width="314" alt="Screen Shot 2024-05-27 at 6 14 33 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/a6f6db96-2a30-433b-9e3d-5ac74727ff5d">

<br>

<img width="314" alt="Screen Shot 2024-05-27 at 6 14 47 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/2973e07f-d9ba-4ec5-aa9d-0733f43c3275">

<br>

<img width="314" alt="Screen Shot 2024-05-27 at 6 15 05 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/b2c17e05-2273-4929-9e69-72ead863342c">

<br>

<img width="495" alt="Screen Shot 2024-05-27 at 6 14 07 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/11fa117a-20ce-4778-a0fb-fd18f908cc9e">

### Model Comparisons

```
test_scores = [test_scores_A, test_scores_B, test_scores_C, test_scores_D, test_scores_E]

print("\nTest Testing Accuracy for Various Classifier with Different DataSets: ")
print("===============================================================================")

comparison_result = pd.DataFrame(data=[['Random Forrest',test_scores[0][0], test_scores[1][0], test_scores[2][0], test_scores[3][0], test_scores[4][0]],
                                  ['XGB',test_scores[0]
                                  [1], test_scores[1][1], test_scores[2][1], test_scores[3][1], test_scores[4][1]],
                                  ["Logistic Regression", test_scores[0][2], test_scores[1][2], test_scores[2][2], test_scores[3][2], test_scores[4][2]],
                                  ['TabNet', test_scores[0][3], test_scores[1][3], test_scores[2][3], test_scores[3][3], test_scores[4][3]]],
                            columns=['Model', 'Cohort A', 'Cohort B', 'Cohort C', 'Cohort D', 'Cohort E'])
print(comparison_result)
print("===============================================================================")
print("\n A: none \n B: age*bmi \n C: age*avg_glucose_level \n D: bmi*avg_glucose_level \n E: B,C,D together")
```

<img width="356" alt="Screen Shot 2024-05-27 at 6 15 29 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/52df6924-8e1f-4ee3-b449-25fa9be8507f">

## Using Tabnet Architecture

### Cohort 0

```
# split data
x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(trainE)

# compile tabnet model
TabNetModel = compile_TabNet(x_train, y_train, x_valid, y_valid,
                             optimizer=torch.optim.Adam, step_size=10, gamma=0.9,
                             eval_met='accuracy', patience=60, batch_size=512)

# test and evaluate model
pred=TabNetModel.predict(x_test.values)
report0 = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
accuracy0 = accuracy_score(y_test, pred)

# print results
print("\n\nTest Result for TabNet Classifier: ")
print("================================================")
print(f"Accuracy Score: {accuracy0 * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{report0}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
```

<img width="318" alt="Screen Shot 2024-05-27 at 6 16 08 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/a91e7f13-b6b0-4d69-b42c-58d6b4049e3c">

### Cohort 1

```
# split data
x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(trainE)

# compile tabnet model
TabNetModel = compile_TabNet(x_train, y_train, x_valid, y_valid,
                             optimizer=torch.optim.SGD, step_size=10, gamma=0.9,
                             eval_met='accuracy', patience=60, batch_size=512)

# test and evaluate model
pred=TabNetModel.predict(x_test.values)
report1 = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
accuracy1 = accuracy_score(y_test, pred)

# print results
print("\n\nTest Result for TabNet Classifier: ")
print("================================================")
print(f"Accuracy Score: {accuracy1 * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{report1}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
```

<img width="318" alt="Screen Shot 2024-05-27 at 6 16 33 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/13dc0c1c-a9d5-44ff-ad63-9929037f6d05">

### Cohort 2

```
# split data
x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(trainE)

# compile tabnet model
TabNetModel = compile_TabNet(x_train, y_train, x_valid, y_valid,
                             optimizer=torch.optim.Adam, step_size=10, gamma=0.9,
                             eval_met='accuracy', patience=10, batch_size=216)

# test and evaluate model
pred=TabNetModel.predict(x_test.values)
report2 = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
accuracy2 = accuracy_score(y_test, pred)

# print results
print("\n\nTest Result for TabNet Classifier: ")
print("================================================")
print(f"Accuracy Score: {accuracy2 * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{report2}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
```

<img width="322" alt="Screen Shot 2024-05-27 at 6 16 59 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/57138744-ec66-44da-8a17-2113d576fff2">

### Cohort 3

```
# split data
x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(trainE)

# compile tabnet model
TabNetModel = compile_TabNet(x_train, y_train, x_valid, y_valid,
                             optimizer=torch.optim.Adam, step_size=5, gamma=0.95,
                             eval_met='accuracy', patience=60, batch_size=512)

# test and evaluate model
pred=TabNetModel.predict(x_test.values)
report3 = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
accuracy3 = accuracy_score(y_test, pred)

# print results
print("\n\nTest Result for TabNet Classifier: ")
print("================================================")
print(f"Accuracy Score: {accuracy3 * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{report3}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
```

<img width="316" alt="Screen Shot 2024-05-27 at 6 17 22 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/f23b8af3-66af-48ef-8cb8-87b3c52e5987">

### Tabnet Comparisons

```
print("\nTest Testing Accuracy for TabNet with Various Parameters: ")
print("==========================================================")

comparison_result = pd.DataFrame(data=[['TabNet', accuracy0, accuracy1, accuracy2, accuracy3]],
                            columns=['Model','Cohort 0', 'Cohort 1', 'Cohort 2', 'Cohort 3'])
print(comparison_result)
print("==========================================================")
print("\n 0: Original \n 1: SGB optimizer \n 2: Small batchsize \n 3: Small StepSize, Big Gamma")
```

<img width="259" alt="Screen Shot 2024-05-27 at 6 17 55 PM" src="https://github.com/jlee92603/StrokePrediction_Model/assets/70551445/867a4a3d-fef6-4eeb-985d-d8f39693fa1b">

## Conclusions


