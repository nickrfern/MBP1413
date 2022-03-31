#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:30:07 2022

@author: nickfernandez
"""
print("Loading Packages")

import numpy as np
import pandas as pd

from sklearn import svm
from joblib import load
from sklearn.metrics import confusion_matrix, accuracy_score

# Cross-fold names
cvs = ["cv2"]

acc = []
cm = []

for i in cvs:
    print(i)
    in_path = "/hpf/largeprojects/tabori/users/yuan/mbp1413/data/train_test_val/train_val_" + i
    out_path = "/hpf/largeprojects/tabori/users/yuan/mbp1413/output/svm/models/" + i

    print("Loading Training Data")
    data = pd.read_csv(in_path + "/train/pbmc68k_norm_x_train.csv")
    labels = pd.read_csv(in_path + "/train/raw_data/pbmc68k_y_train.csv")

    # Set first column to rownames
    print("Transforming training data...")
    data = data.set_index(data["Unnamed: 0"])
    data = data.iloc[: , 1:]
    # Transpose data matrix
    X_train = data.T
    del data
    # Take labels column
    y_train = list(labels["x"])

    print("Loading Validation Data")
    data_val = pd.read_csv(in_path + "/validation/pbmc68k_norm_x_validation.csv")
    labels_val = pd.read_csv(in_path + "/validation/raw_data/pbmc68k_y_validation.csv")

    print("Transforming validation data...")
    data_val = data_val.set_index(data_val["Unnamed: 0"])
    data_val = data_val.iloc[: , 1:]
    # Transpose data matrix
    X_val = data_val.T
    del data_val
    # Take labels column
    y_val = list(labels_val["x"])
    
    print("Trainign SVC classifier")
    clf = svm.SVC(decision_function_shape='ovo',kernel = "linear", random_state = 33)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_val)
    pred_cm = confusion_matrix(y_val, pred)
    pred_acc = accuracy_score(y_val, pred)
    
    cm.append(pred_cm)
    acc.append(pred_acc)

tb = {"names": cvs, "CM": cm, "ACC": acc}
df = pd.DataFrame(data=tb)
df_path = out_path + ".lin.csv"
df.to_csv(df_path, index=False)

