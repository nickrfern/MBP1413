#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:30:07 2022

@author: nickfernandez
"""
print("Loading Packages")

import numpy as np
import pandas as pd
import time

from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score

# Cross-fold names
cvs = ["final"]

print("Building Finalized Model...")

acc = []
cm = []
recall = []

for i in cvs:
    load_str = "Loading" + i
    print(load_str)

    in_path = "/hpf/largeprojects/tabori/users/yuan/mbp1413/data/train_test_val"
    out_path = "/hpf/largeprojects/tabori/users/yuan/mbp1413/output/svm/models/"

    print("Loading Training Data")
    data = pd.read_csv(in_path + "/train_cv/pbmc68k_norm_x_train.csv")
    labels = pd.read_csv(in_path + "/train_cv/raw_data/pbmc68k_y_train.csv")

    print("Loading Test Data")
    data_val = pd.read_csv(in_path + "/test_cv/pbmc68k_norm_x_test.csv")
    labels_val = pd.read_csv(in_path + "/test_cv/raw_data/pbmc68k_y_test_cv.csv")

    # Set first column to rownames
    print("Transforming training data...")
    data = data.set_index(data["Unnamed: 0"])
    data = data.iloc[: , 1:]
    # Transpose data matrix
    X_train = data.T
    del data
    # Take labels column
    y_train = list(labels["x"])

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
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()

    print(end - start)
    
    pred = clf.predict(X_val)
    pred_cm = confusion_matrix(y_val, pred)
    pred_acc = accuracy_score(y_val, pred)
    
    pred_tb = {"pred": pred, "true": y_val}
    df_pred = pd.DataFrame(data=pred_tb)
    df_pred_path = out_path + "final_pred.csv"
    df_pred.to_csv(df_pred_path, index=False)

    cm.append(pred_cm)
    acc.append(pred_acc)

tb = {"names": cvs, "CM": cm, "ACC": acc}
df = pd.DataFrame(data=tb)
df_path = out_path + "final.csv"
df.to_csv(df_path, index=False)

