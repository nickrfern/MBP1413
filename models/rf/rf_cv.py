#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:30:07 2022

@author: nickfernandez
"""
print("Loading Packages")

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

in_path = "/hpf/largeprojects/tabori/users/yuan/mbp1413/data/train_test_val/train_val_cv1"
out_path = "/hpf/largeprojects/tabori/users/yuan/mbp1413/output/rf/models/cv1."

print("Loading Training Data")
data = pd.read_csv(in_path + "/train/pbmc68k_norm_x_train.csv")
labels = pd.read_csv(in_path + "/train/raw_data/pbmc68k_y_train.csv")

print("Loading Validation Data")
data_val = pd.read_csv(in_path + "/validation/pbmc68k_norm_x_validation.csv")
labels_val = pd.read_csv(in_path + "/validation/raw_data/pbmc68k_y_validation.csv")


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


# Optimization values
n_estimators = [100, 300, 500, 800]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]



print("Optimizing n_estimators...")

acc = []
cm = []
recall = []

for i in n_estimators:
    rf = RandomForestClassifier(n_estimators = i, random_state = 33)
    rf_model = rf.fit(X_train, y_train)
    
    pred = rf_model.predict(X_val)
    pred_cm = confusion_matrix(y_val, pred)
    pred_acc = accuracy_score(y_val, pred)
    pred_recall = recall_score(y_val, pred, average='micro')
    
    cm.append(pred_cm)
    acc.append(pred_acc)
    recall.append(pred_recall)
    
    dump_path = out_path + ".n_est." + str(i) + ".joblib"
    dump(rf_model, out_path)

tb = {"names": n_estimators, "CM": cm, "ACC": acc, "Recall": recall}
df = pd.DataFrame(data=tb)
df_path = out_path + ".n_est.csv"
df.to_csv(df_path, index=False)




print("Optimizing max_depth...")

acc = []
cm = []
recall = []

for i in max_depth:
    rf = RandomForestClassifier(max_depth = i, random_state = 33)
    rf_model = rf.fit(X_train, y_train)
    
    pred = rf_model.predict(X_val)
    pred_cm = confusion_matrix(y_val, pred)
    pred_acc = accuracy_score(y_val, pred)
    pred_recall = recall_score(y_val, pred, average='micro')
    
    cm.append(pred_cm)
    acc.append(pred_acc)
    recall.append(pred_recall)
    
    dump_path = out_path + ".max_depth." + str(i) + ".joblib"
    dump(rf_model, out_path)

tb = {"names": max_depth, "CM": cm, "ACC": acc, "Recall": recall}
df = pd.DataFrame(data=tb)
df_path = out_path + ".max_depth.csv"
df.to_csv(df_path, index=False)



print("Optimizing min_samples_split...")

acc = []
cm = []
recall = []

for i in min_samples_split:
    rf = RandomForestClassifier(min_samples_split = i, random_state = 33)
    rf_model = rf.fit(X_train, y_train)
    
    pred = rf_model.predict(X_val)
    pred_cm = confusion_matrix(y_val, pred)
    pred_acc = accuracy_score(y_val, pred)
    pred_recall = recall_score(y_val, pred, average='micro')
    
    cm.append(pred_cm)
    acc.append(pred_acc)
    recall.append(pred_recall)
    
    dump_path = out_path + ".min_split." + str(i) + ".joblib"
    dump(rf_model, out_path)

tb = {"names": min_samples_split, "CM": cm, "ACC": acc, "Recall": recall}
df = pd.DataFrame(data=tb)
df_path = out_path + ".min_split.csv"
df.to_csv(df_path, index=False)



print("Optimizing min_samples_leaf...")

acc = []
cm = []
recall = []

for i in min_samples_leaf:
    rf = RandomForestClassifier(min_samples_leaf = i, random_state = 33)
    rf_model = rf.fit(X_train, y_train)
    
    pred = rf_model.predict(X_val)
    pred_cm = confusion_matrix(y_val, pred)
    pred_acc = accuracy_score(y_val, pred)
    pred_recall = recall_score(y_val, pred, average='micro')
    
    cm.append(pred_cm)
    acc.append(pred_acc)
    recall.append(pred_recall)
    
    dump_path = out_path + ".min_leaf." + str(i) + ".joblib"
    dump(rf_model, out_path)

tb = {"names": min_samples_leaf, "CM": cm, "ACC": acc, "Recall": recall}
df = pd.DataFrame(data=tb)
df_path = out_path + ".min_leaf.csv"
df.to_csv(df_path, index=False)


