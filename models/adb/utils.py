import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np


def loadDataLocal(i):
  data_path = f'~/Documents/MBP/AI_course/code/sc/test_data/test/'

  #load data

  X_train = pd.read_csv((data_path+f'pbmc68k_norm_x_test.csv'))
  X_validation = pd.read_csv((data_path+f'pbmc68k_norm_x_test.csv'))
  X_train = X_train.set_index('Unnamed: 0').T
  X_validation = X_validation.set_index('Unnamed: 0').T

  X_test = pd.read_csv(data_path+'pbmc68k_norm_x_test.csv')
  X_test = X_test.set_index('Unnamed: 0').T

  y_train_ = pd.read_csv((data_path+f'pbmc68k_y_test.csv'))
  y_validation_ = pd.read_csv((data_path+f'pbmc68k_y_test.csv'))
  y_train = y_train_.x.to_list()
  y_validation=y_validation_.x.to_list()

  y_test = pd.read_csv(data_path+f'pbmc68k_y_test.csv')
  y_test = y_test.x.to_list()

  return X_train,X_validation,y_train,y_validation,X_test,y_test


def loadDataHPF(i):
  data_path = f'/hpf/largeprojects/tabori/users/yuan/mbp1413/data/train_test_val/train_val_cv{i}'

  #load data
  X_train = pd.read_csv((data_path+f'/train/pbmc68k_norm_x_train.csv'))
  X_validation = pd.read_csv((data_path+f'/validation/pbmc68k_norm_x_validation.csv'))
  X_train = X_train.set_index('Unnamed: 0').T
  X_validation = X_validation.set_index('Unnamed: 0').T

  X_test = pd.read_csv('/hpf/largeprojects/tabori/users/yuan/mbp1413/data/train_test_val/test_cv/pbmc68k_norm_x_test.csv')
  X_test = X_test.set_index('Unnamed: 0').T

  y_train_ = pd.read_csv((data_path+f'/train/raw_data/pbmc68k_y_train.csv'))
  y_validation_ = pd.read_csv((data_path+f'/validation/raw_data/pbmc68k_y_validation.csv'))
  y_train = y_train_.x.to_list()
  y_validation=y_validation_.x.to_list()

  y_test = pd.read_csv('/hpf/largeprojects/tabori/users/yuan/mbp1413/data/train_test_val/test_cv/raw_data/pbmc68k_y_test_cv.csv')
  y_test = y_test.x.to_list()

  return X_train,X_validation,y_train,y_validation,X_test,y_test

def history(model,X_train,y_train,X_validation,y_validation,metrics,matrix):
  train_acc = accuracy_score(y_train, model.predict(X_train))
  val_acc = accuracy_score(y_validation, model.predict(X_validation))
  records = {'metrics': metrics,'train_acc': train_acc,'val_acc': val_acc,'matrix':matrix}
  return records
