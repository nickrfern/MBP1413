from json import load
import utils
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from sklearn.ensemble import AdaBoostClassifier
import gc

SERVER='HPF'
if SERVER=='local':
  from utils import loadDataLocal as loadData
elif SERVER=='HPF':
  from utils import loadDataHPF as loadData

OUTPUT='/hpf/largeprojects/tabori/users/yuan/mbp1413/output/adb/'



metrics=[5,10,12,50]

###DF to save record
records = pd.DataFrame(columns=['metrics','train_acc','val_acc','matrix'])

for cv in range(1,6):
  X_train,X_validation,y_train,y_validation,X_test,y_test= loadData(cv)
  del X_test,y_test
  gc.collect()
  for i in range(len(metrics)):
    ##change code
    metric = metrics[i]
    start=time.time()
    ####define model######
    adb = AdaBoostClassifier(base_estimator=None,n_estimators=metric,random_state=42).fit(X_train, y_train)
    ######################
    end=time.time()

    ##change code
    matrix = confusion_matrix(y_validation,adb.predict(X_validation),labels=np.unique(y_validation))
    record_metric = utils.history(adb,X_train,y_train,X_validation,y_validation,metric,matrix)
    records = records.append(record_metric,ignore_index=True)

    ####save model
    if i == 0:
      time_used=end-start
      acc = np.mean(record_metric['val_acc'])
      ##change code
      variabels=metric
      filename = f'{OUTPUT}adb_model.sav'
      pickle.dump(adb, open(filename, 'wb'))
    elif np.mean(record_metric['val_acc'])>acc:
      acc = np.mean(record_metric['val_acc'])
      variabels=metric
      time_used=end-start
      pickle.dump(adb, open(filename, 'wb'))
      print('dumped')
    ##########
    print(f'{metric} done')

  ##final Model test accuracies
  loaded_model = pickle.load(open(filename, 'rb'))
  matrix = confusion_matrix(y_test,loaded_model.predict(X_test),labels=np.unique(y_test))
  records = records.append({'metrics': variabels,'train_acc':time_used,'val_acc': accuracy_score(y_test, loaded_model.predict(X_test)),'matrix':matrix},ignore_index=True)


  records.to_csv(f'{OUTPUT}adb_CV5_records.csv')
  del X_train,X_validation,y_train,y_validation
  gc.collect()
  print('done for adb')
