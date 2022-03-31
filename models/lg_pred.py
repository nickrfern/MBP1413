from json import load
import utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import time
import gc

SERVER='HPF'
if SERVER=='local':
  from utils import loadDataLocal as loadData
elif SERVER=='HPF':
  from utils import loadDataHPF as loadData

OUTPUT='/hpf/largeprojects/tabori/users/yuan/mbp1413/output/lgr/'

X_train,X_validation,y_train,y_validation,X_test,y_test= loadData(1)
X_train = pd.concat([X_train,X_validation])
del X_validation
gc.collect()
y_train = np.append(y_train,y_validation)



metrics=[['l2',None]]
###DF to save record
records = pd.DataFrame(columns=['metrics','train_acc','val_acc','matrix'])


for i in range(len(metrics)):
  ##change code
  metric = metrics[i]
  start=time.time()
  ####define model######
  lgr = LogisticRegression(penalty=metric[0],class_weight=metric[1],n_jobs=(-1),random_state=42,max_iter=150).fit(X_train, y_train)
  ######################
  end=time.time()

#   ##change code
#   record_metric = utils.history(lgr,X_train,y_train,X_validation,y_validation,metric,None)
#   records = records.append(record_metric,ignore_index=True)

#   ####save model
#   if i == 0:
#     time_used=end-start
#     acc = np.mean(record_metric['val_acc'])
#     ##change code
#     variabels=metric
#     filename = f'{OUTPUT}lgr_model.sav'
#     pickle.dump(lgr, open(filename, 'wb'))
#   elif np.mean(record_metric['val_acc'])>acc:
#     acc = np.mean(record_metric['val_acc'])
#     variabels=metric
#     time_used=end-start
#     pickle.dump(lgr, open(filename, 'wb'))
#     print('dumped')
#   ##########

# ##final Model test accuracies
# loaded_model = pickle.load(open(filename, 'rb'))
y_predict = lgr.predict(X_test)
matrix = confusion_matrix(y_test,y_predict,labels=np.unique(y_test))
records = records.append({'metrics': metrics,'train_acc':f'time{end-start}','val_acc': accuracy_score(y_test,y_predict),'matrix':matrix},ignore_index=True)
records.to_csv(f'{OUTPUT}lgr_realtest_records.csv')
np.save(f'/hpf/largeprojects/tabori/users/yuan/mbp1413/output/Figs/lgr_ypred',np.array(y_predict))

print('done for lgr')