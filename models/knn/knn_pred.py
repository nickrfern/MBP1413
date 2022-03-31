from json import load
import utils
from sklearn.neighbors import KNeighborsClassifier
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

OUTPUT='/hpf/largeprojects/tabori/users/yuan/mbp1413/output/knn/'


X_train,X_validation,y_train,y_validation,X_test,y_test= loadData(1)
X_train = pd.concat([X_train,X_validation])
del X_validation
gc.collect()
y_train = np.append(y_train,y_validation)


k_range = [10]

###DF to save record
records = pd.DataFrame(columns=['metrics','train_acc','val_acc','matrix'])

###


for i in range(len(k_range)):
  k=k_range[i]
  
  ####define model######
  knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski',weights='uniform',n_jobs=(-1)).fit(X_train, y_train)
  ######################
  

  # record_k = utils.history(knn,X_train,y_train,X_validation,y_validation,k,None)
  # records = records.append(record_k,ignore_index=True)

  # ####save model
  # if i == 0:
  #   time_used=end-start
  #   acc = np.mean(record_k['val_acc'])
  #   variabels=k
  #   filename = f'{OUTPUT}knn_model.sav'
  #   pickle.dump(knn, open(filename, 'wb'))
  # elif np.mean(record_k['val_acc'])>acc:
  #   acc=np.mean(record_k['val_acc'])
  #   variabels=k
  #   time_used=end-start
  #   pickle.dump(knn, open(filename, 'wb'))
  #   print('dumped')
  ##########

##final Model test accuracies
# loaded_model = pickle.load(open(filename, 'rb'))
start=time.time()
y_predict = knn.predict(X_test)
end=time.time()

matrix = confusion_matrix(y_test,y_predict,labels=np.unique(y_test))
records = records.append({'metrics': 10,'train_acc':f'time{end-start}','val_acc': accuracy_score(y_test,y_predict),'matrix':matrix},ignore_index=True)
records.to_csv(f'{OUTPUT}knn_realtest_records.csv')
np.save(f'/hpf/largeprojects/tabori/users/yuan/mbp1413/output/Figs/knn_ypred',np.array(y_predict))

print('done for knn')
