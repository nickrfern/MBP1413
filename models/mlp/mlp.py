from json import load
import utils
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import time

SERVER='HPF'
if SERVER=='local':
  from utils import loadDataLocal as loadData
elif SERVER=='HPF':
  from utils import loadDataHPF as loadData

OUTPUT='/hpf/largeprojects/tabori/users/yuan/mbp1413/output/mlp/'

X_train,X_validation,y_train,y_validation,X_test,y_test= loadData(1)

metrics=[(100),(100,100),(300),(300,300)]
###DF to save record
records = pd.DataFrame(columns=['metrics','train_acc','val_acc','matrix'])


for i in range(len(metrics)):
  ##change code
  metric = metrics[i]
  start=time.time()
  ####define model######
  mlp = MLPClassifier(random_state=42,hidden_layer_sizes=metric,learning_rate_init=0.001,activation='relu').fit(X_train, y_train)
  ######################
  end=time.time()

  ##change code
  record_metric = utils.history(mlp,X_train,y_train,X_validation,y_validation,metric,None)
  records = records.append(record_metric,ignore_index=True)

  ####save model
  if i == 0:
    time_used=end-start
    acc = np.mean(record_metric['val_acc'])
    ##change code
    variabels=metric
    filename = f'{OUTPUT}mlp_model.sav'
    pickle.dump(mlp, open(filename, 'wb'))
  elif np.mean(record_metric['val_acc'])>acc:
    acc=np.mean(record_metric['val_acc'])
    variabels=metric
    time_used=end-start
    pickle.dump(mlp, open(filename, 'wb'))
    print('dumped')
  ##########

##final Model test accuracies
loaded_model = pickle.load(open(filename, 'rb'))
matrix = confusion_matrix(y_test,loaded_model.predict(X_test),labels=np.unique(y_test))
records = records.append({'metrics': variabels,'train_acc':time_used,'val_acc': accuracy_score(y_test, loaded_model.predict(X_test)),'matrix':matrix},ignore_index=True)

##change code
records.to_csv(f'{OUTPUT}mlp_records.csv')
print('done for mlp')
