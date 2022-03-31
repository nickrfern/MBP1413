from json import load
import utils
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import time
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras import Model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from sklearn.linear_model import LogisticRegression
import pydot
import gc

SERVER='HPF'
if SERVER=='local':
  from utils import loadDataLocal as loadData
elif SERVER=='HPF':
  from utils import loadDataHPF as loadData

OUTPUT='/hpf/largeprojects/tabori/users/yuan/mbp1413/output/encoder/'


###change code
metrics=[500]

###DF to save record
records = pd.DataFrame(columns=['metrics','time','val_acc','matrix'])

for cv in range(3,6):
  X_train,X_validation,y_train,y_validation,X_test,y_test= loadData(cv)
  del X_test,y_test
  gc.collect()
  for i in range(len(metrics)):
    ##change code
    metric = metrics[i]
    X_pca=PCA(n_components=metric).fit_transform(X_train)
    Val_pca= PCA(n_components=metric).fit_transform(X_validation)
    X_pca = MinMaxScaler().fit_transform(X_pca)
    Val_pca = MinMaxScaler().fit_transform(Val_pca)

    #input
    input = layers.Input(shape=(metric,))
    #encoder 1
    encoder = layers.Dense(400)(input)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.ReLU()(encoder)
    # encoder 2
    encoder = layers.Dense(100)(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.ReLU()(encoder)
    #bottleneck
    bottleneck = layers.Dense(90,name="bottleneck")(encoder)
    # decoder 1
    decoder = layers.Dense(100)(bottleneck)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.ReLU()(decoder)
    # decoder 2
    decoder = layers.Dense(400)(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.ReLU()(decoder)
    # output layer
    output = layers.Dense(metric, activation='linear')(decoder)

    #fit model
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
    start=time.time()
    history=model.fit(X_pca,X_pca,epochs=15,validation_data=(Val_pca,Val_pca))
    history=pd.DataFrame(history.history)
    end=time.time()

    # save the encoder to file
    if i == 0:
      time_used=end-start
      acc = history.iloc[-1,-1]
      variabels=metric
      filename = f'{OUTPUT}encoder_model'
      encode_model = Model(inputs=input, outputs=bottleneck)
      #plot_model(encode_model, f'{OUTPUT}encoder.png', show_shapes=True)
      encode_model.save(filename)
      history.to_csv(f'{OUTPUT}enc_dec_history.csv')

    elif history.iloc[-1,-1]>acc:
      acc = history.iloc[-1,-1]
      variabels=metric
      time_used=end-start
      encode_model  = Model(inputs=input, outputs=bottleneck)
      #plot_model(encode_model, f'{OUTPUT}encoder.png', show_shapes=True)
      encode_model.save(filename)
      history.to_csv(f'{OUTPUT}enc_dec_history.csv')
      print('encoder saved')

  records=records.append({'metrics':f'encoder_cv{cv}_{variabels}','time':time_used,'val_acc':acc,'matrix':None},ignore_index=True)
  records.to_csv(f'{OUTPUT}encoder_lgr_cv3_records.csv')

  ###lgr classification on bottelneck layer
  X_pca=PCA(n_components=variabels).fit_transform(X_train)
  Val_pca= PCA(n_components=variabels).fit_transform(X_validation)
  X_pca = MinMaxScaler().fit_transform(X_pca)
  Val_pca = MinMaxScaler().fit_transform(Val_pca)

#   del X_train,X_validation
#   gc.collect()

  model = load_model(f'{filename}')
  train_encoded = model.predict(X_pca)
  lgr = LogisticRegression(random_state=1).fit(train_encoded,y_train)
  Val_encoded = model.predict(Val_pca)

  #matrix = confusion_matrix(y_test,lgr.predict(test_encoded),labels=np.unique(y_test))
  #records = records.append({'metrics': variabels,'train_acc':time_used,'val_acc': accuracy_score(y_test, lgr.predict(test_encoded)),'matrix':matrix},ignore_index=True)

  matrix = confusion_matrix(y_validation,lgr.predict(Val_encoded),labels=np.unique(y_validation))
  records = records.append({'metrics': f'cv{cv}_{variabels}','time':time_used,'val_acc': accuracy_score(y_validation, lgr.predict(Val_encoded)),'matrix':matrix},ignore_index=True)
  del X_train,X_validation,y_train,y_validation
  gc.collect()
  records.to_csv(f'{OUTPUT}encoder_lgr_cv3_records.csv')
