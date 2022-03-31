Scripts for Autoencoder and LGR model training and final prediction on test data.
We only trined 1 set of hyperperameters on this ensembled model.
The encoder_CV.py file would save metrics of the best performed model for later useage and investigating. The encoder layers is also isolated and saved.
The plot funcion (commented now) in the encoder_CV.py can plot the structure of the model.
We rebuild the model in encoder_pred.py which is fitted by concating training and validation data to ensure a complete traning dataset.
The encoder_pred.py file would save the prediction matrix, accuracy and hyperperameters used.
