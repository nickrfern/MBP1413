Scripts for logstic regression classifier model training and final prediction on test data.

The lg_CV.py file would save hyperperameters of the best performed model for later useage and testing.

We rebuild the model in lg_pred.py which is fitted by concating training and validation data to ensure a complete traning dataset.

The lg_pred.py file would save the prediction matrix, accuracy and hyperperameters used.

utils.py is used for data loading.
