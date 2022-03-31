Scripts for training, validating and testing models. The final predictions for each model can be found in _all_y_pred.csv_

**Brief Summary of Model Training Pipeline**
1. **{ModelName}_CV.py:** _We trained each of our models on the 5-folds of normalized data prepared in our preprocessing stage. Each CV script searches multiple hyperparameter arguments for each model and returns an accuracy, training time and confusion matrix as its output._

2. **{ModelName}_pred.py:** _Once we completed cross-validation, we selected the optimized hyperparameters to run for each model in our testing phase. Each pred script trains a model with selected hyperparameters and predicts classes of the test dataset. They return an accuracy, training time, confusion matrix and list of **predicted classes** as its output._

3. **utils.py:** _General utility functions for model training._

