# MBP1413

Code Repository for the final project of MBP1413H, March 2022. Written by Nicholas Fernandez and Yuan Chang

Data were accessed from the pbmc68k paper associated with the following GitHub repository: https://github.com/10XGenomics/single-cell-3prime-paper. (Zheng, G., Terry, J., Belgrader, P. et al. Massively parallel digital transcriptional profiling of single cells. Nat Commun 8, 14049 (2017).)

**Brief Summary of Data Analysis Pipeline**
1. Preprocessing: _Data was pulled from the GitHub repository and the counts matrix was split into train, validation and test cohorts (7:1:2) before log-normalization in Seurat._

2. Models: _We evaluated the performance of 7 supervised (Logistic Regression, Autoencoder + Logistic Regression, Adaboost, RandomForest, SVM, SingleR and MLP) and 1 unsupervised (KNN) models on the training set with 5-fold cross validation to tune our hyperparameters before testing for final accuracy on the 20% reserved set of test data._

3. Plots: _Scripts for the visualization of all figures contained in the paper can be found here._




_All training, validation and testing was run on the RIT-HPC compute cluster at the Hospital for Sick Children, Toronto, ON. Scripts were completed using CPU resources with Xeon E5-2670 v2 and Xeon Gold 6140 cores and usage of GPU resources due to cluster access limitations. All jobs requested 1 node, 32 processors and a flexible amount of RAM as required by the specific method. Scripts were run in R version 4.1.2 and Python 3.8.1. _
