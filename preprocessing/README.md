Scripts for creating train/validation/test (7:1:2) splits with 5-fold cross-validation from the pbmc68k dataset and normalizing the resulting datasets.

**Brief Summary of Data Preprocessing Pipeline**
1. **pullCounts:** _Pulls data from the GitHub repository before extracting the counts matrices and cell type labels._

2. **Cross_Val_Split:** _Splits counts matrices into train/validation/test with stratification using train_test_split from sklearn/1.0.2._

3. **PreprocessCV:** _Separately normalizes train, validation and test matrices using NormalizeData from Seurat/4.0.1._
