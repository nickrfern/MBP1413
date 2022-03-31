import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report, f1_score

y = pd.read_csv("~/Desktop/all_y_pred.csv")
g_y = pd.read_csv("~/Desktop/grouped_y_pred.csv")

truth = y["truth"]
g_truth = g_y["truth"]

models = ["LR", "SVM", "MLP", "RF", "AE", "ADB", "KNN", "SingleR"]

types = ["CD14+ Monocyte","CD19+ B","CD34+","CD4+ T Helper2","CD4+/CD25 T Reg","CD4+/CD45RA+/CD25- Naive","CD4+/CD45RO+ Memory","CD56+ NK","CD8+ Cytotoxic T","CD8+/CD45RA+ Naive Cytotoxic","Dendritic"]
group_types = ["CD14+ Monocyte", "CD19+ B", "CD34+", "CD4+ T-cell", "CD56+ NK", "CD8+ T-cell", "Dendritic"]


print(classification_report(truth, y["SVM"]))
f1 = f1_score(g_truth, g_y["SVM"], average = None)

f_list = []

for i in models:
    f1 = f1_score(truth, y[i], average = None)
    f_list.append(f1)
    
df = pd.DataFrame(f_list, columns = types)
df["model"] = models


f_group_list = []

for i in models:
    f1_group = f1_score(g_truth, g_y[i], average = None)
    f_group_list.append(f1_group)
    
df_group = pd.DataFrame(f_group_list, columns = group_types)
df_group["model"] = models

df.to_csv("~/Desktop/f1.csv")
df_group.to_csv("~/Desktop/f1.group.csv")