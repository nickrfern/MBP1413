y_pred <- read.csv("~/Desktop/all_y_pred.csv")
y <- read.csv("~/Desktop/all_y_pred.csv")

models <- c("truth", "LR", "SVM", "MLP", "RF", "AE", "ADB", "KNN", "SingleR")

# truth
y[y$truth=="CD4+/CD25 T Reg", "truth"] <- "CD4+ T-cell"
y[y$truth=="CD4+/CD45RA+/CD25- Naive T", "truth"] <- "CD4+ T-cell"
y[y$truth=="CD4+/CD45RO+ Memory", "truth"] <- "CD4+ T-cell"
y[y$truth=="CD4+ T Helper2", "truth"] <- "CD4+ T-cell"
y[y$truth=="CD8+ Cytotoxic T", "truth"] <- "CD8+ T-cell"
y[y$truth=="CD8+/CD45RA+ Naive Cytotoxic", "truth"] <- "CD8+ T-cell"

# lr
y[y$LR=="CD4+/CD25 T Reg", "LR"] <- "CD4+ T-cell"
y[y$LR=="CD4+/CD45RA+/CD25- Naive T", "LR"] <- "CD4+ T-cell"
y[y$LR=="CD4+/CD45RO+ Memory", "LR"] <- "CD4+ T-cell"
y[y$LR=="CD4+ T Helper2", "LR"] <- "CD4+ T-cell"
y[y$LR=="CD8+ Cytotoxic T", "LR"] <- "CD8+ T-cell"
y[y$LR=="CD8+/CD45RA+ Naive Cytotoxic", "LR"] <- "CD8+ T-cell"

# svm
y[y$SVM=="CD4+/CD25 T Reg", "SVM"] <- "CD4+ T-cell"
y[y$SVM=="CD4+/CD45RA+/CD25- Naive T", "SVM"] <- "CD4+ T-cell"
y[y$SVM=="CD4+/CD45RO+ Memory", "SVM"] <- "CD4+ T-cell"
y[y$SVM=="CD4+ T Helper2", "SVM"] <- "CD4+ T-cell"
y[y$SVM=="CD8+ Cytotoxic T", "SVM"] <- "CD8+ T-cell"
y[y$SVM=="CD8+/CD45RA+ Naive Cytotoxic", "SVM"] <- "CD8+ T-cell"

#mlp
y[y$MLP=="CD4+/CD25 T Reg", "MLP"] <- "CD4+ T-cell"
y[y$MLP=="CD4+/CD45RA+/CD25- Naive T", "MLP"] <- "CD4+ T-cell"
y[y$MLP=="CD4+/CD45RO+ Memory", "MLP"] <- "CD4+ T-cell"
y[y$MLP=="CD4+ T Helper2", "MLP"] <- "CD4+ T-cell"
y[y$MLP=="CD8+ Cytotoxic T", "MLP"] <- "CD8+ T-cell"
y[y$MLP=="CD8+/CD45RA+ Naive Cytotoxic", "MLP"] <- "CD8+ T-cell"

#rf
y[y$RF=="CD4+/CD25 T Reg", "RF"] <- "CD4+ T-cell"
y[y$RF=="CD4+/CD45RA+/CD25- Naive T", "RF"] <- "CD4+ T-cell"
y[y$RF=="CD4+/CD45RO+ Memory","RF"] <- "CD4+ T-cell"
y[y$RF=="CD4+ T Helper2", "RF"] <- "CD4+ T-cell"
y[y$RF=="CD8+ Cytotoxic T", "RF"] <- "CD8+ T-cell"
y[y$RF=="CD8+/CD45RA+ Naive Cytotoxic", "RF"] <- "CD8+ T-cell"


#ae 
y[y$AE=="CD4+/CD25 T Reg", "AE"] <- "CD4+ T-cell"
y[y$AE=="CD4+/CD45RA+/CD25- Naive T", "AE"] <- "CD4+ T-cell"
y[y$AE=="CD4+/CD45RO+ Memory", "AE"] <- "CD4+ T-cell"
y[y$AE=="CD4+ T Helper2", "AE"] <- "CD4+ T-cell"
y[y$AE=="CD8+ Cytotoxic T", "AE"] <- "CD8+ T-cell"
y[y$AE=="CD8+/CD45RA+ Naive Cytotoxic", "AE"] <- "CD8+ T-cell"

#adb
y[y$ADB=="CD4+/CD25 T Reg", 'ADB'] <- "CD4+ T-cell"
y[y$ADB=="CD4+/CD45RA+/CD25- Naive T", "ADB"] <- "CD4+ T-cell"
y[y$ADB=="CD4+/CD45RO+ Memory", 'ADB'] <- "CD4+ T-cell"
y[y$ADB=="CD4+ T Helper2", "ADB"] <- "CD4+ T-cell"
y[y$ADB=="CD8+ Cytotoxic T", "ADB"] <- "CD8+ T-cell"
y[y$ADB=="CD8+/CD45RA+ Naive Cytotoxic", "ADB"] <- "CD8+ T-cell"

#knn
y[y$KNN=="CD4+/CD25 T Reg", 'KNN'] <- "CD4+ T-cell"
y[y$KNN=="CD4+/CD45RA+/CD25- Naive T", "KNN"] <- "CD4+ T-cell"
y[y$KNN=="CD4+/CD45RO+ Memory", "KNN"] <- "CD4+ T-cell"
y[y$KNN=="CD4+ T Helper2", 'KNN'] <- "CD4+ T-cell"
y[y$KNN=="CD8+ Cytotoxic T", 'KNN'] <- "CD8+ T-cell"
y[y$KNN=="CD8+/CD45RA+ Naive Cytotoxic", "KNN"] <- "CD8+ T-cell"

#singler
y[y$SingleR=="CD4+/CD25 T Reg", "SingleR"] <- "CD4+ T-cell"
y[y$SingleR=="CD4+/CD45RA+/CD25- Naive T", "SingleR"] <- "CD4+ T-cell"
y[y$SingleR=="CD4+/CD45RO+ Memory", 'SingleR'] <- "CD4+ T-cell"
y[y$SingleR=="CD4+ T Helper2", 'SingleR'] <- "CD4+ T-cell"
y[y$SingleR=="CD8+ Cytotoxic T", "SingleR"] <- "CD8+ T-cell"
y[y$SingleR=="CD8+/CD45RA+ Naive Cytotoxic", "SingleR"] <- "CD8+ T-cell"

write.csv(y, "~/Desktop/grouped_y_pred.csv")


types <- c("CD14+ Monocyte","CD19+ B","CD34+","CD4+ T Helper2","CD4+/CD25 T Reg","CD4+/CD45RA+/CD25- Naive","CD4+/CD45RO+ Memory","CD56+ NK","CD8+ Cytotoxic T","CD8+/CD45RA+ Naive Cytotoxic","Dendritic")
