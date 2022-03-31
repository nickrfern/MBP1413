library(ggplot2)
library(dplyr)
library(reshape2)
library(patchwork)

f1 <- read.csv("~/Desktop/f1_subgroup/f1.csv")
f1.group <- read.csv("~/Desktop/f1_subgroup/f1.group.csv") 

colnames(f1) <- c("CD14+ Monocyte","CD19+ B","CD34+","CD4+ T Helper2","CD4+/CD25 T Reg","CD4+/CD45RA+/CD25- Naive","CD4+/CD45RO+ Memory","CD56+ NK","CD8+ Cytotoxic T","CD8+/CD45RA+ Naive Cytotoxic","Dendritic","model")
df <- melt(f1)

colnames(f1.group) <- c("CD14+ Monocyte", "CD19+ B", "CD34+", "CD4+ T-cell", "CD56+ NK", "CD8+ T-cell", "Dendritic","model")
df.group <- melt(f1.group)

df$model <- factor(df$model,levels = c("LR", "SVM", "MLP", "RF", "AE + LR", "ADB", "KNN", "SingleR"))
df.group$model <- factor(df.group$model,levels = c("LR", "SVM", "MLP", "RF", "AE + LR", "ADB", "KNN", "SingleR"))

p1 <- ggplot(df, aes(x = model, y = variable, fill = value)) +
  geom_tile() + 
  theme_minimal() + 
  xlab("Model") + 
  ylab("Cell Type") +
  labs(fill='F1-Score') +
  scale_fill_gradient(low = "#FFFFD3", high = "#060B5E") +
  theme(legend.position="none") 

p2 <- ggplot(df.group, aes(x = model, y = variable, fill = value)) +
  geom_tile() + 
  theme_minimal() +
  xlab("Model") + 
  ylab("") +
  scale_fill_gradient(low = "#FFFFD3", high = "#060B5E") +
  labs(fill='F1-Score') 

p1 | p2
