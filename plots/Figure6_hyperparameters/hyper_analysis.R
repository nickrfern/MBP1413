library(ggplot2)
library(patchwork)

models <- c("adb", "encoder", "knn", "lgr", "mlp", "rf", "svm")
model.df <- c()
df <- data.frame()

for (i in models) {
  path = paste0("~/Desktop/hyper/", i, "_records_cv.csv")
  df.cv <- read.csv(path)
  df.cv$model <- i
  model.df <- c(model.df, df.cv)
  df <- rbind(df, df.cv)
}

pal <- c("#ff6961", "#ffb480","#42d6a4","#59adf6", "#301934")
knn_pal <- c("#ff6961", "#ffb480", "#f8f38d", "#42d6a4", "#08cad1", "#59adf6", "#9d94ff", "#c780e8", "#301934")
rf_pal <- c("#ff6961", "#ffb480", "#f8f38d", "#42d6a4", "#08cad1", "#59adf6", "#9d94ff", "#c780e8", "#301934", "#ff6961", "#ffb480","#42d6a4","#59adf6")

### Adaboost
adb <- read.csv(paste0("~/Desktop/hyper/adb_records_cv.csv"))

adb$metrics <- as.character(adb$metrics)
adb$metrics <- factor(adb$metrics,levels = c("5", "10", "12", "50"))
adb.plot <- ggplot(data = adb, aes(x = metrics, y = val_acc, fill = metrics)) +
              geom_boxplot(alpha = 0.8) +
              geom_point() + 
              ylim(c(0.3,1)) +
              theme_classic() +
              scale_fill_manual(values = pal) +
              xlab("n_estimators") + 
              ylab("Accuracy") +
              theme(legend.position="none") +
              ggtitle("Adaboost") +
              theme(plot.title = element_text(hjust = 0.5))

### Encoder
encoder <- read.csv(paste0("~/Desktop/hyper/encoder_records_cv.csv"))
encoder$metrics <- as.character(encoder$metrics)
encoder.plot <- ggplot(data = encoder, aes(x = metrics, y = val_acc, fill = metrics)) +
                  geom_boxplot(alpha = 0.8) +
                  geom_point() + 
                  ylim(c(0.3,1)) +
                  theme_classic() +
                  scale_fill_manual(values = pal) +
                  xlab("n_components") + 
                  ylab("Accuracy") +
                  theme(legend.position="none") +
                  ggtitle("Logistic Regression + Autoencoder") +
                  theme(plot.title = element_text(hjust = 0.5))

### KNN
knn <- read.csv(paste0("~/Desktop/hyper/knn_records_cv.csv"))
knn$metrics <- as.character(knn$metrics)
knn$metrics <- factor(knn$metrics,levels = c("4", "5", "6", "8", "10", "30", "50", "70", "99"))
knn.plot <- ggplot(data = knn, aes(x = metrics, y = val_acc, fill = metrics)) +
              geom_boxplot(alpha = 0.8) +
              geom_point() + 
              ylim(c(0.3,1)) +
              theme_classic() +
              scale_fill_manual(values = knn_pal) +
              xlab("n_neighbors") + 
              ylab("Accuracy") +
              theme(legend.position="none") +
              ggtitle("KNN") +
              theme(plot.title = element_text(hjust = 0.5))


### LGR
lgr <- read.csv(paste0("~/Desktop/hyper/lgr_records_cv.csv"))
lgr$metrics <- factor(lgr$metrics,levels = c("['none', None]", "['l2', None]", "['none', 'balanced']", "['l2', 'balanced']"))
lgr.plot <- ggplot(data = lgr, aes(x = metrics, y = val_acc, fill = metrics)) +
              geom_boxplot(alpha = 0.8) +
              geom_point() + 
              ylim(c(0.3,1)) +
              theme_classic() +
              scale_fill_manual(values = pal) +
              xlab("Penalty; Class_Weight") + 
              ylab("Accuracy") +
              theme(legend.position="none") +
              ggtitle("LGR") +
              theme(plot.title = element_text(hjust = 0.5))
### MLP
mlp <- read.csv(paste0("~/Desktop/hyper/mlp_records_cv.csv"))
mlp$metrics <- factor(mlp$metrics,levels = c("100", "(100, 100)", "300", "(300, 300)"))
mlp.plot <- ggplot(data = mlp, aes(x = metrics, y = val_acc, fill = metrics)) +
  geom_boxplot(alpha = 0.8) +
  geom_point() + 
  ylim(c(0.3,1)) +
  theme_classic() +
  scale_fill_manual(values = pal) +
  xlab("hidden_layer_sizes") + 
  ylab("Accuracy") +
  theme(legend.position="none") +
  ggtitle("MLP") +
  theme(plot.title = element_text(hjust = 0.5))

### SVM
svm <- read.csv(paste0("~/Desktop/hyper/svm_records_cv.csv"))
svm.plot <- ggplot(data = svm, aes(x = metrics, y = val_acc, fill = metrics)) +
  geom_boxplot(alpha = 0.8) +
  geom_point() + 
  ylim(c(0.3,1)) +
  theme_classic() +
  scale_fill_manual(values = pal) +
  xlab("Kernel") + 
  ylab("Accuracy") +
  theme(legend.position="none") +
  ggtitle("SVM") +
  theme(plot.title = element_text(hjust = 0.5))


### RF
rf <- read.csv(paste0("~/Desktop/hyper/rf_records_cv.csv"))
rf$metrics <- as.character(rf$metrics)

rf.depth <- rf[rf$param == "max_depth",]
rf.depth$metrics <- factor(rf.depth$metrics,levels = c("5", "8", "15", "25", "30"))

rf.leaf <- rf[rf$param == "min_leaf",]
rf.leaf$metrics <- factor(rf.leaf$metrics,levels = c("1", "2", "5", "10"))

rf.split <- rf[rf$param == "min_split",]
rf.split$metrics <- factor(rf.split$metrics,levels = c("2", "5", "10", "15", "100"))

rf.est <- rf[rf$param == "n_est",]
rf.est$metrics <- factor(rf.est$metrics,levels = c("100", "300", "500", "800"))

rf.depth.plot <- ggplot(data = rf.depth, aes(x = metrics, y = val_acc, fill = metrics)) +
  geom_boxplot(alpha = 0.8) +
  geom_point() + 
  ylim(c(0.3,1)) +
  theme_classic() +
  scale_fill_manual(values = pal) +
  xlab("max_depth") + 
  ylab("Accuracy") +
  theme(legend.position="none") +
  ggtitle("Random Forest") +
  theme(plot.title = element_text(hjust = 0.5))

rf.leaf.plot <- ggplot(data = rf.leaf, aes(x = metrics, y = val_acc, fill = metrics)) +
  geom_boxplot(alpha = 0.8) +
  geom_point() + 
  ylim(c(0.3,1)) +
  theme_classic() +
  scale_fill_manual(values = pal) +
  xlab("min_samples_leaf") + 
  ylab("Accuracy") +
  theme(legend.position="none") +
  ggtitle("Random Forest") +
  theme(plot.title = element_text(hjust = 0.5))

rf.split.plot <- ggplot(data = rf.split, aes(x = metrics, y = val_acc, fill = metrics)) +
  geom_boxplot(alpha = 0.8) +
  geom_point() + 
  ylim(c(0.3,1)) +
  theme_classic() +
  scale_fill_manual(values = pal) +
  xlab("min_samples_split") + 
  ylab("Accuracy") +
  theme(legend.position="none") +
  ggtitle("Random Forest") +
  theme(plot.title = element_text(hjust = 0.5))

rf.est.plot <- ggplot(data = rf.est, aes(x = metrics, y = val_acc, fill = metrics)) +
  geom_boxplot(alpha = 0.8) +
  geom_point() + 
  ylim(c(0.3,1)) +
  theme_classic() +
  scale_fill_manual(values = pal) +
  xlab("n_estimators") + 
  ylab("Accuracy") +
  theme(legend.position="none") +
  ggtitle("Random Forest") +
  theme(plot.title = element_text(hjust = 0.5))

gc()
pp <- ((svm.plot / encoder.plot) | (lgr.plot / mlp.plot) | (knn.plot / adb.plot))  
pp.rf <- ((rf.depth.plot / rf.leaf.plot) | (rf.split.plot / rf.est.plot))
pp
pp.rf
#ggsave("~/Desktop/hyper.plot.pdf", width = 25, height = 5)
