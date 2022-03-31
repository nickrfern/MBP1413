library(SingleR)


print("Loading train x...")
x.train <- read.csv("/hpf/largeprojects/tabori/users/yuan/mbp1413/data/train_test_val/train_cv/pbmc68k_norm_x_train.csv")
rownames(x.train) <- x.train[,1]
x.train <- x.train[,-1]

print("Loading test x...")
x.test <- read.csv("/hpf/largeprojects/tabori/users/yuan/mbp1413/data/train_test_val/test_cv/pbmc68k_norm_x_test.csv")
rownames(x.test) <- x.test[,1]
x.test <- x.test[,-1]

print("Loading train y...")
y.train <- read.csv("/hpf/largeprojects/tabori/users/yuan/mbp1413/data/train_test_val/train_cv/raw_data/pbmc68k_y_train.csv")
y.train.types <- y.train[c("x")]

print("Loading test y...")
y.test <- read.csv("/hpf/largeprojects/tabori/users/yuan/mbp1413/data/train_test_val/test_cv/raw_data/pbmc68k_y_test_cv.csv")
y.test.types <- y.test[c("x")]

print("Creating ref experiment...")
x.train.list <- list(logcounts = x.train)
ref.exp <- SummarizedExperiment(assays = x.train.list, colData = y.train.types)
ref.exp

print("Creating test experiment...")
x.test.list <- list(logcounts = x.test)
test.exp <- SummarizedExperiment(assays = x.test.list)
test.exp

print("Running SingleR...")
pred <- SingleR(test = x.test, ref = x.train, assay.type.test=1, labels = ref.exp@colData@listData[["x"]])
saveRDS(pred, file = "/hpf/largeprojects/tabori/users/yuan/mbp1413/output/singler/pred.rds")

df <- data.frame(pred = pred$labels,
                 true = y.test$x)

write.csv(df, "/hpf/largeprojects/tabori/users/yuan/mbp1413/output/singler/pred.csv")


