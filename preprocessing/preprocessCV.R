library(Seurat)
library(readr)

print("Seurat loaded...")

normCounts <- function(raw_data) {
  raw_data <- as.data.frame(raw_data)
  counts <- as.data.frame(raw_data[,-1])
  row.names(counts) <- raw_data[,1]
  
  pbmc <- CreateSeuratObject(counts = counts)
  pbmc_norm <- NormalizeData(object = pbmc)
  
  out <- as.data.frame(pbmc_norm@assays[["RNA"]]@data)
  
  return(out)
}

allDataPath <- '/hpf/largeprojects/tabori/users/yuan/mbp1413/data/train_test_val/train_val_cv'

for (i in (1:5)){
  
  dataPath <- paste(allDataPath,i,'/',sep = '')
  
  ############################
  # Normalize Validation Data
  ############################
  
  print("Normalizing validation...")
  
  val <- read_csv(paste(dataPath,"validation/raw_data/pbmc68k_raw_x_validation.csv",
                        sep = ''))
  val_norm <- normCounts(val)
  write.csv(val_norm, paste(dataPath,"validation/pbmc68k_norm_x_validation.csv",
                            sep = ''))
  
  # Clean memory
  rm(val)
  rm(val_norm)
  gc() 
  #######################
  # Normalize Train Data
  #######################
  
  print("Normalizing train...")
  
  train <- read_csv(paste(dataPath,"train/raw_data/pbmc68k_raw_x_train.csv",
                          sep = ''))
  
  train_norm <- normCounts(train)
  
  write.csv(train_norm, paste(dataPath,"train/pbmc68k_norm_x_train.csv",
                              sep = ''))
  
  rm(train)
  rm(train_norm)
  gc()
  print(paste("Completed for CV",i))
}


######################
# Normalize Test Data
######################

print("Normalizing test...")

test <- read_csv("/hpf/largeprojects/tabori/users/yuan/mbp1413/data/train_test_val/test_cv/raw_data/pbmc68k_raw_x_test_cv.csv")
test_norm <- normCounts(test)
write.csv(test_norm, "/hpf/largeprojects/tabori/users/yuan/mbp1413/data/train_test_val/test_cv/pbmc68k_norm_x_test.csv")

# Clean memory
rm(test)
rm(test_norm)
gc()
print("Completed")
