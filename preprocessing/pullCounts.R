library(Seurat)
library(Matrix)

print("Loading pbmc68k")
pbmc <- readRDS("/hpf/largeprojects/tabori/users/yuan/mbp1413/data/pbmc68k/pbmc68k_data.rds")
all_data <- pbmc[["all_data"]]
rm(pbmc)

anno <- read.delim("/hpf/largeprojects/tabori/users/yuan/mbp1413/data/pbmc68k/68k_pbmc_barcodes_annotation.tsv")
  
cells <- all_data[["17820"]][["hg19"]][["barcodes"]]
genes <- all_data[["17820"]][["hg19"]][["genes"]] 

print("Pulling and transforming Matrix")  
mat <- all_data[["17820"]][["hg19"]][["mat"]]
mat <- as.matrix(mat)
mat <- t(mat)
dimnames(mat) = list(genes, cells)

pbmc <- CreateSeuratObject(counts = mat)
pbmc <- AddMetaData(pbmc, anno$celltype, col.name = "celltype")

write.csv(mat, "/hpf/largeprojects/tabori/users/yuan/mbp1413/data/raw_data/pbmc68k_raw_x.csv")
