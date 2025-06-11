# =======================================================================
# 微生物豐度數據合併及批次效應校正腳本（MMUPHin 正確格式）
# =======================================================================

library(tidyverse)
library(readr)
library(MMUPHin)

setwd("/Users/yangkeyi/Downloads/predict_relapse/2metaphalan_data")

# ------------- 讀取數據 -------------
enigma_abundance <- read_tsv("ENIGMA_abundance_clean.txt")
ibdr_abundance <- read_tsv("IBDR_abundance_clean.txt")
longitude_abundance <- read_tsv("longitude_abundance_clean.txt")

cat("ENIGMA:", dim(enigma_abundance), "\n")
cat("IBDR:", dim(ibdr_abundance), "\n")
cat("Longitude:", dim(longitude_abundance), "\n")

# ------------- 預處理 -------------
colnames(enigma_abundance)[1] <- "taxonomy"
colnames(ibdr_abundance)[1] <- "taxonomy"
colnames(longitude_abundance)[1] <- "taxonomy"

# 合併
merged_data <- enigma_abundance %>%
  full_join(ibdr_abundance, by = "taxonomy") %>%
  full_join(longitude_abundance, by = "taxonomy")
merged_data[is.na(merged_data)] <- 0
cat("合併後維度:", dim(merged_data), "\n")
write_tsv(merged_data, "merged_abundance_all_studies_no_na.txt")

# ------------- 移除問題樣本 -------------
problem_samples <- c("CD117", "CD148", "CD149", "CD150", "CD160", "CD177", "CD178", "CD179")
sample_names <- setdiff(colnames(merged_data)[-1], problem_samples)
keep_cols <- c("taxonomy", sample_names)
merged_data_filtered <- merged_data %>% dplyr::select(dplyr::all_of(keep_cols))



# ------------- 建立批次資訊 -------------
enigma_samples <- colnames(enigma_abundance)[-1]
ibdr_samples <- colnames(ibdr_abundance)[-1]
longitude_samples <- colnames(longitude_abundance)[-1]

batch_info <- data.frame(
  sample_id = sample_names,
  batch = NA_character_,
  stringsAsFactors = FALSE
)
batch_info$batch[batch_info$sample_id %in% enigma_samples] <- "ENIGMA"
batch_info$batch[batch_info$sample_id %in% ibdr_samples] <- "IBDR"
batch_info$batch[batch_info$sample_id %in% longitude_samples] <- "Longitude"
batch_info$batch[is.na(batch_info$batch)] <- "Unknown"
print(table(batch_info$batch))

# ------------- 準備 MMUPHin 格式 -------------
# 行:分類單位，列:樣本
abundance_matrix <- as.matrix(merged_data_filtered[,-1])
rownames(abundance_matrix) <- merged_data_filtered$taxonomy

# 僅保留已知批次
if ("Unknown" %in% batch_info$batch) {
  known_samples <- batch_info$sample_id[batch_info$batch != "Unknown"]
  batch_info <- batch_info[batch_info$batch != "Unknown", ]
  abundance_matrix <- abundance_matrix[, colnames(abundance_matrix) %in% known_samples]
}

# 強制型態與順序一致
batch_info$sample_id <- as.character(batch_info$sample_id)
# 依 abundance_matrix 的樣本順序排列 batch_info
batch_info <- batch_info[match(colnames(abundance_matrix), batch_info$sample_id), ]
rownames(batch_info) <- batch_info$sample_id

# abundance_matrix 為數值型
abundance_matrix <- apply(abundance_matrix, 2, as.numeric)
abundance_matrix <- as.matrix(abundance_matrix)

# 轉為比例
abundance_matrix <- apply(abundance_matrix, 2, function(x) x / sum(x))
abundance_matrix[is.na(abundance_matrix)] <- 0

# Debug check
cat("abundance_matrix: ", dim(abundance_matrix), "\n")
cat("batch_info: ", dim(batch_info), "\n")
cat("colnames(abundance_matrix)[1:5]: ", head(colnames(abundance_matrix)), "\n")
cat("batch_info$sample_id[1:5]: ", head(batch_info$sample_id), "\n")
cat("完全一致？", all(colnames(abundance_matrix) == batch_info$sample_id), "\n")

# ------------- 執行校正 -------------
tryCatch({
  batch_corrected <- adjust_batch(
    feature_abd = abundance_matrix,
    batch = "batch",
    data = batch_info,
    covariates = NULL,
    control = list(verbose = TRUE)
  )
  corrected_abundance <- batch_corrected$feature_abd_adj
  # 轉置回 行:分類單位，列:樣本
  corrected_data <- as.data.frame(corrected_abundance)
  # 確保行名被保留
  corrected_data$taxonomy <- merged_data_filtered$taxonomy
  # 重新排列列，確保 taxonomy 在第一列
  corrected_data <- corrected_data[, c("taxonomy", setdiff(colnames(corrected_data), "taxonomy"))]
  
  # 檢查每列的和
  col_sums <- colSums(corrected_data[, -1])  # 排除 taxonomy 列
  cat("\n每列的和：\n")
  print(summary(col_sums))
  cat("\n前5個樣本的列和：\n")
  print(head(col_sums))
  
  write_tsv(corrected_data, "merged_abundance_batch_corrected.txt")
  cat("處理完成！\n")
}, error = function(e) {
  cat("執行批次效應校正時發生錯誤：\n")
  print(e)
  cat("\nabundance_matrix: ", dim(abundance_matrix), "\n")
  cat("batch_info: ", dim(batch_info), "\n")
  cat("colnames(abundance_matrix)[1:5]: ", head(colnames(abundance_matrix)), "\n")
  cat("batch_info$sample_id[1:5]: ", head(batch_info$sample_id), "\n")
  cat("完全一致？", all(colnames(abundance_matrix) == batch_info$sample_id), "\n")
})
