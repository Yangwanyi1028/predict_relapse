# =======================================================================
# 微生物组丰度数据(Abundance Data)处理脚本
# 功能：ID映射、文件列名清理、样本选择与数据提取
# =======================================================================

# ------------- 基本设置 -------------
library(readxl)
rm(list = ls())
setwd("/Users/yangkeyi/Downloads/predict_relapse")


# 读取样本分组信息
group_info <- read.csv("0data_cleasing/IBD_Outcomes_CD_8May2025_with_progression.csv", stringsAsFactors = FALSE)


# 将study_name列中的空白行填充为"Longitude"
group_info$study_name[group_info$study_name == "" | is.na(group_info$study_name)] <- "Longitude"
table(group_info$study_name)

# 筛选IBDR、Longitude和ENIGMA研究的样本ID
ibdr_samples <- group_info$sampleID[group_info$study_name == "IBDR"]
longitude_samples <- group_info$sampleID[group_info$study_name == "Longitude"]
enigma_samples <- group_info$sampleID[group_info$study_name == "ENIGMA"]

# 输出IBDR样本ID到txt文件
write.table(ibdr_samples, "1metaphlan_data/IBDR_sample_ids.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)

# 输出Longitude样本ID到txt文件
write.table(longitude_samples, "1metaphlan_data/Longitude_sample_ids.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)

# 输出ENIGMA样本ID到txt文件
write.table(enigma_samples, "1metaphlan_data/ENIGMA_sample_ids.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)
# ------------- 读取数据文件 -------------
# 读取对照表 - ID映射关系
id_map <- read_excel("source_data/ID_match_20250602.xlsx", 
                     col_names = TRUE)

region_ids <- id_map$study_id
SCD_ids <- id_map$`SCD No.`

# --------- 添加ENIGMA样本ID映射功能 ---------
# 找到ENIGMA样本ID在id_map中对应的SCD No.
enigma_scd_ids <- c()
enigma_matching_records <- data.frame(study_id=character(), scd_id=character(), stringsAsFactors=FALSE)

for(sample_id in enigma_samples) {
  idx <- which(region_ids == sample_id)
  if(length(idx) > 0) {
    scd_id <- SCD_ids[idx]
    if(!is.na(scd_id)) {  # 只添加非NA的SCD ID
      enigma_scd_ids <- c(enigma_scd_ids, scd_id)
      enigma_matching_records <- rbind(enigma_matching_records, data.frame(study_id=sample_id, scd_id=scd_id, stringsAsFactors=FALSE))
    }
  }
}

# 再次确保没有NA值
enigma_scd_ids <- enigma_scd_ids[!is.na(enigma_scd_ids)]
enigma_matching_records <- enigma_matching_records[!is.na(enigma_matching_records$scd_id), ]

# 输出ENIGMA匹配结果
write.table(enigma_scd_ids, "1metaphlan_data/ENIGMA_SCD_ids.txt", row.names = FALSE, col.names = FALSE, quote = FALSE)

# 输出完整的ENIGMA匹配记录
write.csv(enigma_matching_records, "1metaphlan_data/ENIGMA_ID_mapping.csv", row.names = FALSE)

# 打印ENIGMA匹配情况
cat("ENIGMA样本总数:", length(enigma_samples), "\n")
cat("成功匹配到SCD No.的ENIGMA样本数:", length(enigma_scd_ids), "\n")