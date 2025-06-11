rm(list = ls())  # 清空工作環境
setwd("/Users/yangkeyi/Downloads/predict_relapse/0data_cleasing")

library(readr)
library(dplyr)
library(lubridate)
data = readxl::read_excel("../source_data/IBD_Outcomes_CD_8May2025_withoutHKID copy.xlsx",col_names = TRUE)

colnames(data) <- c(
  "no", "sampleID", "study_name", "study_no", "duplicates", "enrol_date",
  "duplicated_prev_excel", "prev_excel_check_period", 
  "start_date", "end_date",
  "change_to_b2","change_to_b2_date", "change_to_b3","change_to_b3_date", "note_for_b3",
  "surgery","surgery_type","surgery_date",
   "flare_med","flare_med_5asa", "flare_med_thiopurines", "flare_med_mtx", "flare_med_steroid", "flare_med_date","note_for_flare_med",
   "new_biologic_use","new_biologic_date",
  "swith_biologic","switch_biologic_date",
   "other_baseline_meds", "lost_to_fu"
)

print(paste("Number of columns in data:", ncol(data)))
print(paste("Number of new column names:", length(colnames(data))))
print("Column names after renaming:")
print(colnames(data))
write.csv(data, "IBD_Outcomes_CD_8May2025_withoutHKID1.csv", row.names = FALSE, quote = TRUE)
rm(list = ls())  # 清空工作環境
setwd("/Users/yangkeyi/Downloads/predict_relapse/0data_cleasing")

library(readr)
library(dplyr)
library(lubridate)

# 讀取數據
data <- read.csv("IBD_Outcomes_CD_8May2025_withoutHKID1.csv", stringsAsFactors = FALSE)

# 定義一個日期轉換函數
convert_date <- function(x) {
  if(is.na(x) || x == "" || x == "NA" || x == "0") return(NA)
  tryCatch({
    as.Date(x, format = "%Y-%m-%d")  # 假設日期格式為 "YYYY-MM-DD"
  }, error = function(e) {
    NA
  })
}

# 需要轉換的日期列
date_cols <- c("enrol_date", "start_date", "end_date", "change_to_b2_date", 
               "change_to_b3_date", "surgery_date", "flare_med_date", 
               "new_biologic_date", "switch_biologic_date")

# 轉換日期列
for(col in date_cols) {
  if(col %in% names(data)) {
    data[[col]] <- as.Date(data[[col]], format = "%Y-%m-%d")
  }
}

# 驗證日期轉換是否正確
print("查看日期轉換結果:")
print(head(data$start_date))
print(head(data$end_date))
print(class(data$start_date))

# 修改後的函數：不僅返回日期，還返回進展類型
get_progression_info <- function(row) {
  # 各復發條件及其對應日期
  event_list <- list()
  
  if(!is.na(row["change_to_b2"]) && row["change_to_b2"] == 1 && !is.na(row["change_to_b2_date"]))
    event_list$change_to_b2 <- row["change_to_b2_date"]
    
  if(!is.na(row["change_to_b3"]) && row["change_to_b3"] == 1 && !is.na(row["change_to_b3_date"]))
    event_list$change_to_b3 <- row["change_to_b3_date"]
    
  if(!is.na(row["surgery"]) && row["surgery"] == 1 && !is.na(row["surgery_date"]))
    event_list$surgery <- row["surgery_date"]
    
  if(!is.na(row["flare_med"]) && row["flare_med"] == 1 && !is.na(row["flare_med_date"]))
    event_list$flare_med <- row["flare_med_date"]
    
  if(!is.na(row["new_biologic_use"]) && row["new_biologic_use"] == 1 && !is.na(row["new_biologic_date"]))
    event_list$new_biologic_use <- row["new_biologic_date"]
    
  if(!is.na(row["swith_biologic"]) && row["swith_biologic"] == 1 && !is.na(row["switch_biologic_date"]))
    event_list$swith_biologic <- row["switch_biologic_date"]
  
  # 過濾在start_date和end_date內的日期
  valid_events <- list()
  for(event_name in names(event_list)) {
    date <- event_list[[event_name]]
    if(!is.na(row["start_date"]) && !is.na(row["end_date"]) && !is.na(date)) {
      if(date >= row["start_date"] && date <= row["end_date"]) {
        valid_events[[event_name]] <- date
      }
    }
  }
  
  # 如果沒有有效進展，返回NA
  if(length(valid_events) == 0) {
    return(list(date = NA, type = NA))
  }
  
  # 找到最早的進展日期
  earliest_date <- min(unlist(valid_events))
  
  # 找到所有在這個最早日期發生的進展類型
  earliest_events <- names(valid_events)[which(unlist(valid_events) == earliest_date)]
  
  # 返回最早日期和對應的進展類型
  return(list(date = earliest_date, type = paste(earliest_events, collapse = ", ")))
}

# 計算復發信息
progression_info <- apply(data, 1, get_progression_info)

# 從結果中提取日期和類型
data$progression_date <- sapply(progression_info, function(x) x$date)
data$progression_type <- sapply(progression_info, function(x) x$type)
data$progression <- ifelse(!is.na(data$progression_date), "prog", "noprog")

# 確保progression_date是Date類型
data$progression_date <- as.Date(data$progression_date, origin = "1970-01-01")

# 打印檢查progression_date和progression_type
print("檢查progression_date類型:")
print(class(data$progression_date))
print(head(data$progression_date))
print("檢查progression_type:")
print(head(data$progression_type))

# 使用更簡單和穩健的方法計算間隔月數
data$interval_months <- NA  # 先初始化

for(i in 1:nrow(data)) {
  start <- data$start_date[i]
  prog <- data$progression_date[i]
  end <- data$end_date[i]
  
  if(is.na(start)) next
  
  # 選擇目標日期
  target <- end
  if(!is.na(prog)) target <- prog
  if(is.na(target)) next
  
  # 確保是日期類型
  start <- as.Date(start)
  target <- as.Date(target)
  
  # 使用interval和time_length計算
  data$interval_months[i] <- floor(time_length(interval(start, target), "months"))
}

# 檢查間隔月數計算結果
print("查看間隔月數計算結果:")
print(head(data.frame(start=data$start_date, prog=data$progression_date, end=data$end_date, months=data$interval_months)))


# 保存帶有進展狀態的完整數據集
write.csv(data, "IBD_Outcomes_CD_8May2025_with_progression.csv", 
          row.names = FALSE, quote = TRUE)

# 打印摘要統計信息
print("進展狀態摘要:")
prog_table <- table(data$progression)
print(prog_table)
print(paste("進展百分比:", 
            round(prop.table(prog_table)["prog"] * 100, 2), "%"))

# 打印進展類型分佈
print("進展類型分佈:")
prog_type_table <- table(data$progression_type[data$progression == "prog"])
print(prog_type_table)


# 打印間隔月數的摘要統計
print("間隔月數摘要統計:")
print(summary(data$interval_months))