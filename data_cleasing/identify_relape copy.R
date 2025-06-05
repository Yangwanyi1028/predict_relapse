rm(list = ls())  # 清空工作環境
setwd("C:\\Users\\wanyiyang\\Downloads\\predict relapse20250422\\data_cleasing")

library(readr)
library(dplyr)
library(lubridate)

# 讀取數據
data <- read.csv("../IBD_Outcomes_CD_8May2025_withoutHKID.csv", stringsAsFactors = FALSE)

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

# 計算復發節點的日期 - 確保結果是日期類型
get_progression_date <- function(row) {
  # 各復發條件及其對應日期
  event_list <- list()
  
  if(!is.na(row["change_to_b2"]) && row["change_to_b2"] == 1 && !is.na(row["change_to_b2_date"]))
    event_list$change_to_b2 <- row["change_to_b2_date"]
    
  if(!is.na(row["change_to_b3"]) && row["change_to_b3"] == 1 && !is.na(row["change_to_b3_date"]))
    event_list$change_to_b3 <- row["change_to_b3_date"]
    
  if(!is.na(row["surgery"]) && row["surgery"] == 1 && !is.na(row["surgery_date"]))
    event_list$surgery <- row["surgery_date"]
    
  if(!is.na(row["flare_event"]) && row["flare_event"] == 1 && !is.na(row["flare_med_date"]))
    event_list$flare_event <- row["flare_med_date"]
    
  if(!is.na(row["new_biologic_use"]) && row["new_biologic_use"] == 1 && !is.na(row["new_biologic_date"]))
    event_list$new_biologic_use <- row["new_biologic_date"]
    
  if(!is.na(row["swith_biologic"]) && row["swith_biologic"] == 1 && !is.na(row["switch_biologic_date"]))
    event_list$swith_biologic <- row["switch_biologic_date"]
  
  # 過濾在start_date和end_date內的日期
  valid_dates <- c()
  for(date in event_list) {
    if(!is.na(row["start_date"]) && !is.na(row["end_date"]) && !is.na(date)) {
      if(date >= row["start_date"] && date <= row["end_date"]) {
        valid_dates <- c(valid_dates, date)
      }
    }
  }
  
  # 返回最早的有效日期
  if(length(valid_dates) == 0) return(NA)
  return(min(valid_dates))
}

# 計算復發日期和狀態
data$progression_date <- apply(data, 1, get_progression_date)
data$progression <- ifelse(!is.na(data$progression_date), "prog", "noprog")

# 確保progression_date是Date類型
data$progression_date <- as.Date(data$progression_date, origin = "1970-01-01")

# 打印檢查progression_date
print("檢查progression_date類型:")
print(class(data$progression_date))
print(head(data$progression_date))

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

# 提取有other_baseline_meds的患者ID
patients_with_meds <- data %>%
  filter(!is.na(other_baseline_meds) & other_baseline_meds != "") %>%
  select(sampleID, other_baseline_meds)

write.table(patients_with_meds, "../patients_with_baseline_meds.txt", 
            row.names = FALSE, sep = "\t", quote = FALSE)

# 保存帶有進展狀態的完整數據集
write.csv(data, "../IBD_Outcomes_CD_8May2025_with_progression.csv", 
          row.names = FALSE, quote = TRUE)

# 打印摘要統計信息
print("進展狀態摘要:")
prog_table <- table(data$progression)
print(prog_table)
print(paste("進展百分比:", 
            round(prop.table(prog_table)["prog"] * 100, 2), "%"))

# 打印有基線藥物的患者數量
print(paste("有基線藥物記錄的患者數量:", nrow(patients_with_meds)))

# 打印間隔月數的摘要統計
print("間隔月數摘要統計:")
print(summary(data$interval_months))
