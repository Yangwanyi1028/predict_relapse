rm(list = ls())  # 清空工作環境
setwd("C:\\Users\\wanyiyang\\Downloads\\predict relapse20250422\\data_cleasing")

library(readr)
library(dplyr)
library(lubridate)

# 讀取數據
data <- read.csv("../IBD_Outcomes_CD_8May2025_withoutHKID.csv", stringsAsFactors = FALSE)

# 定義一個簡單的日期轉換函數
convert_date <- function(x) {
  if(is.na(x) || x == "" || x == "NA" || x == "0") return(NA)
  as.Date(x, format = "%d-%m-%Y")
}

# 需要轉換的日期列
date_cols <- c("enrol_date", "start_date", "end_date", "change_to_b2_date", 
               "change_to_b3_date", "surgery_date", "flare_med_date", 
               "new_biologic_date", "switch_biologic_date")

for(col in date_cols) {
  if(col %in% names(data)) {
    data[[col]] <- sapply(data[[col]], convert_date)
  }
}

# 計算復發節點的日期
get_progression_date <- function(row) {
  # 各復發條件及其對應日期
  event_list <- list(
    change_to_b2 = ifelse(!is.na(row["change_to_b2"]) && row["change_to_b2"] == 1, row["change_to_b2_date"], NA),
    change_to_b3 = ifelse(!is.na(row["change_to_b3"]) && row["change_to_b3"] == 1, row["change_to_b3_date"], NA),
    surgery = ifelse(!is.na(row["surgery"]) && row["surgery"] == 1, row["surgery_date"], NA),
    flare_event = ifelse(!is.na(row["flare_event"]) && row["flare_event"] == 1, row["flare_med_date"], NA),
    new_biologic_use = ifelse(!is.na(row["new_biologic_use"]) && row["new_biologic_use"] == 1, row["new_biologic_date"], NA),
    swith_biologic = ifelse(!is.na(row["swith_biologic"]) && row["swith_biologic"] == 1, row["switch_biologic_date"], NA)
  )
  # 過濾在start_date和end_date內的日期
  valid_dates <- sapply(event_list, function(d) {
    if(is.na(d)) return(NA)
    if(!is.na(row["start_date"]) && !is.na(row["end_date"])) {
      if(d >= row["start_date"] && d <= row["end_date"]) return(d)
    }
    return(NA)
  })
  # 返回最早的有效日期
  valid_dates <- valid_dates[!is.na(valid_dates)]
  if(length(valid_dates) == 0) return(NA)
  return(min(valid_dates))
}

# 計算復發日期和狀態
data$progression_date <- apply(data, 1, get_progression_date)
data$progression <- ifelse(!is.na(data$progression_date), "prog", "noprog")

# 計算復發到start_date的間隔月數（若無復發則到end_date）
data$interval_months <- mapply(function(start, end) {
  if(is.na(start) || is.na(end)) return(NA)
  interval(start, end) %/% months(1) + (interval(start, end) %% months(1))/30.44
}, data$start_date, ifelse(!is.na(data$progression_date), data$progression_date, data$end_date))

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
