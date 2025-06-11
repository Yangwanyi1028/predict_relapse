rm(list = ls())  # 清空工作環境
setwd("C:\\Users\\wanyiyang\\Downloads\\predict_relapse\\0data_cleasing")

# Load required packages
library(ggplot2)
library(dplyr)
library(gridExtra)
library(scales)

# Read the previously saved data
data <- read.csv("IBD_Outcomes_CD_8May2025_with_progression.csv", stringsAsFactors = FALSE)

# 1. Progression status pie chart
prog_counts <- data.frame(
  status = c("noprog", "prog"),
  count = c(158, 121),
  label = c("No Progression", "Progression")
)
prog_counts$percentage <- paste0(round(prog_counts$count/sum(prog_counts$count) * 100, 1), "%")

pie1 <- ggplot(prog_counts, aes(x = "", y = count, fill = label)) +
  geom_bar(stat = "identity", width = 1, color = "white", size = 1) +
  coord_polar("y", start = 0) +
  theme_minimal() +
  scale_fill_manual(values = c("No Progression" = "#3498db", "Progression" = "#e74c3c")) +
  labs(title = "Progression Status Summary", fill = "Status") +
  geom_text(aes(label = paste0(label, "\n", count, " (", percentage, ")")), 
            position = position_stack(vjust = 0.5), size = 5, fontface = "bold", color = "white") +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        plot.margin = margin(t = 20, r = 20, b = 20, l = 20))

# 2. Progression type distribution pie chart
prog_types <- data.frame(
  type = c("change_to_b2", "change_to_b3", "flare_med", "surgery", 
           "surgery, flare_med", "swith_biologic"),
  count = c(7, 4, 80, 7, 1, 22)
)

# Add English labels
type_mapping <- c(
  "change_to_b2" = "B1 to B2 Progression",
  "change_to_b3" = "B1 to B3 Progression",
  "flare_med" = "Disease Flare Requiring Medication",
  "surgery" = "Surgery Required",
  "surgery, flare_med" = "Surgery and Medication Required",
  "swith_biologic" = "Need to Switch Biologic"
)
prog_types$nice_type <- type_mapping[prog_types$type]

prog_types$percentage <- paste0(round(prog_types$count/sum(prog_types$count) * 100, 1), "%")

# Sort by count
prog_types <- prog_types[order(-prog_types$count),]
prog_types$nice_type <- factor(prog_types$nice_type, levels = prog_types$nice_type)

# Use attractive color palette
pie2 <- ggplot(prog_types, aes(x = "", y = count, fill = nice_type)) +
  geom_bar(stat = "identity", width = 1, color = "white", size = 0.5) +
  coord_polar("y", start = 0) +
  theme_minimal() +
  scale_fill_brewer(palette = "Spectral") +
  labs(title = "Progression Type Distribution", fill = "Progression Type") +
  geom_text(aes(label = paste0(count, "\n(", percentage, ")")), 
            position = position_stack(vjust = 0.5), size = 4, fontface = "bold") +
  theme(legend.title = element_text(size = 12, face = "bold"),
        legend.text = element_text(size = 9),
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        plot.margin = margin(t = 20, r = 20, b = 20, l = 20))

# 3. Follow-up interval months histogram - ONLY for progression samples
# Filter only prog cases
prog_data <- data[data$progression == "prog", ]

# Check and print zero time progression cases
zero_time_cases <- prog_data[prog_data$interval_months == 0, ]
cat("\nNumber of progression cases with Time to Progression = 0 months:", nrow(zero_time_cases), "\n")

if(nrow(zero_time_cases) > 0) {
  cat("Details of zero time progression cases:\n")
  # Select relevant columns for review
  selected_cols <- c("sampleID", "progression_type", "start_date", "progression_date", "interval_months")
  print(zero_time_cases[, selected_cols[selected_cols %in% names(zero_time_cases)]])
  
  # Create a CSV file with these cases for further investigation
  write.csv(zero_time_cases, "Zero_Time_Progression_Cases.csv", row.names = FALSE)
  cat("These cases have been saved to 'Zero_Time_Progression_Cases.csv' for further review\n\n")
}

# Calculate basic statistics for prog cases only
# Option 1: Include zero time cases
mean_interval <- mean(prog_data$interval_months, na.rm = TRUE)
median_interval <- median(prog_data$interval_months, na.rm = TRUE)

# Option 2: Create a version without zero time cases for comparison
if(nrow(zero_time_cases) > 0) {
  prog_data_no_zeros <- prog_data[prog_data$interval_months > 0, ]
  mean_interval_no_zeros <- mean(prog_data_no_zeros$interval_months, na.rm = TRUE)
  median_interval_no_zeros <- median(prog_data_no_zeros$interval_months, na.rm = TRUE)
  
  cat("Statistics excluding zero time cases:\n")
  cat(paste("Mean:", round(mean_interval_no_zeros, 1), "months\n"))
  cat(paste("Median:", round(median_interval_no_zeros, 1), "months\n"))
  cat(paste("Range:", round(min(prog_data_no_zeros$interval_months, na.rm = TRUE), 1), 
            "-", round(max(prog_data_no_zeros$interval_months, na.rm = TRUE), 1), "months\n\n"))
}

# Get max density for positioning text annotations
max_density <- max(density(prog_data$interval_months, na.rm = TRUE)$y, na.rm = TRUE)

hist <- ggplot(prog_data, aes(x = interval_months)) +
  geom_histogram(aes(y = ..density..), binwidth = 3, fill = "#5DADE2", color = "#2471A3", alpha = 0.8) +
  geom_density(color = "#CB4335", size = 1.2) +
  geom_vline(xintercept = mean_interval, color = "#229954", linetype = "dashed", size = 1) +
  geom_vline(xintercept = median_interval, color = "#8E44AD", linetype = "dashed", size = 1) +
  annotate("text", x = mean_interval + 2, y = max_density * 0.9, 
           label = paste0("Mean: ", round(mean_interval, 1), " months"), 
           color = "#229954", size = 5, fontface = "bold") +
  annotate("text", x = median_interval + 2, y = max_density * 0.8, 
           label = paste0("Median: ", round(median_interval, 1), " months"), 
           color = "#8E44AD", size = 5, fontface = "bold") +
  labs(title = "Time to Progression Distribution", 
       subtitle = paste0("Progression cases only (n=", nrow(prog_data), ")",
                        " | Range: ", round(min(prog_data$interval_months, na.rm = TRUE), 1), 
                        "-", round(max(prog_data$interval_months, na.rm = TRUE), 1), " months"),
       x = "Time to Progression (months)", y = "Density") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 12),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12)) +
  scale_x_continuous(breaks = seq(0, max(prog_data$interval_months, na.rm = TRUE) + 5, by = 6))

# Save individual charts
ggsave("Progression_Status_Distribution.png", pie1, width = 8, height = 6, dpi = 300)
ggsave("Progression_Type_Distribution.png", pie2, width = 10, height = 7, dpi = 300)
ggsave("Time_to_Progression_Distribution.png", hist, width = 10, height = 6, dpi = 300)

# Create combined chart
combined_plot <- grid.arrange(
  pie1, pie2, hist, 
  ncol = 2, 
  nrow = 2,
  layout_matrix = rbind(c(1, 2), c(3, 3)),
  heights = c(1, 0.8),
  top = grid::textGrob("IBD Patient Progression Status and Time to Progression Analysis", 
                 gp = grid::gpar(fontface = "bold", fontsize = 20))
)

# Save combined chart
ggsave("IBD_Progression_Analysis_Combined_Report.png", combined_plot, width = 14, height = 10, dpi = 300)

# Output file locations
cat("Charts have been saved to the following locations:\n")
cat("1. Progression_Status_Distribution.png\n")
cat("2. Progression_Type_Distribution.png\n")
cat("3. Time_to_Progression_Distribution.png\n")
cat("4. IBD_Progression_Analysis_Combined_Report.png\n")