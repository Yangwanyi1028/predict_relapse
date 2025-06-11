rm(list = ls())
# 设置工作目录到当前脚本所在目录
setwd('/Users/yangkeyi/Downloads/predict_relapse/2diversity/')
library(vegan)
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(dplyr)

# 读取分组信息，分组列为 progression
# progression 列作为 Group

group_info <- read.csv('IBD_Outcomes_CD_8May2025_with_progression.csv', row.names = 1)
group_info$Group <- as.character(group_info$progression)

# 读取丰度表
species_abundance <- read_tsv('merged_abundance_batch_corrected.txt')
id_mapping <- readxl::read_excel('/Users/yangkeyi/Downloads/predict_relapse/source_data/ID_match_20250602.xlsx')
id_mapping <- id_mapping %>%
  dplyr::select(`SCD No.`, REMARKS) %>%
  dplyr::rename(oldname = `SCD No.`, newname = REMARKS)

# 用id_mapping将丰度表的列名 oldname 替换为 newname
abun_colnames <- colnames(species_abundance)
# 只替换与id_mapping$oldname匹配的列
replace_idx <- match(id_mapping$oldname, abun_colnames)
# 有匹配才替换
abun_colnames[replace_idx[!is.na(replace_idx)]] <- id_mapping$newname[!is.na(replace_idx)]
colnames(species_abundance) <- abun_colnames

# 写出转换列名后的丰度表
write_tsv(species_abundance, 'merged_abundance_batch_corrected_renamed.txt')

# 取 abundance 和 group_info 样本名的交集
abun_samples <- colnames(species_abundance)[-1]
group_samples <- group_info$sampleID
common_samples <- intersect(abun_samples, group_samples)

# 按交集顺序排列 abundance 和 group_info
ordered_indices <- match(common_samples, abun_samples)
species_abundance_ordered <- species_abundance[, c(1, ordered_indices + 1)]
colnames(species_abundance_ordered)[-1] <- common_samples

group_info_filtered <- group_info[group_info$sampleID %in% common_samples, ]

# 转为矩阵，行为分类，列为样本
species_abundance_matrix <- as.matrix(species_abundance_ordered[,-1])
rownames(species_abundance_matrix) <- species_abundance_ordered$taxonomy

# 计算Bray-Curtis距离
bray_curtis_dist <- vegdist(t(species_abundance_matrix), method = "bray")

# 距离矩阵转长表
sample_names <- colnames(species_abundance_matrix)
dist_df <- as.data.frame(as.matrix(bray_curtis_dist))
dist_df$Sample1 <- rownames(dist_df)
dist_df_long <- dist_df %>%
  tidyr::pivot_longer(-Sample1, names_to = "Sample2", values_to = "Distance") %>%
  filter(Sample1 < Sample2)

# 添加分组信息
dist_df_long <- dist_df_long %>%
  left_join(group_info_filtered %>% dplyr::select(sampleID, Group), by = c("Sample1" = "sampleID")) %>%
  left_join(group_info_filtered %>% dplyr::select(sampleID, Group), by = c("Sample2" = "sampleID")) %>%
  dplyr::rename(Group1 = Group.x, Group2 = Group.y) %>%
  mutate(ComparisonType = case_when(
    Group1 == Group2 & Group1 == "noprog" ~ "Within-noprog",
    Group1 == Group2 & Group1 == "prog" ~ "Within-prog",
    Group1 != Group2 ~ "prog-vs-noprog"
  ))

# 统一组间比较命名顺序
# 针对 progression 分组，levels 需要调整为 Within-noprog, Within-prog, prog-vs-noprog

dist_df_long <- dist_df_long %>%
  mutate(
    ComparisonType = case_when(
      Group1 == Group2 & Group1 == "noprog" ~ "Within-noprog",
      Group1 == Group2 & Group1 == "prog" ~ "Within-prog",
      Group1 != Group2 ~ "prog-vs-noprog"
    ),
    ComparisonType = factor(
      ComparisonType,
      levels = c(
        "Within-noprog", "Within-prog", "prog-vs-noprog"
      )
    )
  )

# 检查NA
stopifnot(sum(is.na(dist_df_long$ComparisonType)) == 0)

# 需要比较的组间配对
comparisons <- list(
  c("Within-noprog", "Within-prog"),
  c("Within-noprog", "prog-vs-noprog"),
  c("Within-prog", "prog-vs-noprog")
)

# 绘图
p1 <- ggplot(dist_df_long, aes(x = ComparisonType, y = Distance, fill = ComparisonType)) +
      geom_violin(alpha = 0.3, cex = 0.4, colour = "gray70") +
      geom_boxplot(width = 0.1, outlier.size = 0.01, colour = "black", cex = 0.2, alpha = 0.2) +
      labs(x = NULL, y = "Bray-Curtis Distance") +
      stat_compare_means(
        comparisons = comparisons,
        method = "wilcox.test",
        label = "p.signif",
        step.increase = 0.15,
        symnum.args = list(
          cutpoints = c(0, 0.001, 0.01, 0.05, 1),
          symbols = c("***", "**", "*", "ns")
        )
      ) +
      theme_minimal() +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
        legend.position = "none",
        plot.title = element_text(size = 16),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.y = element_text(size = 12),
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 10),
        panel.grid = element_blank()
      ) +
      scale_fill_manual(values = c(
        "Within-noprog" = "#BCBD22",
        "Within-prog" = "#377EB8",
        "prog-vs-noprog" = "#FF7F00"
      )) +
      coord_cartesian(ylim = c(0, 2))

ggsave('bray-curtis_boxplot.png', width = 8, height = 6, dpi = 300)