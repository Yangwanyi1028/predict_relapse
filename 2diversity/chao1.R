rm(list = ls())
setwd('/Users/yangkeyi/Downloads/predict_relapse/2diversity/')
library(vegan)
library(tidyverse)
library(ggplot2)
library(ggpubr)

# 读取分组信息，分组列为 progression
meta <- read.csv('IBD_Outcomes_CD_8May2025_with_progression.csv', row.names = 1)
meta$Group <- as.character(meta$progression)

# 读取丰度表
species_abundance <- read_tsv('merged_abundance_batch_corrected.txt')

# 读取id映射表
id_mapping <- readxl::read_excel('../source_data/ID_match_20250602.xlsx')
id_mapping <- id_mapping %>%
  dplyr::select(`SCD No.`, REMARKS) %>%
  dplyr::rename(oldname = `SCD No.`, newname = REMARKS)

# 用id_mapping将丰度表的列名 oldname 替换为 newname
abun_colnames <- colnames(species_abundance)
replace_idx <- match(id_mapping$oldname, abun_colnames)
abun_colnames[replace_idx[!is.na(replace_idx)]] <- id_mapping$newname[!is.na(replace_idx)]
colnames(species_abundance) <- abun_colnames

# 取 abundance 和 meta 样本名的交集
abun_samples <- colnames(species_abundance)[-1]
group_samples <- meta$sampleID
common_samples <- intersect(abun_samples, group_samples)

# 按交集顺序排列 abundance 和 meta
ordered_indices <- match(common_samples, abun_samples)
species_abundance_ordered <- species_abundance[, c(1, ordered_indices + 1)]
colnames(species_abundance_ordered)[-1] <- common_samples
meta_filtered <- meta[meta$sampleID %in% common_samples, ]

# 转为矩阵，行为物种，列为样本
species_abundance_matrix <- as.matrix(species_abundance_ordered[,-1])
rownames(species_abundance_matrix) <- species_abundance_ordered$taxonomy
species_abundance_int <- round(species_abundance_matrix * 1000)

# 计算Chao1 Alpha多样性
# 注意estimateR要求行为样本，列为物种
species_abundance_int_t <- t(species_abundance_int)
chao1_diversity <- estimateR(species_abundance_int_t)

# 整理Chao1结果
chao1_df <- data.frame(
  sampleID = rownames(species_abundance_int_t),
  chao1 = chao1_diversity["S.chao1", ]
)

# 合并分组信息
chao1_df <- merge(chao1_df, meta_filtered, by = "sampleID")

# 绘图
p1 <- ggplot(chao1_df, aes(x = Group, y = chao1, fill = Group)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  stat_compare_means(
    method = "kruskal.test",
    label = "p.signif",
    label.y = max(chao1_df$chao1, na.rm = TRUE) * 0.95
  ) +
  labs(x = 'Group', y = 'Chao1 Diversity') +
  scale_fill_manual(values = c('prog' = '#b22222', 'noprog' = '#90a5a6')) +
  theme_classic() +
  theme(
    legend.position = "right",
    legend.title = element_blank(),
    legend.text = element_text(size = 16),
    plot.title = element_text(hjust = 0.5, size = 16),
    axis.title = element_text(size = 16),
    axis.text = element_text(color = "black", size = 12)
  )
ggsave('chao1_group.png', width = 6, height = 6)