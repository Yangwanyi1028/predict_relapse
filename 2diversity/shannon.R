rm(list = ls())
setwd('/Users/yangkeyi/Downloads/predict_relapse/2diversity/')
library(dplyr)
library(vegan)
library(ggplot2)
library(ggpubr)
library(readxl)

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
# 过滤低丰度物种（保留在>1%样本中出现的物种）
keep <- colSums(species_abundance_int > 0) >= ncol(species_abundance_int) * 0.01
species_abundance_int_filtered <- species_abundance_int[ , keep]
# 转置，行为样本，列为物种
species_abundance_int_t <- t(species_abundance_int_filtered)

# 计算Shannon多样性指数（行为样本，列为物种）
if (nrow(species_abundance_int_t) == 0 || ncol(species_abundance_int_t) == 0) {
  stop("No samples or features available for Shannon diversity calculation!")
}
shannon <- diversity(species_abundance_int_t, index = "shannon")

# 整理Shannon结果
df_shannon <- data.frame(
  sampleID = rownames(species_abundance_int_t),
  Shannon = shannon
)



# 合并分组信息
df_shannon <- merge(df_shannon, meta_filtered, by = "sampleID")

# 绘图
p1 <- ggplot(df_shannon, aes(x = Group, y = Shannon, fill = Group)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  stat_compare_means(
    method = "kruskal.test",
    label = "p.signif",
    label.y = max(df_shannon$Shannon, na.rm = TRUE) * 1.1
  ) +
  labs(x = 'Group', y = 'Shannon Diversity') +
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
ggsave('shannon_group.png', width = 6, height = 6)