# 清空环境变量
rm(list=ls())

# 加载包
library(ggplot2)
library(ggExtra)
library(vegan)
library(ggthemes)
library(readr)

# 设置工作路径
setwd('/Users/yangkeyi/Downloads/predict_relapse/2diversity/')

# 加载丰度表
species_abundance <- read_tsv('merged_abundance_batch_corrected.txt')
# 加载分组表
meta <- read.csv('IBD_Outcomes_CD_8May2025_with_progression.csv', row.names = 1)
meta$Group <- as.character(meta$progression)

# 取交集，保证样本顺序一致
abun_samples <- colnames(species_abundance)[-1]
group_samples <- meta$sampleID
common_samples <- intersect(abun_samples, group_samples)
ordered_indices <- match(common_samples, abun_samples)
species_abundance_ordered <- species_abundance[, c(1, ordered_indices + 1)]
colnames(species_abundance_ordered)[-1] <- common_samples
meta_filtered <- meta[meta$sampleID %in% common_samples, ]

# 转为矩阵，行为分类，列为样本
spe <- as.matrix(species_abundance_ordered[,-1])
rownames(spe) <- species_abundance_ordered$taxonomy
spe <- t(spe)
# # 计算所有物种相对丰度
# spe <- spe / rowSums(spe)
data2 <- spe

# 进行PCoA分析
bray <- vegdist(data2 <- spe, method = 'bray', na.rm = TRUE)
bray <- as.matrix(bray)
write.table(bray, 'bray-crutis.txt', sep = '\t')

# PCoA
pcoa <- cmdscale(bray, k = 3, eig = TRUE)
pcoa_data <- data.frame(pcoa$points)
pcoa_data$Sample_ID <- rownames(pcoa_data)
names(pcoa_data)[1:3] <- paste0('PCoA', 1:3)
eig = pcoa$eig
eig_percent <- round(pcoa$eig / sum(eig) * 100, 1)

# 合并分组信息
pcoa_result <- merge(pcoa_data, meta_filtered, by.x = 'Sample_ID', by.y = 'sampleID')

# PERMANOVA
adonis_res <- adonis2(data2 ~ Group, data = meta_filtered, permutations = 999, method = 'bray')
dune_adonis <- paste0('adonis R2: ', round(adonis_res$R2, 2), '; P-value: ', adonis_res$`Pr(>F)`)

# 绘图
p <- ggplot(pcoa_result, aes(x = PCoA1, y = PCoA2, color = Group)) +
  geom_point(aes(color = Group), size = 5, alpha = 0.7) +
  labs(
    x = paste('PCoA 1 (', eig_percent[1], '%)', sep = ''),
    y = paste('PCoA 2 (', eig_percent[2], '%)', sep = ''),
    caption = dune_adonis
  ) +
  scale_colour_manual(values = c('#b22222',  '#90a5a6')) + #'#da8953',
  theme_classic() +
  theme(
    legend.position = c(0.86, 0.07),
    legend.title = element_blank(),
    legend.text = element_text(size = 20),
    plot.title = element_text(hjust = 0.5, size = 20),
    plot.caption = element_text(size = 26),
    axis.title = element_text(size = 26),
    axis.text = element_text(color = 'black', size = 26),
    panel.border = element_rect(color = 'black', fill = NA)
  )
p <- p + stat_ellipse(
  data = pcoa_result,
  geom = 'polygon',
  level = 0.9,
  linetype = 2,
  linewidth = 0.5,
  aes(fill = Group),
  alpha = 0.3,
  show.legend = TRUE
) +
  scale_fill_manual(values = c('#b22222',  '#90a5a6'))

png(filename = 'bray_pcoa_group.png', width = 10, height = 10, res = 300, units = 'in')
ggMarginal(
  p,
  type = c('density'),
  margins = 'both',
  size = 3.5,
  groupColour = FALSE,
  groupFill = TRUE
)
dev.off()