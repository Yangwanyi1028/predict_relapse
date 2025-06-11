# 生存分析：IBD复发时间
# 依赖包：survival, survminer
# 如未安装请先运行：install.packages(c('survival', 'survminer'))
library(survival)
library(survminer)
setwd('/Users/yangkeyi/Downloads/predict_relapse/3predict/')
# 读取数据
# 路径适配3predict目录
ibd <- read.csv('IBD_Outcomes_CD_8May2025_with_progression.csv')

# 事件：progression == 'prog' 记为1，否则0
ibd$event <- ifelse(ibd$progression == 'prog', 1, 0)
# 时间：interval_months
ibd$time <- ibd$interval_months

# 构建生存对象
surv_obj <- Surv(time = ibd$time, event = ibd$event)

# 1. 全体Kaplan-Meier曲线
fit_all <- survfit(surv_obj ~ 1)
png('survival_km_all.png', width = 1500, height = 1000, res = 300, bg = 'white')
ggsurvplot(fit_all, data = ibd, risk.table = TRUE, conf.int = TRUE,
           xlab = 'Months', ylab = 'Relapse-free Survival Probability',
           title = 'Kaplan-Meier Curve for IBD Relapse',
           palette = 'jco',
           risk.table.height = 0.2,
           ggtheme = theme_minimal(base_size = 14) + theme(plot.background = element_rect(fill = 'white', color = NA),
                                                          panel.background = element_rect(fill = 'white', color = NA)))
dev.off()
# 2. 按复发状态分组（可选）
fit_group <- survfit(surv_obj ~ progression, data = ibd)
png('survival_km_by_progression.png', width = 3000, height = 2500, res = 300, bg = 'white')
ggsurvplot(fit_group, data = ibd, risk.table = TRUE, conf.int = TRUE,
           pval = TRUE, xlab = 'Months', ylab = 'Relapse-free Survival Probability',
           title = 'Survival by Progression Status',    
           palette = c('#BCBD22', '#377EB8'),
           risk.table.height = 0.2,
           ggtheme = theme_minimal(base_size = 14) + theme(plot.background = element_rect(fill = 'white', color = NA),
                                                          panel.background = element_rect(fill = 'white', color = NA)))
dev.off()

# 3. Cox回归（加入微生物特征）
# 读取微生物特征表，假设第一列为taxonomy，后面为样本，列名与ibd$sampleID一致
abun <- readr::read_tsv('merged_abundance_batch_corrected_renamed.txt')
# 转置并整理为样本为行，特征为列
abun_mat <- as.data.frame(t(as.matrix(abun[,-1])))
colnames(abun_mat) <- make.names(abun$taxonomy)
abun_mat$sampleID <- rownames(abun_mat)  # 用行名作为sampleID

# 合并生存信息和微生物特征
ibd_cox <- merge(ibd, abun_mat, by.x = 'sampleID', by.y = 'sampleID')

# 构建生存对象
surv_obj_cox <- Surv(time = ibd_cox$time, event = ibd_cox$event)

# 仅用前20个丰度特征做示例（防止特征过多报错，可自行调整）
feature_cols <- colnames(abun_mat)[1:min(20, ncol(abun_mat)-1)]
cox_formula <- as.formula(paste('surv_obj_cox ~', paste(feature_cols, collapse = ' + ')))
cox_fit <- coxph(cox_formula, data = ibd_cox)
summary_cox <- summary(cox_fit)

# 输出cox回归结果到txt
sink('cox_microbiome_result.txt')
print(summary_cox)
sink()

# 4. 可视化Cox回归结果（森林图）
library(forestmodel)
# 提取summary_cox结果，筛选HR在合理范围且p<0.05的特征
cox_table <- as.data.frame(summary_cox$coefficients)
cox_table$feature <- rownames(cox_table)
# HR = exp(coef)
cox_table$HR <- exp(cox_table$coef)
# 置信区间
confint_cox <- as.data.frame(summary_cox$conf.int)
cox_table$HR_lower <- confint_cox$`lower .95`
cox_table$HR_upper <- confint_cox$`upper .95`
# 过滤极端值和非显著特征
sel <- (cox_table$HR > 0.01 & cox_table$HR < 100 & cox_table$`Pr(>|z|)` < 0.05)
if(sum(sel) == 0) sel <- (cox_table$HR > 0.01 & cox_table$HR < 100) # 若无显著特征则只按HR过滤
sel_features <- cox_table$feature[sel]

if(length(sel_features) > 0) {
  cox_formula_sel <- as.formula(paste('surv_obj_cox ~', paste(sel_features, collapse = ' + ')))
  cox_fit_sel <- coxph(cox_formula_sel, data = ibd_cox)
  png('cox_microbiome_forest.png', width = 2000, height = 1200, res = 200, bg = 'white')
  forest_model(cox_fit_sel)
  dev.off()
} else {
  cat('无可视化特征，未生成森林图\n')
}

# 输出删失/复发人数
cat('复发人数:', sum(ibd$event == 1), '\n')
cat('删失人数:', sum(ibd$event == 0), '\n')
cat('总样本数:', nrow(ibd), '\n')
