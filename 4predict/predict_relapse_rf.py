import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 创建结果保存目录
def create_results_directory():
    """创建保存结果的目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f"results_rf_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    print(f"结果将保存到目录: {result_dir}")
    return result_dir

def load_and_match_data():
    """
    加载并正确匹配特征和标签数据
    """
    print("加载数据...")
    
    # 加载特征数据 (微生物丰度矩阵)
    features_df = pd.read_csv('lefse_diff_abundance_matrix.csv', index_col=0)
    print(f"原始特征数据形状: {features_df.shape}")
    
    # 加载标签数据
    labels_df = pd.read_csv('IBD_Outcomes_CD_8May2025_with_progression.csv')
    print(f"原始标签数据形状: {labels_df.shape}")
    
    # 转置特征矩阵，使样本为行，特征为列
    features_df = features_df.T
    print(f"转置后特征数据形状: {features_df.shape}")
    
    # 确保样本ID为字符串类型，便于匹配
    features_df.index = features_df.index.astype(str)
    labels_df['sampleID'] = labels_df['sampleID'].astype(str)
    
    # 检查样本ID格式
    print(f"特征数据样本ID示例: {list(features_df.index[:5])}")
    print(f"标签数据样本ID示例: {list(labels_df['sampleID'][:5])}")
    
    # 找到共同的样本ID
    common_samples = list(set(features_df.index) & set(labels_df['sampleID']))
    print(f"共同样本数量: {len(common_samples)}")
    
    if len(common_samples) == 0:
        print("警告：没有找到匹配的样本ID！")
        print("检查样本ID格式是否一致...")
        return None, None, None
    
    # 筛选匹配的样本
    X = features_df.loc[common_samples]
    
    # 创建标签字典并筛选
    label_dict = dict(zip(labels_df['sampleID'], labels_df['progression']))
    y = pd.Series([label_dict[sample] for sample in common_samples], index=common_samples)
    
    # 转换为二进制标签 (prog=1, noprog=0)
    y_binary = (y == 'prog').astype(int)
    
    print(f"最终数据形状:")
    print(f"  特征矩阵: {X.shape}")
    print(f"  标签向量: {y_binary.shape}")
    print(f"  类别分布: {y_binary.value_counts()}")
    
    return X, y_binary, common_samples

def preprocess_microbiome_data(X):
    """
    微生物组数据预处理
    """
    print("\n预处理微生物组数据...")
    
    # 1. 移除全零特征
    non_zero_features = (X != 0).any(axis=0)
    X_filtered = X.loc[:, non_zero_features]
    print(f"移除全零特征后: {X_filtered.shape}")
    
    # 2. 移除低方差特征
    feature_variance = X_filtered.var()
    high_var_features = feature_variance > feature_variance.quantile(0.1)
    X_filtered = X_filtered.loc[:, high_var_features]
    print(f"移除低方差特征后: {X_filtered.shape}")
    
    # 3. 对数变换 (添加伪计数避免零值)
    X_log = np.log10(X_filtered + 1e-6)
    
    # 4. 相对丰度标准化
    X_relative = X_filtered.div(X_filtered.sum(axis=1), axis=0)
    
    # 5. CLR变换 (中心对数比变换)
    def clr_transform(data):
        # 添加伪计数
        data_pseudo = data + 1e-6
        # 计算几何平均
        geom_mean = np.exp(np.log(data_pseudo).mean(axis=1))
        # CLR变换
        clr_data = np.log(data_pseudo.div(geom_mean, axis=0))
        return clr_data
    
    X_clr = clr_transform(X_filtered)
    
    return {
        'raw': X_filtered,
        'log': X_log,
        'relative': X_relative,
        'clr': X_clr
    }

def build_random_forest_model(X, y, method_name):
    """
    构建和训练随机森林模型
    """
    print(f"\n构建随机森林模型 ({method_name})...")
    
    # 检查样本数量
    if len(X) < 20:
        print(f"警告：样本数量过少 ({len(X)})，建议至少20个样本")
        test_size = 0.2
    else:
        test_size = 0.25
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"训练集类别分布: {y_train.value_counts()}")
    
    # 参数网格 (简化版，适合小样本)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    # 基础随机森林模型
    rf = RandomForestClassifier(
        random_state=42, 
        class_weight='balanced',  # 处理类别不平衡
        n_jobs=-1
    )
    
    # 网格搜索 (使用较少的CV折数)
    cv_folds = min(5, len(y_train) // 2)  # 确保每折至少有足够样本
    
    grid_search = GridSearchCV(
        rf, param_grid, 
        cv=cv_folds, 
        scoring='roc_auc', 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # 最佳模型
    best_rf = grid_search.best_estimator_
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳CV AUC: {grid_search.best_score_:.3f}")
    
    # 预测
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return best_rf, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, fpr, tpr, roc_auc

def evaluate_model(model, X_test, y_test, y_pred, y_pred_proba, feature_names, method_name, result_dir):
    """
    模型评估和可视化
    """
    print(f"\n{'='*50}")
    print(f"模型评估结果 ({method_name})")
    print(f"{'='*50}")
    
    # 分类报告
    report = classification_report(y_test, y_pred, output_dict=True)
    print("分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 保存分类报告为CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{result_dir}/{method_name}_classification_report.csv")
    
    # AUC分数
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC Score: {auc_score:.3f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Progression', 'Progression'],
                yticklabels=['No Progression', 'Progression'])
    plt.title(f'Confusion Matrix - {method_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # 保存混淆矩阵图
    cm_fig_path = f"{result_dir}/{method_name}_confusion_matrix.png"
    plt.savefig(cm_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存到: {cm_fig_path}")
    
    # 特征重要性分析
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 保存特征重要性表格
    feature_importance.to_csv(f"{result_dir}/{method_name}_feature_importance.csv", index=False)
    
    # 绘制前15个重要特征
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title(f'Top 15 Feature Importances - {method_name}')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    # 保存特征重要性图
    fi_fig_path = f"{result_dir}/{method_name}_feature_importance.png"
    plt.savefig(fi_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征重要性图已保存到: {fi_fig_path}")
    
    return feature_importance, auc_score

def cross_validation_evaluation(X, y, model, method_name):
    """
    交叉验证评估
    """
    print(f"\n交叉验证评估 ({method_name})...")
    
    # 根据样本数量调整CV折数
    cv_folds = min(5, len(y) // 4)
    if cv_folds < 3:
        cv_folds = 3
    
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc')
    print(f"CV AUC scores: {cv_scores}")
    print(f"Mean CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return cv_scores

def plot_roc_curves(results, result_dir):
    """
    绘制所有方法的ROC曲线对比
    """
    plt.figure(figsize=(10, 8))
    
    for method_name, result in results.items():
        plt.plot(result['fpr'], result['tpr'], 
                label=f'{method_name} (AUC = {result["auc"]:.3f})',
                linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存ROC曲线
    roc_fig_path = f"{result_dir}/roc_curves_comparison.png"
    plt.savefig(roc_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC曲线对比图已保存到: {roc_fig_path}")

def main():
    """
    主函数
    """
    # 创建结果保存目录
    result_dir = create_results_directory()
    
    # 1. 加载和匹配数据
    X, y, sample_ids = load_and_match_data()
    
    if X is None:
        print("数据加载失败，请检查文件和样本ID格式")
        return
    
    # 2. 数据预处理
    preprocessed_data = preprocess_microbiome_data(X)
    
    # 3. 训练和评估不同预处理方法的模型
    results = {}
    
    for method_name, X_processed in preprocessed_data.items():
        try:
            print(f"\n{'='*60}")
            print(f"处理方法: {method_name.upper()}")
            print(f"{'='*60}")
            
            # 构建模型
            model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, fpr, tpr, roc_auc = build_random_forest_model(
                X_processed, y, method_name
            )
            
            # 评估模型
            feature_importance, auc_score = evaluate_model(
                model, X_test, y_test, y_pred, y_pred_proba, 
                X_processed.columns, method_name, result_dir
            )
            
            # 交叉验证
            cv_scores = cross_validation_evaluation(X_processed, y, model, method_name)
            
            # 保存结果
            results[method_name] = {
                'model': model,
                'auc': auc_score,
                'cv_scores': cv_scores,
                'feature_importance': feature_importance,
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc
            }
            
            # 保存交叉验证结果
            cv_df = pd.DataFrame({'fold': range(1, len(cv_scores)+1), 'auc': cv_scores})
            cv_df.to_csv(f"{result_dir}/{method_name}_cv_scores.csv", index=False)
            
        except Exception as e:
            print(f"处理 {method_name} 时出错: {str(e)}")
            continue
    
    # 4. 结果对比
    if results:
        print(f"\n{'='*60}")
        print("模型性能对比")
        print(f"{'='*60}")
        
        comparison_df = pd.DataFrame({
            method: {
                'Test AUC': result['auc'],
                'CV AUC Mean': result['cv_scores'].mean(),
                'CV AUC Std': result['cv_scores'].std()
            }
            for method, result in results.items()
        }).T
        
        print(comparison_df.round(3))
        
        # 保存比较结果
        comparison_path = f"{result_dir}/model_comparison.csv"
        comparison_df.to_csv(comparison_path)
        print(f"模型比较结果已保存到: {comparison_path}")
        
        # 绘制ROC曲线对比并保存
        plot_roc_curves(results, result_dir)
        
        # 找出最佳方法
        best_method = max(results.keys(), key=lambda x: results[x]['auc'])
        print(f"\n最佳预处理方法: {best_method}")
        print(f"最佳AUC: {results[best_method]['auc']:.3f}")
        
        # 显示最重要的特征
        print(f"\n{best_method}方法的前10个重要特征:")
        top_features = results[best_method]['feature_importance'].head(10)
        for idx, row in top_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # 保存最佳方法信息
        with open(f"{result_dir}/best_model_info.txt", "w") as f:
            f.write(f"最佳预处理方法: {best_method}\n")
            f.write(f"最佳AUC: {results[best_method]['auc']:.3f}\n\n")
            f.write("前10个重要特征:\n")
            for idx, row in top_features.iterrows():
                f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
    
    print(f"\n所有结果已保存到目录: {result_dir}")
    return results

if __name__ == "__main__":
    results = main()