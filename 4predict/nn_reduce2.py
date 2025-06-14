import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, f_classif, chi2, mutual_info_classif,
    RFE, RFECV, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LassoCV, ElasticNetCV, LogisticRegressionCV
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

class FeatureReductionAnalyzer:
    """
    特征递减分析器 - 找到最优特征数量
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        self.optimal_features = None
        self.performance_curve = None
        
    def progressive_feature_reduction(self, X, y, 
                                    feature_ranges=None,
                                    selection_method='random_forest',
                                    evaluation_method='neural_network',
                                    cv_folds=5,
                                    auc_drop_threshold=0.1):
        """
        渐进式特征减少分析
        
        Parameters:
        -----------
        X : DataFrame
            特征数据
        y : Series
            标签数据
        feature_ranges : list
            要测试的特征数量列表
        selection_method : str
            特征选择方法 ('random_forest', 'lasso', 'f_classif', 'mutual_info')
        evaluation_method : str
            评估方法 ('neural_network', 'logistic_regression', 'random_forest')
        cv_folds : int
            交叉验证折数
        auc_drop_threshold : float
            AUC下降阈值，超过此值认为性能明显下降
        """
        
        print("开始渐进式特征减少分析...")
        print(f"原始特征数: {X.shape[1]}")
        print(f"样本数: {X.shape[0]}")
        
        # 设置默认的特征数量范围
        if feature_ranges is None:
            max_features = X.shape[1]
            # 创建一个从最大特征数到10的递减序列
            feature_ranges = []
            
            # 密集采样区间
            if max_features > 1000:
                feature_ranges.extend(list(range(max_features, 500, -100)))  # 每100个
                feature_ranges.extend(list(range(500, 100, -50)))           # 每50个
                feature_ranges.extend(list(range(100, 20, -10)))            # 每10个
                feature_ranges.extend(list(range(20, 5, -2)))               # 每2个
            elif max_features > 500:
                feature_ranges.extend(list(range(max_features, 100, -50)))
                feature_ranges.extend(list(range(100, 20, -10)))
                feature_ranges.extend(list(range(20, 5, -2)))
            elif max_features > 100:
                feature_ranges.extend(list(range(max_features, 30, -10)))
                feature_ranges.extend(list(range(20, 5, -2)))
            elif max_features > 10:
                feature_ranges.extend(list(range(max_features, 20, -1)))
                feature_ranges.extend(list(range(20, 1, -1)))
            else:
                feature_ranges.extend(list(range(max_features, 2, -2)))
            
            feature_ranges = sorted(list(set(feature_ranges)), reverse=True)
        
        print(f"将测试的特征数量: {feature_ranges}")
        
        # 存储结果
        results = {
            'n_features': [],
            'auc_mean': [],
            'auc_std': [],
            'auc_scores': [],
            'selected_features': [],
            'feature_importance': []
        }
        
        # 首先获取特征重要性排序
        print("\n获取特征重要性排序...")
        feature_importance = self._get_feature_importance(X, y, method=selection_method)
        
        # 按重要性排序特征
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        feature_names_sorted = [f[0] for f in sorted_features]
        
        print(f"特征重要性计算完成，使用方法: {selection_method}")
        
        # 逐步减少特征数量进行测试
        best_auc = 0
        best_n_features = 0
        auc_drop_detected = False
        
        for n_features in feature_ranges:
            if n_features > len(feature_names_sorted):
                continue
                
            print(f"\n测试特征数: {n_features}")
            
            # 选择前n个最重要的特征
            selected_features = feature_names_sorted[:n_features]
            X_selected = X[selected_features]
            
            # 评估性能
            auc_scores = self._evaluate_performance(
                X_selected, y, 
                method=evaluation_method, 
                cv_folds=cv_folds
            )
            
            auc_mean = np.mean(auc_scores)
            auc_std = np.std(auc_scores)
            
            print(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f}")
            
            # 存储结果
            results['n_features'].append(n_features)
            results['auc_mean'].append(auc_mean)
            results['auc_std'].append(auc_std)
            results['auc_scores'].append(auc_scores)
            results['selected_features'].append(selected_features)
            results['feature_importance'].append({f: feature_importance[f] for f in selected_features})
            
            # 检查是否是最佳性能
            if auc_mean > best_auc:
                best_auc = auc_mean
                best_n_features = n_features
            
            # 检查AUC是否有明显下降
            if len(results['auc_mean']) > 1:
                auc_drop = best_auc - auc_mean
                if auc_drop > auc_drop_threshold:
                    print(f"  检测到AUC明显下降: {auc_drop:.4f} > {auc_drop_threshold}")
                    auc_drop_detected = True
                    break
        
        # 保存结果
        self.results = results
        self.performance_curve = pd.DataFrame({
            'n_features': results['n_features'],
            'auc_mean': results['auc_mean'],
            'auc_std': results['auc_std']
        })
        
        # 确定最优特征数
        if auc_drop_detected:
            # 找到下降前的最后一个高性能点
            optimal_idx = np.argmax(results['auc_mean'])
            self.optimal_features = {
                'n_features': results['n_features'][optimal_idx],
                'auc_mean': results['auc_mean'][optimal_idx],
                'auc_std': results['auc_std'][optimal_idx],
                'selected_features': results['selected_features'][optimal_idx],
                'feature_importance': results['feature_importance'][optimal_idx]
            }
        else:
            # 如果没有检测到明显下降，选择性能最好的
            optimal_idx = np.argmax(results['auc_mean'])
            self.optimal_features = {
                'n_features': results['n_features'][optimal_idx],
                'auc_mean': results['auc_mean'][optimal_idx],
                'auc_std': results['auc_std'][optimal_idx],
                'selected_features': results['selected_features'][optimal_idx],
                'feature_importance': results['feature_importance'][optimal_idx]
            }
        
        print(f"\n分析完成!")
        print(f"最优特征数: {self.optimal_features['n_features']}")
        print(f"最优AUC: {self.optimal_features['auc_mean']:.4f} ± {self.optimal_features['auc_std']:.4f}")
        print(f"特征减少比例: {(X.shape[1] - self.optimal_features['n_features']) / X.shape[1] * 100:.1f}%")
        
        return self.optimal_features
    
    def _get_feature_importance(self, X, y, method='random_forest'):
        """
        获取特征重要性
        """
        if method == 'random_forest':
            clf = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            )
            clf.fit(X, y)
            importance = clf.feature_importances_
            
        elif method == 'extra_trees':
            clf = ExtraTreesClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            )
            clf.fit(X, y)
            importance = clf.feature_importances_
            
        elif method == 'lasso':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf = LassoCV(cv=5, random_state=self.random_state, max_iter=1000)
            clf.fit(X_scaled, y)
            importance = np.abs(clf.coef_)
            
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(X, y)
            importance = selector.scores_
            
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k='all')
            selector.fit(X, y)
            importance = selector.scores_
        
        # 标准化重要性分数
        importance = importance / np.sum(importance)
        
        return dict(zip(X.columns, importance))
    
    def _evaluate_performance(self, X, y, method='neural_network', cv_folds=5):
        """
        评估模型性能
        """
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        auc_scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if method == 'neural_network':
                auc = self._evaluate_neural_network(X_train, X_val, y_train, y_val)
            elif method == 'logistic_regression':
                auc = self._evaluate_logistic_regression(X_train, X_val, y_train, y_val)
            elif method == 'random_forest':
                auc = self._evaluate_random_forest(X_train, X_val, y_train, y_val)
            elif method == 'attention_network':
                auc = self._evaluate_attention_network(X_train, X_val, y_train, y_val)
            elif method == 'autoencoder_classifier':
                auc = self._evaluate_autoencoder_classifier(X_train, X_val, y_train, y_val)
            
            auc_scores.append(auc)
    
        return auc_scores
    
    def _create_autoencoder_model(self, input_dim, encoding_dim=64, dropout_rate=0.2):
        """
        创建自编码器+分类器模型
        用于特征降维和分类的组合架构
        """
        # 编码器
        input_layer = layers.Input(shape=(input_dim,))
        # 编码部分
        encoded = layers.Dense(512, activation='relu')(input_layer)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(dropout_rate)(encoded)
        encoded = layers.Dense(256, activation='relu')(encoded)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(dropout_rate)(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu', name='encoding')(encoded)
        # 解码部分
        decoded = layers.Dense(256, activation='relu')(encoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(dropout_rate)(decoded)
        decoded = layers.Dense(512, activation='relu')(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(dropout_rate)(decoded)
        decoded = layers.Dense(input_dim, activation='linear', name='reconstruction')(decoded)
        # 分类部分
        classifier = layers.Dense(32, activation='relu')(encoded)
        classifier = layers.BatchNormalization()(classifier)
        classifier = layers.Dropout(dropout_rate)(classifier)
        classifier = layers.Dense(16, activation='relu')(classifier)
        classifier = layers.Dropout(dropout_rate)(classifier)
        classification_output = layers.Dense(1, activation='sigmoid', name='classification')(classifier)
        # 创建模型
        model = keras.Model(inputs=input_layer,
                            outputs=[classification_output, decoded])
        return model
    
    def _create_attention_model(self, input_dim, dropout_rate=0.3):
        """
        创建带注意力机制的神经网络
        用于识别重要的微生物特征
        """
        input_layer = layers.Input(shape=(input_dim,))
        # 特征嵌入
        embedded = layers.Dense(256, activation='relu')(input_layer)
        embedded = layers.BatchNormalization()(embedded)
        embedded = layers.Dropout(dropout_rate)(embedded)
        # 注意力机制
        attention_weights = layers.Dense(256, activation='tanh')(embedded)
        attention_weights = layers.Dense(1, activation='softmax')(attention_weights)
        # 应用注意力权重
        attended_features = layers.Multiply()([embedded, attention_weights])
        # 分类层
        x = layers.Dense(128, activation='relu')(attended_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs=input_layer, outputs=output)
        return model

    def _evaluate_attention_network(self, X_train, X_val, y_train, y_val):
        """
        使用带注意力机制的神经网络评估
        """
        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 创建注意力模型
        model = self._create_attention_model(X_train_scaled.shape[1])
        
        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # 训练
        model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=min(32, len(X_train_scaled) // 4),
            validation_data=(X_val_scaled, y_val),
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ],
            verbose=0
        )
        
        # 预测和评估
        y_pred_proba = model.predict(X_val_scaled, verbose=0).flatten()
        auc = roc_auc_score(y_val, y_pred_proba)
        
        # 清理内存
        del model
        tf.keras.backend.clear_session()
        
        return auc
    
    def _evaluate_autoencoder_classifier(self, X_train, X_val, y_train, y_val):
        """
        使用自编码器+分类器模型评估
        """
        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 创建自编码器模型
        model = self._create_autoencoder_model(X_train_scaled.shape[1], encoding_dim=64)
        
        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'classification': 'binary_crossentropy',
                'reconstruction': 'mse'
            },
            loss_weights={
                'classification': 1.0,  # 分类损失权重
                'reconstruction': 0.5   # 重建损失权重
            },
            metrics={
                'classification': 'accuracy'
            }
        )
        
        # 训练
        model.fit(
            X_train_scaled,
            {
                'classification': y_train,
                'reconstruction': X_train_scaled
            },
            epochs=50,
            batch_size=min(32, len(X_train_scaled) // 4),
            validation_data=(
                X_val_scaled,
                {
                    'classification': y_val,
                    'reconstruction': X_val_scaled
                }
            ),
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_classification_loss', patience=10, restore_best_weights=True)
            ],
            verbose=0
        )
        
        # 预测和评估 (只取分类输出)
        y_pred_proba = model.predict(X_val_scaled, verbose=0)[0].flatten()
        auc = roc_auc_score(y_val, y_pred_proba)
        
        # 清理内存
        del model
        tf.keras.backend.clear_session()
        
        return auc

    def _evaluate_neural_network(self, X_train, X_val, y_train, y_val):
        """
        使用神经网络评估
        """
        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 创建简化的神经网络
        model = keras.Sequential([
            layers.Input(shape=(X_train_scaled.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # 训练
        model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=min(32, len(X_train_scaled) // 4),
            validation_data=(X_val_scaled, y_val),
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ],
            verbose=0
        )
        
        # 预测和评估
        y_pred_proba = model.predict(X_val_scaled, verbose=0).flatten()
        auc = roc_auc_score(y_val, y_pred_proba)
        
        # 清理内存
        del model
        tf.keras.backend.clear_session()
        
        return auc
    
    def _evaluate_logistic_regression(self, X_train, X_val, y_train, y_val):
        """
        使用逻辑回归评估
        """
        from sklearn.linear_model import LogisticRegression
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        clf = LogisticRegression(random_state=self.random_state, max_iter=1000)
        clf.fit(X_train_scaled, y_train)
        
        y_pred_proba = clf.predict_proba(X_val_scaled)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        
        return auc
    
    def _evaluate_random_forest(self, X_train, X_val, y_train, y_val):
        """
        使用随机森林评估
        """
        clf = RandomForestClassifier(
            n_estimators=50, 
            random_state=self.random_state,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        y_pred_proba = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        
        return auc
    
    def plot_performance_curve(self, result_dir=None, figsize=(12, 8)):
        """
        绘制性能曲线
        """
        if self.performance_curve is None:
            print("请先运行progressive_feature_reduction")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 性能曲线
        ax1.errorbar(
            self.performance_curve['n_features'], 
            self.performance_curve['auc_mean'],
            yerr=self.performance_curve['auc_std'],
            marker='o', capsize=5, capthick=2
        )
        
        # 标记最优点
        if self.optimal_features:
            ax1.axvline(x=self.optimal_features['n_features'], 
                       color='red', linestyle='--', alpha=0.7,
                       label=f"Optimal: {self.optimal_features['n_features']} features")
            ax1.axhline(y=self.optimal_features['auc_mean'], 
                       color='red', linestyle='--', alpha=0.7)
        
        ax1.set_xlabel('Number of Features')
        ax1.set_ylabel('Cross-Validation AUC')
        ax1.set_title('Feature Reduction Performance Curve')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # AUC分布箱线图（选择几个关键点）
        key_points = []
        if len(self.results['n_features']) > 10:
            indices = np.linspace(0, len(self.results['n_features'])-1, 8, dtype=int)
            key_points = [self.results['n_features'][i] for i in indices]
        else:
            key_points = self.results['n_features']
        
        box_data = []
        box_labels = []
        for n_feat in key_points:
            idx = self.results['n_features'].index(n_feat)
            box_data.append(self.results['auc_scores'][idx])
            box_labels.append(str(n_feat))
        
        ax2.boxplot(box_data, labels=box_labels)
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('AUC Score Distribution')
        ax2.set_title('AUC Distribution at Key Feature Counts')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if result_dir:
            plt.savefig(f"{result_dir}/feature_reduction_curve.png", dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_feature_importance(self, top_n=20, result_dir=None):
        """
        绘制最优特征的重要性
        """
        if not self.optimal_features:
            print("请先运行progressive_feature_reduction")
            return
        
        importance_dict = self.optimal_features['feature_importance']
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # 选择top特征
        top_features = sorted_features[:top_n]
        features, scores = zip(*top_features)
        
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(features))
        
        bars = plt.barh(y_pos, scores, alpha=0.8)
        plt.yticks(y_pos, features)
        plt.xlabel('Feature Importance Score')
        plt.title(f'Top {top_n} Most Important Features (Optimal Set)')
        plt.gca().invert_yaxis()
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if result_dir:
            plt.savefig(f"{result_dir}/optimal_feature_importance.png", dpi=300, bbox_inches='tight')
        # plt.show()
    
    def save_results(self, result_dir):
        """
        保存分析结果
        """
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        # 保存性能曲线数据
        self.performance_curve.to_csv(f"{result_dir}/performance_curve.csv", index=False)
        
        # 保存最优特征信息
        if self.optimal_features:
            optimal_df = pd.DataFrame({
                'feature': list(self.optimal_features['feature_importance'].keys()),
                'importance': list(self.optimal_features['feature_importance'].values())
            }).sort_values('importance', ascending=False)
            
            optimal_df.to_csv(f"{result_dir}/optimal_features.csv", index=False)
            
            # 保存摘要信息
            with open(f"{result_dir}/analysis_summary.txt", "w") as f:
                f.write("Feature Reduction Analysis Summary\n")
                f.write("="*50 + "\n\n")
                f.write(f"Optimal number of features: {self.optimal_features['n_features']}\n")
                f.write(f"Optimal AUC: {self.optimal_features['auc_mean']:.4f} ± {self.optimal_features['auc_std']:.4f}\n")
                f.write(f"Total features tested: {len(self.results['n_features'])}\n")
                f.write(f"Feature reduction ratio: {(len(self.optimal_features['selected_features']) / len(self.optimal_features['selected_features'])) * 100:.1f}%\n")
        
        print(f"结果已保存到: {result_dir}")

def load_and_match_data():
    """
    加载并正确匹配特征和标签数据
    """
    print("加载数据...")
    
    # 加载特征数据 (微生物丰度矩阵)
    features_df = pd.read_csv('lefse_diff_abundance_matrix_sp_only1.csv', index_col=0)
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
    
    # 找到共同的样本ID
    common_samples = list(set(features_df.index) & set(labels_df['sampleID']))
    print(f"共同样本数量: {len(common_samples)}")
    
    if len(common_samples) == 0:
        print("警告：没有找到匹配的样本ID！")
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
    
    # 3. CLR变换 (中心对数比变换)
    def clr_transform(data):
        data_pseudo = data + 1e-6
        geom_mean = np.exp(np.log(data_pseudo).mean(axis=1))
        clr_data = np.log(data_pseudo.div(geom_mean, axis=0))
        return clr_data
    
    X_clr = clr_transform(X_filtered)
    
    return X_clr



def comprehensive_feature_reduction_analysis():
    """
    综合特征减少分析主函数
    """
    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f"feature_reduction_analysis_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    print(f"结果将保存到: {result_dir}")
    
    # 1. 加载数据
    X, y, sample_ids = load_and_match_data()
    if X is None:
        return
    
    # 2. 预处理数据
    X_processed = preprocess_microbiome_data(X)
    
    # 3. 初始化分析器
    analyzer = FeatureReductionAnalyzer(random_state=42)
    
    # 4. 运行多种方法的特征减少分析
    # 在comprehensive_feature_reduction_analysis函数中，修改methods_config列表
    methods_config = [
        # 1. Random Forest 特征选择的所有组合
        {
            'selection_method': 'random_forest',
            'evaluation_method': 'neural_network',
            'name': 'RF_Selection_NN_Eval'
        },
        {
            'selection_method': 'random_forest',
            'evaluation_method': 'logistic_regression',
            'name': 'RF_Selection_LR_Eval'
        },
        {
            'selection_method': 'random_forest',
            'evaluation_method': 'random_forest',
            'name': 'RF_Selection_RF_Eval'
        },
        {
            'selection_method': 'random_forest', 
            'evaluation_method': 'attention_network',
            'name': 'RF_Selection_Attention_Eval'
        },
        
        # 2. Lasso 特征选择的所有组合
        {
            'selection_method': 'lasso',
            'evaluation_method': 'neural_network', 
            'name': 'Lasso_Selection_NN_Eval'
        },
        {
            'selection_method': 'lasso',
            'evaluation_method': 'logistic_regression',
            'name': 'Lasso_Selection_LR_Eval'
        },
        {
            'selection_method': 'lasso',
            'evaluation_method': 'random_forest',
            'name': 'Lasso_Selection_RF_Eval'
        },
        {
            'selection_method': 'lasso',
            'evaluation_method': 'attention_network',
            'name': 'Lasso_Selection_Attention_Eval'
        },
        
        # 3. F-classif 特征选择的所有组合
        {
            'selection_method': 'f_classif',
            'evaluation_method': 'neural_network',
            'name': 'FClassif_Selection_NN_Eval'
        },
        {
            'selection_method': 'f_classif',
            'evaluation_method': 'logistic_regression',
            'name': 'FClassif_Selection_LR_Eval'
        },
        {
            'selection_method': 'f_classif',
            'evaluation_method': 'random_forest',
            'name': 'FClassif_Selection_RF_Eval'
        },
        {
            'selection_method': 'f_classif',
            'evaluation_method': 'attention_network',
            'name': 'FClassif_Selection_Attention_Eval'
        },
        
        # 4. Mutual Info 特征选择的所有组合
        {
            'selection_method': 'mutual_info',
            'evaluation_method': 'neural_network',
            'name': 'MutualInfo_Selection_NN_Eval'
        },
        {
            'selection_method': 'mutual_info',
            'evaluation_method': 'logistic_regression',
            'name': 'MutualInfo_Selection_LR_Eval'
        },
        {
            'selection_method': 'mutual_info',
            'evaluation_method': 'random_forest',
            'name': 'MutualInfo_Selection_RF_Eval'
        },
        {
            'selection_method': 'mutual_info',
            'evaluation_method': 'attention_network',
            'name': 'MutualInfo_Selection_Attention_Eval'
        }
    ]
    
    all_results = {}
    
    for config in methods_config:
        print(f"\n{'='*80}")
        print(f"运行配置: {config['name']}")
        print(f"特征选择方法: {config['selection_method']}")
        print(f"评估方法: {config['evaluation_method']}")
        print(f"{'='*80}")
        
        try:
            # 创建新的分析器实例
            current_analyzer = FeatureReductionAnalyzer(random_state=42)
            
            # 运行分析
            optimal_result = current_analyzer.progressive_feature_reduction(
                X_processed, y,
                selection_method=config['selection_method'],
                evaluation_method=config['evaluation_method'],
                cv_folds=5,
                auc_drop_threshold=0.1  # 1.5%的AUC下降阈值
            )
            
            # 保存结果
            method_dir = os.path.join(result_dir, config['name'])
            os.makedirs(method_dir, exist_ok=True)
            
            current_analyzer.save_results(method_dir)
            current_analyzer.plot_performance_curve(method_dir)
            current_analyzer.plot_feature_importance(top_n=25, result_dir=method_dir)
            
            all_results[config['name']] = {
                'analyzer': current_analyzer,
                'optimal_result': optimal_result,
                'config': config
            }
            
        except Exception as e:
            print(f"配置 {config['name']} 运行失败: {str(e)}")
            continue
    
    # 5. 比较所有方法的结果
    if all_results:
        print(f"\n{'='*80}")
        print("所有方法结果比较")
        print(f"{'='*80}")
        
        comparison_data = []
        for method_name, result in all_results.items():
            optimal = result['optimal_result']
            comparison_data.append({
                'Method': method_name,
                'Optimal_Features': optimal['n_features'],
                'Optimal_AUC': optimal['auc_mean'],
                'AUC_Std': optimal['auc_std'],
                'Reduction_Ratio': (X_processed.shape[1] - optimal['n_features']) / X_processed.shape[1] * 100
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Optimal_AUC', ascending=False)
        
        print("\n方法性能排序:")
        print(comparison_df.round(3))
        
        # 保存比较结果
        comparison_df.to_csv(f"{result_dir}/methods_comparison.csv", index=False)
        
        # 绘制比较图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # AUC比较
        axes[0, 0].bar(comparison_df['Method'], comparison_df['Optimal_AUC'], 
                      yerr=comparison_df['AUC_Std'], capsize=5)
        axes[0, 0].set_title('Optimal AUC by Method')
        axes[0, 0].set_ylabel('AUC')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 特征数比较
        axes[0, 1].bar(comparison_df['Method'], comparison_df['Optimal_Features'])
        axes[0, 1].set_title('Optimal Feature Count by Method')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 减少比例比较
        axes
        # 减少比例比较
        axes[1, 0].bar(comparison_df['Method'], comparison_df['Reduction_Ratio'])
        axes[1, 0].set_title('Feature Reduction Ratio by Method')
        axes[1, 0].set_ylabel('Reduction Ratio (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 效率比较 (AUC vs 特征数)
        scatter = axes[1, 1].scatter(comparison_df['Optimal_Features'], 
                                   comparison_df['Optimal_AUC'],
                                   s=100, alpha=0.7, c=range(len(comparison_df)), 
                                   cmap='viridis')
        axes[1, 1].set_xlabel('Number of Features')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_title('AUC vs Feature Count Trade-off')
        
        # 添加方法标签
        for i, row in comparison_df.iterrows():
            axes[1, 1].annotate(row['Method'], 
                              (row['Optimal_Features'], row['Optimal_AUC']),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(f"{result_dir}/methods_comparison.png", dpi=300, bbox_inches='tight')
        # plt.show()
        
        # 绘制所有方法的性能曲线对比
        plt.figure(figsize=(15, 10))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (method_name, result) in enumerate(all_results.items()):
            analyzer = result['analyzer']
            curve_data = analyzer.performance_curve
            
            plt.errorbar(
                curve_data['n_features'], 
                curve_data['auc_mean'],
                yerr=curve_data['auc_std'],
                marker='o', label=method_name,
                color=colors[i % len(colors)],
                alpha=0.7, capsize=3
            )
            
            # 标记最优点
            optimal = result['optimal_result']
            plt.scatter(optimal['n_features'], optimal['auc_mean'], 
                       color=colors[i % len(colors)], s=100, marker='*',
                       edgecolors='black', linewidth=1)
        
        plt.xlabel('Number of Features')
        plt.ylabel('Cross-Validation AUC')
        plt.title('Feature Reduction Performance Curves - All Methods')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{result_dir}/all_methods_curves.png", dpi=300, bbox_inches='tight')
        # plt.show()
        
        # 推荐最佳方法
        best_method = comparison_df.iloc[0]
        print(f"\n推荐的最佳方法: {best_method['Method']}")
        print(f"最优特征数: {best_method['Optimal_Features']}")
        print(f"最优AUC: {best_method['Optimal_AUC']:.4f} ± {best_method['AUC_Std']:.4f}")
        print(f"特征减少比例: {best_method['Reduction_Ratio']:.1f}%")
        
        # 保存最佳方法的详细信息
        best_result = all_results[best_method['Method']]
        best_features = best_result['optimal_result']['selected_features']
        
        with open(f"{result_dir}/best_method_summary.txt", "w") as f:
            f.write("最佳特征减少方法总结\n")
            f.write("="*50 + "\n\n")
            f.write(f"方法名称: {best_method['Method']}\n")
            f.write(f"特征选择算法: {best_result['config']['selection_method']}\n")
            f.write(f"评估算法: {best_result['config']['evaluation_method']}\n")
            f.write(f"原始特征数: {X_processed.shape[1]}\n")
            f.write(f"最优特征数: {best_method['Optimal_Features']}\n")
            f.write(f"特征减少数量: {X_processed.shape[1] - best_method['Optimal_Features']}\n")
            f.write(f"减少比例: {best_method['Reduction_Ratio']:.1f}%\n")
            f.write(f"最优AUC: {best_method['Optimal_AUC']:.4f} ± {best_method['AUC_Std']:.4f}\n\n")
            f.write("最优特征列表:\n")
            f.write("-" * 30 + "\n")
            for i, feature in enumerate(best_features, 1):
                f.write(f"{i:3d}. {feature}\n")
        
        # 返回最佳特征集
        best_X = X_processed[best_features]
        
        print(f"\n最佳特征集形状: {best_X.shape}")
        print(f"所有结果已保存到: {result_dir}")
        
        return best_X, y, best_result, all_results



def advanced_feature_analysis(X, y, selected_features, result_dir):
    """
    对选定的特征进行高级分析
    """
    print("\n进行高级特征分析...")
    
    # 1. 特征相关性分析
    feature_corr = X[selected_features].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(feature_corr, dtype=bool))
    sns.heatmap(feature_corr, mask=mask, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix (Selected Features)')
    plt.tight_layout()
    plt.savefig(f"{result_dir}/feature_correlation_matrix.png", dpi=300, bbox_inches='tight')
    # plt.show()
    
    # 2. 特征分布分析
    n_features_to_plot = min(12, len(selected_features))
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(selected_features[:n_features_to_plot]):
        # 按类别分组的特征分布
        for class_label in [0, 1]:
            class_data = X[y == class_label][feature]
            axes[i].hist(class_data, alpha=0.7, bins=20, 
                        label=f'Class {class_label}', density=True)
        
        axes[i].set_title(f'{feature}', fontsize=10)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_features_to_plot, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distributions by Class', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{result_dir}/feature_distributions.png", dpi=300, bbox_inches='tight')
    # plt.show()
    
    # 3. 特征重要性稳定性分析
    print("分析特征重要性稳定性...")
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    feature_importance_matrix = []
    
    for train_idx, _ in skf.split(X[selected_features], y):
        X_train_fold = X[selected_features].iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_fold, y_train_fold)
        
        feature_importance_matrix.append(rf.feature_importances_)
    
    # 计算重要性统计
    importance_df = pd.DataFrame(feature_importance_matrix, columns=selected_features)
    importance_stats = pd.DataFrame({
        'mean': importance_df.mean(),
        'std': importance_df.std(),
        'cv': importance_df.std() / importance_df.mean()  # 变异系数
    }).sort_values('mean', ascending=False)
    
    # 绘制重要性稳定性
    plt.figure(figsize=(14, 8))
    top_20_features = importance_stats.head(20).index
    importance_subset = importance_df[top_20_features]
    
    plt.boxplot([importance_subset[col] for col in top_20_features], 
                labels=top_20_features)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance Stability (Top 20 Features)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{result_dir}/feature_importance_stability.png", dpi=300, bbox_inches='tight')
    # plt.show()
    
    # 保存重要性统计
    importance_stats.to_csv(f"{result_dir}/feature_importance_statistics.csv")
    
    print("高级特征分析完成")
    
    return importance_stats


def create_attention_model(input_dim, dropout_rate=0.3):
    """创建带注意力机制的神经网络"""
    input_layer = layers.Input(shape=(input_dim,))
    # 特征嵌入
    embedded = layers.Dense(256, activation='relu')(input_layer)
    embedded = layers.BatchNormalization()(embedded)
    embedded = layers.Dropout(dropout_rate)(embedded)
    # 注意力机制
    attention_weights = layers.Dense(256, activation='tanh')(embedded)
    attention_weights = layers.Dense(1, activation='softmax')(attention_weights)
    # 应用注意力权重
    attended_features = layers.Multiply()([embedded, attention_weights])
    # 分类层
    x = layers.Dense(128, activation='relu')(attended_features)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=input_layer, outputs=output)
    return model


def attention_network_cross_validation(X, y, cv=5, random_state=42):
    """手动实现注意力网络的交叉验证"""
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 清理TensorFlow会话
        tf.keras.backend.clear_session()
        
        # 创建和训练模型
        model = create_attention_model(X_train.shape[1])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # 训练
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=min(32, len(X_train) // 4),
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ],
            verbose=0
        )
        
        # 预测和计算AUC
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        auc = roc_auc_score(y_test, y_pred_proba)
        scores.append(auc)
        
        # 清理模型
        del model
    
    return np.array(scores)



def validate_optimal_features(X_optimal, y, result_dir):
    """
    验证最优特征集的性能
    """
    print("\n验证最优特征集性能...")
    
    # 使用多种算法验证
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    algorithms = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    validation_results = {}
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_optimal), 
        columns=X_optimal.columns, 
        index=X_optimal.index
    )
    
    for alg_name, algorithm in algorithms.items():
        print(f"验证算法: {alg_name}")
        
        # 交叉验证
        cv_scores = cross_val_score(algorithm, X_scaled, y, cv=cv, scoring='roc_auc')
        
        validation_results[alg_name] = {
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        print(f"  AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 单独处理注意力网络
    print(f"验证算法: Attention Network")
    attention_scores = attention_network_cross_validation(X_scaled, y, cv=10, random_state=42)
    
    validation_results['Attention Network'] = {
        'cv_auc_mean': attention_scores.mean(),
        'cv_auc_std': attention_scores.std(),
        'cv_scores': attention_scores
    }
    
    print(f"  AUC: {attention_scores.mean():.4f} ± {attention_scores.std():.4f}")
    
    # 绘制验证结果
    algorithms_names = list(validation_results.keys())
    mean_scores = [validation_results[alg]['cv_auc_mean'] for alg in algorithms_names]
    std_scores = [validation_results[alg]['cv_auc_std'] for alg in algorithms_names]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(algorithms_names, mean_scores, yerr=std_scores, 
                  capsize=5, alpha=0.8, color=['blue', 'green', 'red', 'orange', 'purple'])
    
    plt.ylabel('Cross-Validation AUC')
    plt.title('Performance Validation with Different Algorithms')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, mean_score, std_score in zip(bars, mean_scores, std_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std_score + 0.01,
                f'{mean_score:.3f}±{std_score:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{result_dir}/validation_results.png", dpi=300, bbox_inches='tight')
    # plt.show()
    
    # 保存验证结果
    validation_df = pd.DataFrame({
        alg: {
            'AUC_Mean': result['cv_auc_mean'],
            'AUC_Std': result['cv_auc_std']
        }
        for alg, result in validation_results.items()
    }).T
    
    validation_df.to_csv(f"{result_dir}/validation_results.csv")
    
    print("性能验证完成")
    return validation_results



def main():
    """
    主函数 - 完整的特征减少分析流程
    """
    print("开始完整的特征减少分析流程...")
    print("="*80)
    
    try:
        # 1. 运行综合特征减少分析
        best_X, y, best_result, all_results = comprehensive_feature_reduction_analysis()
        
        # 2. 创建最终结果目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_result_dir = f"final_feature_analysis_{timestamp}"
        os.makedirs(final_result_dir, exist_ok=True)
        
        # 3. 高级特征分析
        selected_features = best_result['optimal_result']['selected_features']
        importance_stats = advanced_feature_analysis(best_X, y, selected_features, final_result_dir)
        
        # 4. 性能验证
        validation_results = validate_optimal_features(best_X, y, final_result_dir)
        
        # 5. 生成最终报告
        with open(f"{final_result_dir}/final_analysis_report.txt", "w") as f:
            f.write("微生物组特征减少分析最终报告\n")
            f.write("="*60 + "\n\n")
            
            f.write("1. 分析概述\n")
            f.write("-"*30 + "\n")
            f.write(f"原始特征数: {len(best_X.columns) + (len(selected_features) if 'selected_features' in locals() else 0)}\n")
            f.write(f"最优特征数: {len(selected_features)}\n")
            f.write(f"特征减少比例: {(1 - len(selected_features)/len(best_X.columns))*100:.1f}%\n")
            f.write(f"最佳方法: {best_result['config']['name']}\n\n")
            
            f.write("2. 性能指标\n")
            f.write("-"*30 + "\n")
            f.write(f"最优AUC: {best_result['optimal_result']['auc_mean']:.4f} ± {best_result['optimal_result']['auc_std']:.4f}\n\n")
            
            f.write("3. 算法验证结果\n")
            f.write("-"*30 + "\n")
            for alg_name, result in validation_results.items():
                f.write(f"{alg_name}: {result['cv_auc_mean']:.4f} ± {result['cv_auc_std']:.4f}\n")
            f.write("\n")
            
            f.write("4. 特征重要性统计\n")
            f.write("-"*30 + "\n")
            f.write("Top 10 最稳定的重要特征:\n")
            top_stable_features = importance_stats.head(10)
            for i, (feature, stats) in enumerate(top_stable_features.iterrows(), 1):
                f.write(f"{i:2d}. {feature}: 重要性={stats['mean']:.4f}, 稳定性={1/stats['cv']:.2f}\n")
        
        print(f"\n分析完成！最终结果保存在: {final_result_dir}")
        print(f"推荐使用的特征数: {len(selected_features)}")
        print(f"预期AUC性能: {best_result['optimal_result']['auc_mean']:.4f}")
        
        return best_X, y, selected_features, all_results
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 运行完整分析
    result = main()


