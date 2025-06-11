import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

class MicrobiomeNeuralNetwork:
    """
    专门用于微生物组数据的神经网络分类器
    """
    
    def __init__(self, input_dim, random_state=42):
        self.input_dim = input_dim
        self.random_state = random_state
        self.model = None
        self.history = None
        self.scaler = None
        
    def create_dense_model(self, dropout_rate=0.3, l2_reg=0.01):
        """
        创建全连接神经网络模型
        适用于微生物组数据的深度学习架构
        """
        model = keras.Sequential([
            # 输入层
            layers.Input(shape=(self.input_dim,)),
            
            # 第一隐藏层 - 降维
            layers.Dense(512, activation='relu', 
                        kernel_regularizer=regularizers.l2(l2_reg),
                        name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            # 第二隐藏层
            layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(l2_reg),
                        name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            # 第三隐藏层
            layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(l2_reg),
                        name='dense_3'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            # 第四隐藏层
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l2(l2_reg),
                        name='dense_4'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            # 输出层
            layers.Dense(1, activation='sigmoid', name='output')
        ])
        
        return model
    
    def create_autoencoder_model(self, encoding_dim=64, dropout_rate=0.2):
        """
        创建自编码器+分类器模型
        用于特征降维和分类的组合架构
        """
        # 编码器
        input_layer = layers.Input(shape=(self.input_dim,))
        
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
        
        decoded = layers.Dense(self.input_dim, activation='linear', name='reconstruction')(decoded)
        
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
    
    def create_attention_model(self, dropout_rate=0.3):
        """
        创建带注意力机制的神经网络
        用于识别重要的微生物特征
        """
        input_layer = layers.Input(shape=(self.input_dim,))
        
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

def create_results_directory():
    """创建保存结果的目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f"results_nn_{timestamp}"
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
    
    # 3. 对数变换 (添加伪计数避免零值)
    X_log = np.log10(X_filtered + 1e-6)
    
    # 4. CLR变换 (中心对数比变换)
    def clr_transform(data):
        data_pseudo = data + 1e-6
        geom_mean = np.exp(np.log(data_pseudo).mean(axis=1))
        clr_data = np.log(data_pseudo.div(geom_mean, axis=0))
        return clr_data
    
    X_clr = clr_transform(X_filtered)
    
    return {
        'log': X_log,
        'clr': X_clr
    }

def train_neural_network(X, y, model_type='dense', result_dir=None):
    """
    训练神经网络模型
    """
    print(f"\n训练神经网络模型 ({model_type})...")
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"训练集大小: {X_train_scaled.shape}")
    print(f"测试集大小: {X_test_scaled.shape}")
    
    # 计算类别权重
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"类别权重: {class_weight_dict}")
    
    # 创建模型
    nn = MicrobiomeNeuralNetwork(input_dim=X_train_scaled.shape[1])
    
    if model_type == 'dense':
        model = nn.create_dense_model(dropout_rate=0.4, l2_reg=0.01)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
        
    elif model_type == 'autoencoder':
        model = nn.create_autoencoder_model(encoding_dim=64, dropout_rate=0.3)
        loss = {'classification': 'binary_crossentropy', 'reconstruction': 'mse'}
        loss_weights = {'classification': 1.0, 'reconstruction': 0.1}
        metrics = {'classification': ['accuracy']}
        
    elif model_type == 'attention':
        model = nn.create_attention_model(dropout_rate=0.3)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    
    # 编译模型
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    if model_type == 'autoencoder':
        model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)
    else:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    print(f"模型参数数量: {model.count_params():,}")
    
    # 回调函数
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    if result_dir:
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            f"{result_dir}/best_model_{model_type}.h5",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks_list.append(model_checkpoint)
    
    # 准备训练数据
    if model_type == 'autoencoder':
        train_data = X_train_scaled
        train_labels = {'classification': y_train, 'reconstruction': X_train_scaled}
        validation_data = (X_test_scaled, {'classification': y_test, 'reconstruction': X_test_scaled})
    else:
        train_data = X_train_scaled
        train_labels = y_train
        validation_data = (X_test_scaled, y_test)
    
    # 训练模型
    history = model.fit(
        train_data, train_labels,
        epochs=200,
        batch_size=min(32, len(X_train_scaled) // 4),
        validation_data=validation_data,
        class_weight=class_weight_dict if model_type != 'autoencoder' else None,
        callbacks=callbacks_list,
        verbose=1
    )
    
    return model, history, scaler, X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_neural_network(model, X_test, y_test, model_type, result_dir, scaler=None):
    """
    评估神经网络模型
    """
    print(f"\n评估神经网络模型 ({model_type})...")
    
    # 预测
    if model_type == 'autoencoder':
        predictions = model.predict(X_test)
        y_pred_proba = predictions[0].flatten()
        reconstruction = predictions[1]
    else:
        y_pred_proba = model.predict(X_test).flatten()
    
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 计算指标
    auc_score = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    print(f"AUC Score: {auc_score:.3f}")
    
    # 分类报告
    report = classification_report(y_test, y_pred, output_dict=True)
    print("分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 保存结果
    if result_dir:
        # 保存分类报告
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"{result_dir}/{model_type}_classification_report.csv")
        
        # 保存预测结果
        results_df = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        })
        results_df.to_csv(f"{result_dir}/{model_type}_predictions.csv", index=False)
    
    return auc_score, fpr, tpr, y_pred, y_pred_proba

def plot_training_history(history, model_type, result_dir):
    """
    绘制训练历史
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失函数
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率
    if model_type == 'autoencoder':
        acc_key = 'classification_accuracy'
        val_acc_key = 'val_classification_accuracy'
    else:
        acc_key = 'accuracy'
        val_acc_key = 'val_accuracy'
    
    if acc_key in history.history:
        axes[0, 1].plot(history.history[acc_key], label='Training Accuracy')
        axes[0, 1].plot(history.history[val_acc_key], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 学习率
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 如果是自编码器，显示重构损失
    if model_type == 'autoencoder' and 'reconstruction_loss' in history.history:
        axes[1, 1].plot(history.history['reconstruction_loss'], label='Training Reconstruction Loss')
        axes[1, 1].plot(history.history['val_reconstruction_loss'], label='Validation Reconstruction Loss')
        axes[1, 1].set_title('Reconstruction Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Reconstruction Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if result_dir:
        plt.savefig(f"{result_dir}/{model_type}_training_history.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_test, y_pred, model_type, result_dir):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Progression', 'Progression'],
                yticklabels=['No Progression', 'Progression'])
    plt.title(f'Confusion Matrix - {model_type.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if result_dir:
        plt.savefig(f"{result_dir}/{model_type}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

def cross_validation_neural_network(X, y, model_type='dense', cv_folds=5):
    """
    神经网络交叉验证
    """
    print(f"\n神经网络交叉验证 ({model_type})...")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{cv_folds}")
        
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # 创建模型
        nn = MicrobiomeNeuralNetwork(input_dim=X_train_scaled.shape[1])
        
        if model_type == 'dense':
            model = nn.create_dense_model(dropout_rate=0.4, l2_reg=0.01)
        elif model_type == 'attention':
            model = nn.create_attention_model(dropout_rate=0.3)
        
        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # 训练模型
        model.fit(
            X_train_scaled, y_train_fold,
            epochs=100,
            batch_size=min(16, len(X_train_scaled) // 4),
            validation_data=(X_val_scaled, y_val_fold),
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            ],
            verbose=0
        )
        
        # 评估
        y_pred_proba = model.predict(X_val_scaled).flatten()
        auc_score = roc_auc_score(y_val_fold, y_pred_proba)
        cv_scores.append(auc_score)
        
        print(f"  Fold {fold + 1} AUC: {auc_score:.3f}")
    
    print(f"CV AUC scores: {cv_scores}")
    print(f"Mean CV AUC: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores) * 2:.3f})")
    
    return cv_scores

def plot_roc_curves_comparison(results, result_dir):
    """
    绘制所有模型的ROC曲线对比
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, result in results.items():
        plt.plot(result['fpr'], result['tpr'],
                label=f'{model_name} (AUC = {result["auc"]:.3f})',
                linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Neural Network Models ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if result_dir:
        plt.savefig(f"{result_dir}/roc_curves_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

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
    
    # 选择最佳预处理方法 (通常CLR变换对微生物组数据效果最好)
    X_processed = preprocessed_data['clr']
    print(f"使用CLR变换后的数据: {X_processed.shape}")
    
    # 3. 训练不同类型的神经网络模型
    model_types = ['dense', 'attention', 'autoencoder']
    results = {}
    
    for model_type in model_types:
        try:
            print(f"\n{'='*60}")
            print(f"训练 {model_type.upper()} 神经网络")
            print(f"{'='*60}")
            
            # 训练模型
            model, history, scaler, X_train, X_test, y_train, y_test = train_neural_network(
                X_processed, y, model_type=model_type, result_dir=result_dir
            )
            
            # 评估模型
            auc_score, fpr, tpr, y_pred, y_pred_proba = evaluate_neural_network(
                model, X_test, y_test, model_type, result_dir, scaler
            )
            
            # 绘制训练历史
            plot_training_history(history, model_type, result_dir)
            
            # 绘制混淆矩阵
            plot_confusion_matrix(y_test, y_pred, model_type, result_dir)
            
            # 交叉验证
            cv_scores = cross_validation_neural_network(X_processed, y, model_type=model_type)
            
            # 保存结果
            results[model_type] = {
                'auc': auc_score,
                'fpr': fpr,
                'tpr': tpr,
                'cv_scores': cv_scores,
                'model': model
            }
            
            # 保存交叉验证结果
            cv_df = pd.DataFrame({'fold': range(1, len(cv_scores)+1), 'auc': cv_scores})
            cv_df.to_csv(f"{result_dir}/{model_type}_cv_scores.csv", index=False)
            
        except Exception as e:
            print(f"训练 {model_type} 模型时出错: {str(e)}")
            continue
    
    # 4. 结果对比
    if results:
        print(f"\n{'='*60}")
        print("神经网络模型性能对比")
        print(f"{'='*60}")
        
        comparison_df = pd.DataFrame({
            model: {
                'Test AUC': result['auc'],
                'CV AUC Mean': np.mean(result['cv_scores']),
                'CV AUC Std': np.std(result['cv_scores'])
            }
            for model, result in results.items()
        }).T
        
        print(comparison_df.round(3))
        
        # 保存比较结果
        comparison_df.to_csv(f"{result_dir}/neural_network_comparison.csv")
        
        # 绘制ROC曲线对比
        plot_roc_curves_comparison(results, result_dir)
        
        # 找出最佳模型
        best_model = max(results.keys(), key=lambda x: results[x]['auc'])
        print(f"\n最佳神经网络模型: {best_model}")
        print(f"最佳AUC: {results[best_model]['auc']:.3f}")
        
        # 保存最佳模型信息
        with open(f"{result_dir}/best_neural_network_info.txt", "w") as f:
            f.write(f"最佳神经网络模型: {best_model}\n")
            f.write(f"最佳AUC: {results[best_model]['auc']:.3f}\n")
            f.write(f"CV AUC: {np.mean(results[best_model]['cv_scores']):.3f} ± {np.std(results[best_model]['cv_scores']):.3f}\n")
    
    print(f"\n所有结果已保存到目录: {result_dir}")
    return results

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, f_classif, chi2, mutual_info_classif,
    RFE, RFECV, SelectFromModel, VarianceThreshold
)
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis, NMF
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LassoCV, ElasticNetCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class MicrobiomeFeatureSelector:
    """
    专门用于微生物组数据的特征选择和降维类
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.selected_features = {}
        self.feature_scores = {}
        self.transformers = {}
        
    def variance_filter(self, X, threshold=0.01):
        """
        方差过滤：移除低方差特征
        适用于微生物组数据中的零丰度特征
        """
        print(f"应用方差过滤 (threshold={threshold})...")
        
        # 计算方差
        selector = VarianceThreshold(threshold=threshold)
        X_filtered = selector.fit_transform(X)
        
        selected_features = X.columns[selector.get_support()]
        
        print(f"原始特征数: {X.shape[1]}")
        print(f"过滤后特征数: {len(selected_features)}")
        print(f"移除特征数: {X.shape[1] - len(selected_features)}")
        
        self.transformers['variance'] = selector
        self.selected_features['variance'] = selected_features
        
        return pd.DataFrame(X_filtered, columns=selected_features, index=X.index)
    
    def prevalence_filter(self, X, min_prevalence=0.1):
        """
        流行度过滤：保留在足够多样本中出现的特征
        这是微生物组数据特有的过滤方法
        """
        print(f"应用流行度过滤 (min_prevalence={min_prevalence})...")
        
        # 计算每个特征的流行度（非零样本比例）
        prevalence = (X > 0).sum(axis=0) / len(X)
        selected_features = X.columns[prevalence >= min_prevalence]
        
        print(f"原始特征数: {X.shape[1]}")
        print(f"过滤后特征数: {len(selected_features)}")
        print(f"平均流行度: {prevalence.mean():.3f}")
        
        self.selected_features['prevalence'] = selected_features
        
        return X[selected_features]
    
    def abundance_filter(self, X, min_abundance=1e-5):
        """
        丰度过滤：保留平均丰度足够高的特征
        """
        print(f"应用丰度过滤 (min_abundance={min_abundance})...")
        
        # 计算平均相对丰度
        mean_abundance = X.mean(axis=0)
        selected_features = X.columns[mean_abundance >= min_abundance]
        
        print(f"原始特征数: {X.shape[1]}")
        print(f"过滤后特征数: {len(selected_features)}")
        print(f"平均丰度范围: {mean_abundance.min():.2e} - {mean_abundance.max():.2e}")
        
        self.selected_features['abundance'] = selected_features
        
        return X[selected_features]
    
    def statistical_filter(self, X, y, method='f_classif', k=500):
        """
        统计检验特征选择
        """
        print(f"应用统计特征选择 ({method}, k={k})...")
        
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        elif method == 'chi2':
            # Chi2需要非负值
            X_positive = X - X.min().min() + 1e-6
            score_func = chi2
            X = X_positive
        
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()]
        feature_scores = selector.scores_
        
        print(f"选择的特征数: {len(selected_features)}")
        print(f"平均得分: {feature_scores[selector.get_support()].mean():.3f}")
        
        self.transformers[f'statistical_{method}'] = selector
        self.selected_features[f'statistical_{method}'] = selected_features
        self.feature_scores[f'statistical_{method}'] = dict(zip(X.columns, feature_scores))
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def tree_based_selection(self, X, y, method='random_forest', n_features=300):
        """
        基于树模型的特征选择
        """
        print(f"应用基于树的特征选择 ({method}, n_features={n_features})...")
        
        if method == 'random_forest':
            estimator = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            )
        elif method == 'extra_trees':
            estimator = ExtraTreesClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # 训练模型获取特征重要性
        estimator.fit(X, y)
        
        # 选择重要性最高的特征
        feature_importance = estimator.feature_importances_
        top_indices = np.argsort(feature_importance)[-n_features:]
        selected_features = X.columns[top_indices]
        
        print(f"选择的特征数: {len(selected_features)}")
        print(f"特征重要性范围: {feature_importance.min():.4f} - {feature_importance.max():.4f}")
        
        self.selected_features[f'tree_{method}'] = selected_features
        self.feature_scores[f'tree_{method}'] = dict(zip(X.columns, feature_importance))
        
        return X[selected_features]
    
    def lasso_selection(self, X, y, alpha=None, max_features=200):
        """
        基于Lasso正则化的特征选择
        """
        print(f"应用Lasso特征选择 (max_features={max_features})...")
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if alpha is None:
            # 使用交叉验证选择最佳alpha
            lasso = LassoCV(cv=5, random_state=self.random_state, max_iter=1000)
        else:
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=alpha, random_state=self.random_state, max_iter=1000)
        
        lasso.fit(X_scaled, y)
        
        # 选择非零系数的特征
        selected_mask = lasso.coef_ != 0
        selected_features = X.columns[selected_mask]
        
        # 如果选择的特征太多，选择系数绝对值最大的
        if len(selected_features) > max_features:
            coef_abs = np.abs(lasso.coef_)
            top_indices = np.argsort(coef_abs)[-max_features:]
            selected_features = X.columns[top_indices]
        
        print(f"选择的特征数: {len(selected_features)}")
        if hasattr(lasso, 'alpha_'):
            print(f"最佳alpha: {lasso.alpha_:.4f}")
        
        self.transformers['lasso'] = (lasso, scaler)
        self.selected_features['lasso'] = selected_features
        self.feature_scores['lasso'] = dict(zip(X.columns, np.abs(lasso.coef_)))
        
        return X[selected_features]
    
    def recursive_feature_elimination(self, X, y, n_features=100, step=0.1):
        """
        递归特征消除
        """
        print(f"应用递归特征消除 (n_features={n_features})...")
        
        # 使用逻辑回归作为基础估计器
        estimator = LogisticRegressionCV(cv=3, random_state=self.random_state, max_iter=1000)
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        selector = RFE(
            estimator=estimator, 
            n_features_to_select=min(n_features, X.shape[1]),
            step=step
        )
        
        X_selected = selector.fit_transform(X_scaled, y)
        selected_features = X.columns[selector.get_support()]
        
        print(f"选择的特征数: {len(selected_features)}")
        
        self.transformers['rfe'] = (selector, scaler)
        self.selected_features['rfe'] = selected_features
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

class MicrobiomeDimensionalityReducer:
    """
    微生物组数据降维类
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.reducers = {}
        
    def pca_reduction(self, X, n_components=50, explained_variance_threshold=0.95):
        """
        主成分分析降维
        """
        print(f"应用PCA降维...")
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 首先确定需要多少成分来解释足够的方差
        pca_full = PCA(random_state=self.random_state)
        pca_full.fit(X_scaled)
        
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_components_needed = np.argmax(cumsum_var >= explained_variance_threshold) + 1
        n_components_final = min(n_components, n_components_needed, X.shape[1])
        
        # 应用PCA
        pca = PCA(n_components=n_components_final, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"原始特征数: {X.shape[1]}")
        print(f"PCA成分数: {n_components_final}")
        print(f"解释的方差比例: {pca.explained_variance_ratio_.sum():.3f}")
        
        self.reducers['pca'] = (pca, scaler)
        
        # 创建成分名称
        component_names = [f'PC{i+1}' for i in range(n_components_final)]
        
        return pd.DataFrame(X_pca, columns=component_names, index=X.index)
    
    def truncated_svd_reduction(self, X, n_components=50):
        """
        截断SVD降维（适用于稀疏数据）
        """
        print(f"应用截断SVD降维 (n_components={n_components})...")
        
        n_components_final = min(n_components, X.shape[1] - 1)
        
        svd = TruncatedSVD(n_components=n_components_final, random_state=self.random_state)
        X_svd = svd.fit_transform(X)
        
        print(f"原始特征数: {X.shape[1]}")
        print(f"SVD成分数: {n_components_final}")
        print(f"解释的方差比例: {svd.explained_variance_ratio_.sum():.3f}")
        
        self.reducers['svd'] = svd
        
        component_names = [f'SVD{i+1}' for i in range(n_components_final)]
        
        return pd.DataFrame(X_svd, columns=component_names, index=X.index)
    
    def nmf_reduction(self, X, n_components=30):
        """
        非负矩阵分解（适用于微生物组数据的非负特性）
        """
        print(f"应用NMF降维 (n_components={n_components})...")
        
        # 确保数据非负
        X_positive = X - X.min().min() + 1e-6
        
        n_components_final = min(n_components, X.shape[1])
        
        nmf = NMF(n_components=n_components_final, random_state=self.random_state, max_iter=500)
        X_nmf = nmf.fit_transform(X_positive)
        
        print(f"原始特征数: {X.shape[1]}")
        print(f"NMF成分数: {n_components_final}")
        print(f"重构误差: {nmf.reconstruction_err_:.3f}")
        
        self.reducers['nmf'] = nmf
        
        component_names = [f'NMF{i+1}' for i in range(n_components_final)]
        
        return pd.DataFrame(X_nmf, columns=component_names, index=X.index)

def evaluate_feature_selection_methods(X, y, methods_results, cv_folds=5):
    """
    评估不同特征选择方法的效果
    """
    print("\n评估特征选择方法...")
    
    results = {}
    
    for method_name, X_selected in methods_results.items():
        print(f"评估 {method_name}...")
        
        # 使用简单的逻辑回归评估
        from sklearn.linear_model import LogisticRegression
        
        clf = LogisticRegression(random_state=42, max_iter=1000)
        
        # 交叉验证
        cv_scores = cross_val_score(
            clf, X_selected, y, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        results[method_name] = {
            'n_features': X_selected.shape[1],
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        print(f"  特征数: {X_selected.shape[1]}")
        print(f"  CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return results

def plot_feature_importance(feature_scores, method_name, top_n=20, result_dir=None):
    """
    绘制特征重要性图
    """
    if not feature_scores:
        return
    
    # 选择top特征
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    features, scores = zip(*top_features)
    
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(features))
    
    plt.barh(y_pos, scores)
    plt.yticks(y_pos, features)
    plt.xlabel('Feature Importance Score')
    plt.title(f'Top {top_n} Features - {method_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if result_dir:
        plt.savefig(f"{result_dir}/feature_importance_{method_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_dimensionality_reduction_comparison(reduction_results, result_dir=None):
    """
    绘制降维方法比较
    """
    methods = list(reduction_results.keys())
    n_components = [result['n_features'] for result in reduction_results.values()]
    cv_scores = [result['cv_auc_mean'] for result in reduction_results.values()]
    cv_errors = [result['cv_auc_std'] for result in reduction_results.values()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 特征数比较
    ax1.bar(methods, n_components, alpha=0.7)
    ax1.set_ylabel('Number of Features')
    ax1.set_title('Feature Count by Method')
    ax1.tick_params(axis='x', rotation=45)
    
    # 性能比较
    ax2.bar(methods, cv_scores, yerr=cv_errors, alpha=0.7, capsize=5)
    ax2.set_ylabel('Cross-Validation AUC')
    ax2.set_title('Performance by Method')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if result_dir:
        plt.savefig(f"{result_dir}/dimensionality_reduction_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def comprehensive_feature_reduction_pipeline(X, y, result_dir=None):
    """
    综合特征降维流水线
    """
    print("开始综合特征降维流水线...")
    
    # 初始化选择器和降维器
    selector = MicrobiomeFeatureSelector()
    reducer = MicrobiomeDimensionalityReducer()
    
    # 存储所有结果
    all_results = {}
    
    # 1. 基础过滤
    print("\n" + "="*50)
    print("第一步：基础过滤")
    print("="*50)
    
    # 方差过滤
    X_var_filtered = selector.variance_filter(X, threshold=0.001)
    
    # 流行度过滤
    X_prev_filtered = selector.prevalence_filter(X_var_filtered, min_prevalence=0.05)
    
    # 丰度过滤
    X_basic_filtered = selector.abundance_filter(X_prev_filtered, min_abundance=1e-6)
    
    print(f"基础过滤后特征数: {X_basic_filtered.shape[1]}")
    
    # 2. 特征选择方法
    print("\n" + "="*50)
    print("第二步：特征选择")
    print("="*50)
    
    feature_selection_results = {}
    
    # 统计方法
    try:
        X_f_classif = selector.statistical_filter(X_basic_filtered, y, method='f_classif', k=200)
        feature_selection_results['F-statistic'] = X_f_classif
    except Exception as e:
        print(f"F-statistic选择失败: {e}")
    
    try:
        X_mutual_info = selector.statistical_filter(X_basic_filtered, y, method='mutual_info', k=200)
        feature_selection_results['Mutual Info'] = X_mutual_info
    except Exception as e:
        print(f"Mutual info选择失败: {e}")
    
    # 树方法
    try:
        X_rf = selector.tree_based_selection(X_basic_filtered, y, method='random_forest', n_features=150)
        feature_selection_results['Random Forest'] = X_rf
    except Exception as e:
        print(f"Random Forest选择失败: {e}")
    
    # Lasso方法
    try:
        X_lasso = selector.lasso_selection(X_basic_filtered, y, max_features=100)
        feature_selection_results['Lasso'] = X_lasso
    except Exception as e:
        print(f"Lasso选择失败: {e}")
    
    # RFE方法
    try:
        X_rfe = selector.recursive_feature_elimination(X_basic_filtered, y, n_features=80)
        feature_selection_results['RFE'] = X_rfe
    except Exception as e:
        print(f"RFE选择失败: {e}")
    
    # 3. 降维方法
    print("\n" + "="*50)
    print("第三步：降维方法")
    print("="*50)
    
    dimensionality_reduction_results = {}
    
    # PCA
    try:
        X_pca = reducer.pca_reduction(X_basic_filtered, n_components=50)
        dimensionality_reduction_results['PCA'] = X_pca
    except Exception as e:
        print(f"PCA降维失败: {e}")
    
    # SVD
    try:
        X_svd = reducer.truncated_svd_reduction(X_basic_filtered, n_components=40)
        dimensionality_reduction_results['SVD'] = X_svd
    except Exception as e:
        print(f"SVD降维失败: {e}")
    
    # NMF
    try:
        X_nmf = reducer.nmf_reduction(X_basic_filtered, n_components=30)
        dimensionality_reduction_results['NMF'] = X_nmf
    except Exception as e:
        print(f"NMF降维失败: {e}")
    
    # 4. 评估所有方法
    print("\n" + "="*50)
    print("第四步：方法评估")
    print("="*50)
    
    # 合并所有结果
    all_methods = {**feature_selection_results, **dimensionality_reduction_results}
    
    # 添加原始数据作为基准
    all_methods['Original'] = X
    all_methods['Basic Filtered'] = X_basic_filtered
    
    # 评估所有方法
    evaluation_results = evaluate_feature_selection_methods(X, y, all_methods)
    
    # 5. 结果可视化和保存
    print("\n" + "="*50)
    print("第五步：结果总结")
    print("="*50)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(evaluation_results).T
    results_df = results_df.sort_values('cv_auc_mean', ascending=False)
    
    print("所有方法性能排序:")
    print(results_df[['n_features', 'cv_auc_mean', 'cv_auc_std']].round(3))
    
    if result_dir:
        # 保存结果
        results_df.to_csv(f"{result_dir}/feature_reduction_comparison.csv")
        
        # 绘制特征重要性图
        for method_name in selector.feature_scores:
            plot_feature_importance(
                selector.feature_scores[method_name], 
                method_name, 
                result_dir=result_dir
            )
        
        # 绘制比较图
        plot_dimensionality_reduction_comparison(evaluation_results, result_dir)
    
    # 推荐最佳方法
    best_method = results_df.index[0]
    best_auc = results_df.loc[best_method, 'cv_auc_mean']
    best_n_features = results_df.loc[best_method, 'n_features']
    
    print(f"\n推荐的最佳方法: {best_method}")
    print(f"特征数: {best_n_features}")
    print(f"CV AUC: {best_auc:.3f}")
    
    return {
        'best_method': best_method,
        'best_data': all_methods[best_method],
        'all_results': evaluation_results,
        'selector': selector,
        'reducer': reducer
    }

# 主函数示例
def main():
    """
    特征降维主函数
    """
    from datetime import datetime
    import os
    
    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f"feature_reduction_results_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    # 加载数据（使用你之前的数据加载函数）
    X, y, sample_ids = load_and_match_data()
    
    # 运行综合降维流水线
    results = comprehensive_feature_reduction_pipeline(X, y, result_dir)
    
    print(f"所有结果已保存到: {result_dir}")
    # return results

if __name__ == "__main__":
    main()
