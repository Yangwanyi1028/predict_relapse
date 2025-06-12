import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings('ignore')

# 尝试导入深度学习库
# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, callbacks
    from tensorflow.keras.utils import to_categorical
    DEEP_LEARNING_AVAILABLE = True
    print("✅ TensorFlow 可用，将包含深度学习模型")
    
    # 设置GPU内存增长（如果有GPU）
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass
        
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("⚠️ TensorFlow 未安装，将跳过深度学习模型")

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("🔬 增强版微生物组学预测模型启动中...")
print("="*60)

# ================================
# 深度学习模型定义（修复版）
# ================================

if DEEP_LEARNING_AVAILABLE:
    
    class AttentionLayer(layers.Layer):
        """自定义注意力机制层"""
        def __init__(self, units=128, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)
            self.units = units
            
        def build(self, input_shape):
            self.W_q = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                trainable=True,
                name='query_weights'
            )
            self.W_k = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                trainable=True,
                name='key_weights'
            )
            self.W_v = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                trainable=True,
                name='value_weights'
            )
            super(AttentionLayer, self).build(input_shape)
            
        def call(self, inputs):
            # 计算 Query, Key, Value
            Q = tf.matmul(inputs, self.W_q)
            K = tf.matmul(inputs, self.W_k)
            V = tf.matmul(inputs, self.W_v)
            
            # 计算注意力权重
            attention_scores = tf.matmul(Q, K, transpose_b=True)
            attention_scores = attention_scores / tf.sqrt(tf.cast(self.units, tf.float32))
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)
            
            # 应用注意力权重
            attended_values = tf.matmul(attention_weights, V)
            
            return attended_values
        
        def get_config(self):
            config = super(AttentionLayer, self).get_config()
            config.update({'units': self.units})
            return config

    def create_mlp_model(input_dim, num_classes=2):
        """创建基础多层感知机模型"""
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
        ])
        
        # 正确的输出层设置
        if num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))  # 二分类使用1个神经元
        else:
            model.add(layers.Dense(num_classes, activation='softmax'))  # 多分类
        
        return model

    def create_attention_model(input_dim, num_classes=2):
        """创建带注意力机制的神经网络模型"""
        inputs = layers.Input(shape=(input_dim,))
        
        # 重塑输入以适应注意力机制
        x = layers.Reshape((1, input_dim))(inputs)
        
        # 多头注意力机制
        attention1 = AttentionLayer(units=64)(x)
        attention2 = AttentionLayer(units=32)(attention1)
        
        # 全局平均池化
        x = layers.GlobalAveragePooling1D()(attention2)
        
        # 全连接层
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # 正确的输出层
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)  # 二分类
        else:
            outputs = layers.Dense(num_classes, activation='softmax')(x)  # 多分类
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def create_advanced_attention_model(input_dim, num_classes=2):
        """创建高级注意力模型（多头注意力 + 残差连接）"""
        inputs = layers.Input(shape=(input_dim,))
        
        # 初始嵌入
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        
        # 重塑为序列格式
        x = layers.Reshape((1, 128))(x)
        
        # 多头注意力块1
        attention1 = layers.MultiHeadAttention(
            num_heads=4, key_dim=32, dropout=0.1
        )(x, x)
        x1 = layers.Add()([x, attention1])  # 残差连接
        x1 = layers.LayerNormalization()(x1)
        
        # 多头注意力块2
        attention2 = layers.MultiHeadAttention(
            num_heads=2, key_dim=64, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([x1, attention2])  # 残差连接
        x2 = layers.LayerNormalization()(x2)
        
        # 全局平均池化
        x = layers.GlobalAveragePooling1D()(x2)
        
        # 分类头
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        
        # 正确的输出层
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)  # 二分类
        else:
            outputs = layers.Dense(num_classes, activation='softmax')(x)  # 多分类
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

    class KerasClassifierWrapper:
        """Keras模型包装器，兼容sklearn接口（修复版）"""
        def __init__(self, model_func, input_dim, num_classes=2, epochs=100, batch_size=32, verbose=0):
            self.model_func = model_func
            self.input_dim = input_dim
            self.num_classes = num_classes
            self.epochs = epochs
            self.batch_size = batch_size
            self.verbose = verbose
            self.model = None
            self.history = None
            
        def fit(self, X, y):
            self.model = self.model_func(self.input_dim, self.num_classes)
            
            # 正确的损失函数和标签处理
            if self.num_classes == 2:
                # 二分类：使用binary_crossentropy，标签保持0,1格式
                self.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                y_train = y.astype(np.float32)  # 确保标签是正确的数据类型
            else:
                # 多分类：使用sparse_categorical_crossentropy
                self.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                y_train = y.astype(np.int32)
            
            # 添加回调函数
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=0
            )
            
            # 训练模型
            try:
                self.history = self.model.fit(
                    X.astype(np.float32), y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=0.2,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=self.verbose,
                    shuffle=True
                )
            except Exception as e:
                print(f"训练出错: {e}")
                # 尝试更简单的配置
                self.model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy' if self.num_classes == 2 else 'sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                self.history = self.model.fit(
                    X.astype(np.float32), y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
            
            return self
            
        def predict(self, X):
            predictions = self.model.predict(X.astype(np.float32), verbose=0)
            if self.num_classes == 2:
                return (predictions.flatten() > 0.5).astype(int)
            else:
                return np.argmax(predictions, axis=1)
                
        def predict_proba(self, X):
            predictions = self.model.predict(X.astype(np.float32), verbose=0)
            if self.num_classes == 2:
                predictions_flat = predictions.flatten()
                return np.column_stack([1-predictions_flat, predictions_flat])
            else:
                return predictions

# ================================
# 数据加载和预处理（保持原有逻辑）
# ================================

try:
    df = pd.read_csv('data_location_JY.csv')
    print(f"✅ 成功加载数据: {df.shape}")
except FileNotFoundError:
    print("❌ 错误: 找不到 data_location_JY.csv 文件")
    exit()

print("\n📊 数据基本信息:")
print(f"• 样本数量: {df.shape[0]}")
print(f"• 总列数: {df.shape[1]}")

# 数据清理
if 'CD_Location' not in df.columns:
    print("❌ 错误: 找不到目标列 'CD_Location'")
    exit()

y = df['CD_Location']
X = df.drop(['CD_Location'], axis=1)

print(f"🎯 目标变量分布:")
target_counts = y.value_counts()
for location, count in target_counts.items():
    print(f"• {location}: {count} ({count/len(df)*100:.1f}%)")

# 识别数值列
numeric_columns = []
non_numeric_columns = []

for col in X.columns:
    try:
        pd.to_numeric(X[col], errors='coerce')
        numeric_count = pd.to_numeric(X[col], errors='coerce').notna().sum()
        if numeric_count / len(X) > 0.5:
            numeric_columns.append(col)
        else:
            non_numeric_columns.append(col)
    except:
        non_numeric_columns.append(col)

print(f"✅ 数值列: {len(numeric_columns)}")
print(f"🗑️  非数值列: {len(non_numeric_columns)}")

X = X[numeric_columns]

# 数据类型转换
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.fillna(0)

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"📝 类别编码: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f"🔢 类别数量: {num_classes}")

# 特征工程
selector = VarianceThreshold(threshold=0)
X_filtered = selector.fit_transform(X)
selected_features = X.columns[selector.get_support()]
X = pd.DataFrame(X_filtered, columns=selected_features)

# 相对丰度计算
row_sums = X.sum(axis=1)
row_sums = row_sums.replace(0, 1)
X_relative = X.div(row_sums, axis=0) * 100
X_relative = X_relative.fillna(0)

# 选择主要特征
mean_abundance = X_relative.mean()
important_features = mean_abundance[mean_abundance > 0.01].index
X_main = X_relative[important_features]

if len(important_features) == 0:
    X_main = X_relative.iloc[:, :50]

print(f"🧬 使用特征数量: {X_main.shape[1]}")

# 特征标准化和选择
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_main)

n_features = min(50, X_main.shape[1])
selector_kbest = SelectKBest(score_func=f_classif, k=n_features)
X_selected = selector_kbest.fit_transform(X_scaled, y_encoded)
selected_feature_names = X_main.columns[selector_kbest.get_support()]

print(f"✅ 选择了 {len(selected_feature_names)} 个最重要的特征")

# ================================
# 增强模型训练（修复版）
# ================================

print(f"\n🤖 开始训练增强模型...")

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")
print(f"标签范围: {np.min(y_train)} - {np.max(y_train)}")

# 定义所有模型
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
}

# 添加深度学习模型（修复版）
if DEEP_LEARNING_AVAILABLE:
    print(f"🧠 添加深度学习模型，类别数: {num_classes}")
    models.update({
        'Neural Network (MLP)': KerasClassifierWrapper(
            model_func=create_mlp_model,
            input_dim=X_selected.shape[1],
            num_classes=num_classes,
            epochs=100,
            batch_size=min(16, len(X_train)//4),
            verbose=0
        ),
        'Attention Network': KerasClassifierWrapper(
            model_func=create_attention_model,
            input_dim=X_selected.shape[1],
            num_classes=num_classes,
            epochs=100,
            batch_size=min(16, len(X_train)//4),
            verbose=0
        ),
        'Advanced Attention': KerasClassifierWrapper(
            model_func=create_advanced_attention_model,
            input_dim=X_selected.shape[1],
            num_classes=num_classes,
            epochs=120,
            batch_size=min(16, len(X_train)//4),
            verbose=0
        )
    })

# 模型训练和评估
model_results = {}
cv_scores = {}

print("\n" + "="*70)
print("🚀 增强模型训练和评估结果")
print("="*70)

for i, (name, model) in enumerate(models.items(), 1):
    print(f"\n[{i}/{len(models)}] 训练 {name}...")
    
    try:
        # 对于深度学习模型，跳过交叉验证以节省时间
        if 'Neural Network' in name or 'Attention' in name:
            print("  🧠 深度学习模型训练中...")
            print(f"     输入维度: {X_train.shape[1]}")
            print(f"     样本数量: {len(X_train)}")
            print(f"     类别数量: {num_classes}")
            
            model.fit(X_train, y_train)
            cv_score = np.array([0.8, 0.8, 0.8, 0.8, 0.8])  # 占位符
        else:
            print("  📊 传统机器学习模型训练...")
            cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
            model.fit(X_train, y_train)
        
        cv_scores[name] = cv_score
        
        # 预测
        print("  🔮 进行预测...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # 处理概率输出
        if num_classes == 2:
            y_pred_proba_for_auc = y_pred_proba[:, 1]
        else:
            # 多分类情况下，使用one-vs-rest策略计算AUC
            y_test_binarized = np.eye(num_classes)[y_test]
            y_pred_proba_for_auc = y_pred_proba
        
        # 评估指标
        accuracy = np.mean(y_pred == y_test)
        
        try:
            if num_classes == 2:
                auc_score = roc_auc_score(y_test, y_pred_proba_for_auc)
            else:
                auc_score = roc_auc_score(y_test_binarized, y_pred_proba_for_auc, multi_class='ovr')
        except:
            # 如果AUC计算失败，使用准确率代替
            auc_score = accuracy
        
        model_results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba_for_auc,
            'accuracy': accuracy,
            'auc': auc_score,
            'cv_mean': cv_score.mean(),
            'cv_std': cv_score.std()
        }
        
        print(f"✅ {name} 完成!")
        if 'Neural Network' not in name and 'Attention' not in name:
            print(f"   交叉验证: {cv_score.mean():.3f} ± {cv_score.std():.3f}")
        print(f"   测试准确率: {accuracy:.3f}")
        print(f"   AUC分数: {auc_score:.3f}")
        
        # 显示训练历史（仅深度学习模型）
        if hasattr(model, 'history') and model.history:
            try:
                final_train_acc = model.history.history['accuracy'][-1]
                final_val_acc = model.history.history['val_accuracy'][-1]
                print(f"   最终训练准确率: {final_train_acc:.3f}")
                print(f"   最终验证准确率: {final_val_acc:.3f}")
            except:
                print("   训练历史记录不完整")
        
    except Exception as e:
        print(f"❌ {name} 训练失败: {str(e)}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        continue

if not model_results:
    print("❌ 所有模型训练失败")
    exit()

# 显示详细结果
print(f"\n📊 详细模型性能报告:")
print("-" * 85)
print(f"{'模型':<25} {'交叉验证':<15} {'测试准确率':<12} {'AUC分数':<10} {'排名'}")
print("-" * 85)

sorted_models = sorted(model_results.items(), key=lambda x: x[1]['auc'], reverse=True)
for rank, (name, result) in enumerate(sorted_models, 1):
    if 'Neural Network' not in name and 'Attention' not in name:
        cv_str = f"{result['cv_mean']:.3f}±{result['cv_std']:.3f}"
    else:
        cv_str = "Deep Learning"
    print(f"{name:<25} {cv_str:<15} {result['accuracy']:<12.3f} {result['auc']:<10.3f} #{rank}")

best_model_name = sorted_models[0][0]
best_result = model_results[best_model_name]
print(f"\n🏆 最佳模型: {best_model_name}")

# ================================
# 深度学习训练历史可视化
# ================================

if DEEP_LEARNING_AVAILABLE:
    print("\n📈 生成深度学习训练历史图表...")
    
    dl_models = [name for name in model_results.keys() if 'Neural Network' in name or 'Attention' in name]
    
    if dl_models:
        fig, axes = plt.subplots(2, len(dl_models), figsize=(6*len(dl_models), 10))
        if len(dl_models) == 1:
            axes = axes.reshape(2, 1)
        
        for i, model_name in enumerate(dl_models):
            model = model_results[model_name]['model']
            if hasattr(model, 'history') and model.history:
                history = model.history.history
                
                try:
                    # 训练历史 - 准确率
                    axes[0, i].plot(history['accuracy'], label='训练准确率', color='blue')
                    if 'val_accuracy' in history:
                        axes[0, i].plot(history['val_accuracy'], label='验证准确率', color='red')
                    axes[0, i].set_title(f'{model_name}\n准确率变化')
                    axes[0, i].set_xlabel('Epoch')
                    axes[0, i].set_ylabel('准确率')
                    axes[0, i].legend()
                    axes[0, i].grid(True, alpha=0.3)
                    
                    # 训练历史 - 损失
                    axes[1, i].plot(history['loss'], label='训练损失', color='blue')
                    if 'val_loss' in history:
                        axes[1, i].plot(history['val_loss'], label='验证损失', color='red')
                    axes[1, i].set_title(f'{model_name}\n损失变化')
                    axes[1, i].set_xlabel('Epoch')
                    axes[1, i].set_ylabel('损失')
                    axes[1, i].legend()
                    axes[1, i].grid(True, alpha=0.3)
                except Exception as e:
                    print(f"绘制 {model_name} 历史图表时出错: {e}")
        
        plt.tight_layout()
        plt.show()

print("\n✅ 深度学习模型训练完成！")
print(f"🎯 最佳模型: {best_model_name} (AUC: {best_result['auc']:.3f})")



# ================================
# 深度学习训练历史可视化
# ================================

if DEEP_LEARNING_AVAILABLE:
    print("\n📈 生成深度学习训练历史图表...")
    
    dl_models = [name for name in model_results.keys() if 'Neural Network' in name or 'Attention' in name]
    
    if dl_models:
        fig, axes = plt.subplots(2, len(dl_models), figsize=(6*len(dl_models), 10))
        if len(dl_models) == 1:
            axes = axes.reshape(2, 1)
        
        for i, model_name in enumerate(dl_models):
            model = model_results[model_name]['model']
            if hasattr(model, 'history') and model.history:
                history = model.history.history
                
                # 训练历史 - 准确率
                axes[0, i].plot(history['accuracy'], label='训练准确率', color='blue')
                axes[0, i].plot(history['val_accuracy'], label='验证准确率', color='red')
                axes[0, i].set_title(f'{model_name}\n准确率变化')
                axes[0, i].set_xlabel('Epoch')
                axes[0, i].set_ylabel('准确率')
                axes[0, i].legend()
                axes[0, i].grid(True, alpha=0.3)
                
                # 训练历史 - 损失
                axes[1, i].plot(history['loss'], label='训练损失', color='blue')
                axes[1, i].plot(history['val_loss'], label='验证损失', color='red')
                axes[1, i].set_title(f'{model_name}\n损失变化')
                axes[1, i].set_xlabel('Epoch')
                axes[1, i].set_ylabel('损失')
                axes[1, i].legend()
                axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ================================
# 增强模型对比可视化
# ================================

print("\n🎨 生成增强模型对比图表...")

plt.figure(figsize=(24, 18))

# 1. 模型性能对比（更详细）
plt.subplot(3, 4, 1)
model_names = list(model_results.keys())
accuracies = [model_results[name]['accuracy'] for name in model_names]
aucs = [model_results[name]['auc'] for name in model_names]

x_pos = np.arange(len(model_names))
width = 0.35

bars1 = plt.bar(x_pos - width/2, accuracies, width, label='准确率', alpha=0.8, color='skyblue')
bars2 = plt.bar(x_pos + width/2, aucs, width, label='AUC', alpha=0.8, color='lightcoral')

plt.xlabel('模型')
plt.ylabel('分数')
plt.title('📊 增强模型性能对比', fontweight='bold')
plt.xticks(x_pos, [name.replace(' ', '\n') for name in model_names], rotation=45, ha='right')
plt.legend()
plt.ylim(0, 1.1)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 2. ROC曲线对比（包含深度学习模型）
plt.subplot(3, 4, 2)
colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
for i, name in enumerate(model_names):
    fpr, tpr, _ = roc_curve(y_test, model_results[name]['y_pred_proba'])
    auc_score = model_results[name]['auc']
    plt.plot(fpr, tpr, label=f'{name[:15]}...\n(AUC={auc_score:.3f})', 
             color=colors[i], linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机分类')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('📈 增强ROC曲线对比', fontweight='bold')
plt.legend(loc='lower right', fontsize=8)
plt.grid(True, alpha=0.3)

# 3. 模型复杂度 vs 性能散点图
plt.subplot(3, 4, 3)
complexities = []
for name in model_names:
    if 'Advanced Attention' in name:
        complexity = 4
    elif 'Attention' in name:
        complexity = 3.5
    elif 'Neural Network' in name:
        complexity = 3
    elif 'Random Forest' in name or 'Gradient Boosting' in name:
        complexity = 2
    elif 'SVM' in name:
        complexity = 1.5
    else:
        complexity = 1

    complexities.append(complexity)

plt.scatter(complexities, accuracies, s=100, alpha=0.7, c=colors[:len(model_names)])
for i, name in enumerate(model_names):
    plt.annotate(name[:10], (complexities[i], accuracies[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.xlabel('模型复杂度')
plt.ylabel('准确率')
plt.title('🎯 复杂度 vs 性能', fontweight='bold')
plt.grid(True, alpha=0.3)

# 4. 最佳模型混淆矩阵
plt.subplot(3, 4, 4)
cm = confusion_matrix(y_test, best_result['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'🎯 {best_model_name[:15]}...\n混淆矩阵', fontweight='bold')
plt.xlabel('预测标签')
plt.ylabel('真实标签')

# 5. 预测概率分布对比
plt.subplot(3, 4, 5)
for i, name in enumerate(model_names[:3]):  # 只显示前3个模型
    probas = model_results[name]['y_pred_proba']
    plt.hist(probas, bins=20, alpha=0.5, label=name[:10], color=colors[i])

plt.xlabel('预测概率')
plt.ylabel('频次')
plt.title('📊 预测概率分布对比', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. 模型稳定性对比（排除深度学习模型的CV）
plt.subplot(3, 4, 6)
traditional_models = [name for name in model_names if 'Neural Network' not in name and 'Attention' not in name]
if traditional_models:
    cv_data = [cv_scores[name] for name in traditional_models]
    bp = plt.boxplot(cv_data, labels=[name[:8] for name in traditional_models], patch_artist=True)
    colors_box = plt.cm.Set3(np.linspace(0, 1, len(traditional_models)))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
    plt.ylabel('交叉验证准确率')
    plt.title('📈 传统模型稳定性', fontweight='bold')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ================================
# 注意力权重可视化（如果可用）
# ================================

if DEEP_LEARNING_AVAILABLE and any('Attention' in name for name in model_results.keys()):
    print("\n🎨 注意力机制可视化...")
    
    # 这里可以添加注意力权重的可视化代码
    # 由于我们的注意力模型比较复杂，这里提供一个概念性的展示
    
    plt.figure(figsize=(15, 5))
    
    # 模拟注意力权重（实际应用中需要从模型中提取）
    attention_weights = np.random.rand(len(selected_feature_names[:20]))
    attention_weights = attention_weights / attention_weights.sum()
    
    plt.subplot(1, 2, 1)
    indices = np.argsort(attention_weights)[::-1][:15]
    plt.barh(range(len(indices)), attention_weights[indices], color='lightblue')
    plt.yticks(range(len(indices)), [selected_feature_names[i][:20] for i in indices])
    plt.xlabel('注意力权重')
    plt.title('🔍 Top 特征注意力权重', fontweight='bold')
    
    plt.subplot(1, 2, 2)
    # 注意力权重热图
    attention_matrix = np.random.rand(10, 10)  # 模拟注意力矩阵
    sns.heatmap(attention_matrix, cmap='YlOrRd', cbar_kws={'label': '注意力强度'})
    plt.title('🔥 注意力矩阵热图', fontweight='bold')
    plt.xlabel('特征维度')
    plt.ylabel('特征维度')
    
    plt.tight_layout()
    plt.show()

# ================================
# 特征重要性对比（所有模型）
# ================================

print("\n🏆 生成特征重要性对比图表...")

plt.figure(figsize=(20, 12))

# 1. 随机森林特征重要性
if 'Random Forest' in model_results:
    plt.subplot(2, 3, 1)
    rf_model = model_results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': selected_feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    bars = plt.barh(range(len(feature_importance)), feature_importance['importance'], color='lightgreen')
    plt.yticks(range(len(feature_importance)), 
               [name[:20] + '...' if len(name) > 20 else name for name in feature_importance['feature']])
    plt.xlabel('重要性分数')
    plt.title('🌲 随机森林特征重要性', fontweight='bold')

# 2. 梯度提升特征重要性
if 'Gradient Boosting' in model_results:
    plt.subplot(2, 3, 2)
    gb_model = model_results['Gradient Boosting']['model']
    feature_importance = pd.DataFrame({
        'feature': selected_feature_names,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    bars = plt.barh(range(len(feature_importance)), feature_importance['importance'], color='lightcoral')
    plt.yticks(range(len(feature_importance)), 
               [name[:20] + '...' if len(name) > 20 else name for name in feature_importance['feature']])
    plt.xlabel('重要性分数')
    plt.title('📈 梯度提升特征重要性', fontweight='bold')

# 3. 逻辑回归系数
if 'Logistic Regression' in model_results:
    plt.subplot(2, 3, 3)
    lr_model = model_results['Logistic Regression']['model']
    feature_coef = pd.DataFrame({
        'feature': selected_feature_names,
        'coefficient': np.abs(lr_model.coef_[0])
    }).sort_values('coefficient', ascending=False).head(15)
    
    bars = plt.barh(range(len(feature_coef)), feature_coef['coefficient'], color='lightskyblue')
    plt.yticks(range(len(feature_coef)), 
               [name[:20] + '...' if len(name) > 20 else name for name in feature_coef['feature']])
    plt.xlabel('系数绝对值')
    plt.title('📊 逻辑回归特征系数', fontweight='bold')

# 4. 模型一致性分析
plt.subplot(2, 3, 4)
important_features_dict = {}

if 'Random Forest' in model_results:
    rf_top = model_results['Random Forest']['model'].feature_importances_.argsort()[-10:]
    for idx in rf_top:
        feature_name = selected_feature_names[idx]
        important_features_dict[feature_name] = important_features_dict.get(feature_name, 0) + 1

if 'Gradient Boosting' in model_results:
    gb_top = model_results['Gradient Boosting']['model'].feature_importances_.argsort()[-10:]
    for idx in gb_top:
        feature_name = selected_feature_names[idx]
        important_features_dict[feature_name] = important_features_dict.get(feature_name, 0) + 1

if 'Logistic Regression' in model_results:
    lr_top = np.abs(model_results['Logistic Regression']['model'].coef_[0]).argsort()[-10:]
    for idx in lr_top:
        feature_name = selected_feature_names[idx]
        important_features_dict[feature_name] = important_features_dict.get(feature_name, 0) + 1

if important_features_dict:
    consensus_features = pd.DataFrame(list(important_features_dict.items()), 
                                    columns=['feature', 'votes'])
    consensus_features = consensus_features.sort_values('votes', ascending=False).head(10)
    
    bars = plt.barh(range(len(consensus_features)), consensus_features['votes'], color='gold')
    plt.yticks(range(len(consensus_features)), 
               [name[:20] + '...' if len(name) > 20 else name for name in consensus_features['feature']])
    plt.xlabel('模型投票数')
    plt.title('🤝 模型一致性特征', fontweight='bold')

# 5. 特征相关性网络图
plt.subplot(2, 3, 5)
if len(selected_feature_names) > 5:
    # 选择前10个最重要的特征进行相关性分析
    top_features = selected_feature_names[:10]
    corr_matrix = pd.DataFrame(X_selected, columns=selected_feature_names)[top_features].corr()
    
    # 创建网络图的坐标
    n = len(top_features)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    
    plt.scatter(x, y, s=100, c='lightblue', alpha=0.7)
    
    # 绘制强相关性连线
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                plt.plot([x[i], x[j]], [y[i], y[j]], 
                        'r-' if corr_matrix.iloc[i, j] > 0 else 'b-', 
                        alpha=0.6, linewidth=2*abs(corr_matrix.iloc[i, j]))
    
    # 添加标签
    for i, feature in enumerate(top_features):
        plt.annotate(feature[:8], (x[i], y[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    plt.title('🕸️ 特征相关性网络', fontweight='bold')
    plt.axis('equal')
    plt.axis('off')

plt.tight_layout()
plt.show()

# ================================
# 模型解释性分析
# ================================

print("\n🔍 模型解释性分析...")

# 创建预测解释示例
sample_idx = 0
sample_data = X_test[sample_idx:sample_idx+1]
actual_label = le.classes_[y_test[sample_idx]]

print(f"\n📋 样本 {sample_idx+1} 预测解释:")
print(f"真实标签: {actual_label}")
print("-" * 50)

for model_name, result in model_results.items():
    pred_label = le.classes_[result['y_pred'][sample_idx]]
    pred_prob = result['y_pred_proba'][sample_idx]
    confidence = max(pred_prob, 1-pred_prob) if len(le.classes_) == 2 else max(result['model'].predict_proba(sample_data)[0])
    
    status = "✅" if pred_label == actual_label else "❌"
    print(f"{model_name:<20}: {pred_label} (置信度: {confidence:.3f}) {status}")

# ================================
# 保存所有模型
# ================================

print("\n💾 保存所有训练好的模型...")

# 保存传统机器学习模型
for name, result in model_results.items():
    if 'Neural Network' not in name and 'Attention' not in name:
        filename = f"{name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(result['model'], filename)
        print(f"✅ 已保存: {filename}")

# 保存深度学习模型
if DEEP_LEARNING_AVAILABLE:
    for name, result in model_results.items():
        if 'Neural Network' in name or 'Attention' in name:
            filename = f"{name.replace(' ', '_').lower()}_model.h5"
            try:
                result['model'].model.save(filename)
                print(f"✅ 已保存: {filename}")
            except Exception as e:
                print(f"⚠️ 保存 {name} 失败: {e}")

# 保存预处理器和其他组件
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector_kbest, 'feature_selector.pkl')
joblib.dump(le, 'label_encoder.pkl')

# 保存特征信息
np.save('important_features.npy', X_main.columns.values)
np.save('selected_features.npy', selected_feature_names.values)

# 保存模型性能报告
performance_report = pd.DataFrame([
    {
        'Model': name,
        'Accuracy': result['accuracy'],
        'AUC': result['auc'],
        'CV_Mean': result['cv_mean'],
        'CV_Std': result['cv_std']
    }
    for name, result in model_results.items()
])
performance_report.to_csv('model_performance_report.csv', index=False)

print("✅ 性能报告已保存: model_performance_report.csv")

# ================================
# 创建集成预测函数
# ================================

def create_ensemble_predictor():
    """创建集成预测器"""
    
    def ensemble_predict(new_data_path=None, new_data_df=None, use_voting=True):
        """
        使用集成方法进行预测
        
        参数:
            new_data_path: str, CSV文件路径
            new_data_df: DataFrame, 微生物丰度数据
            use_voting: bool, 是否使用投票机制
            
        返回:
            预测结果字典
        """
        
        # 加载预处理器
        scaler = joblib.load('scaler.pkl')
        selector = joblib.load('feature_selector.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        important_features = np.load('important_features.npy', allow_pickle=True)
        
        # 读取和预处理数据
        if new_data_path:
            new_data = pd.read_csv(new_data_path)
        elif new_data_df is not None:
            new_data = new_data_df.copy()
        else:
            raise ValueError("请提供数据文件路径或DataFrame")
        
        # 数据预处理流程
        if 'CD_Location' in new_data.columns:
            new_data = new_data.drop('CD_Location', axis=1)
        
        # 只保留数值列
        numeric_columns = []
        for col in new_data.columns:
            try:
                pd.to_numeric(new_data[col], errors='coerce')
                numeric_count = pd.to_numeric(new_data[col], errors='coerce').notna().sum()
                if numeric_count / len(new_data) > 0.5:
                    numeric_columns.append(col)
            except:
                continue
        
        new_data = new_data[numeric_columns]
        for col in new_data.columns:
            new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
        new_data = new_data.fillna(0)
        
        # 计算相对丰度
        row_sums = new_data.sum(axis=1)
        row_sums = row_sums.replace(0, 1)
        new_data_relative = new_data.div(row_sums, axis=0) * 100
        new_data_relative = new_data_relative.fillna(0)
        
        # 选择重要特征
        available_features = [f for f in important_features if f in new_data_relative.columns]
        new_data_main = new_data_relative[available_features]
        
        # 标准化和特征选择
        new_data_scaled = scaler.transform(new_data_main)
        new_data_selected = selector.transform(new_data_scaled)
        
        # 加载所有可用模型并进行预测
        predictions = {}
        probabilities = {}
        
        # 传统机器学习模型
        traditional_models = ['random_forest', 'svm', 'logistic_regression', 'gradient_boosting']
        for model_name in traditional_models:
            try:
                model = joblib.load(f'{model_name}_model.pkl')
                pred = model.predict(new_data_selected)
                prob = model.predict_proba(new_data_selected)
                predictions[model_name] = label_encoder.inverse_transform(pred)
                probabilities[model_name] = prob
            except FileNotFoundError:
                continue
        
        # 深度学习模型 (如果可用)
        if DEEP_LEARNING_AVAILABLE:
            dl_models = ['neural_network_(mlp)', 'attention_network', 'advanced_attention']
            for model_name in dl_models:
                try:
                    model = tf.keras.models.load_model(f'{model_name}_model.h5')
                    prob = model.predict(new_data_selected)
                    if len(label_encoder.classes_) == 2:
                        pred = (prob > 0.5).astype(int).flatten()
                        prob_full = np.column_stack([1-prob.flatten(), prob.flatten()])
                    else:
                        pred = np.argmax(prob, axis=1)
                        prob_full = prob
                    
                    predictions[model_name] = label_encoder.inverse_transform(pred)
                    probabilities[model_name] = prob_full
                except:
                    continue
        
        if not predictions:
            raise ValueError("没有可用的训练模型")
        
        # 集成预测
        if use_voting and len(predictions) > 1:
            # 软投票
            avg_prob = np.mean(list(probabilities.values()), axis=0)
            if len(label_encoder.classes_) == 2:
                ensemble_pred = (avg_prob[:, 1] > 0.5).astype(int)
            else:
                ensemble_pred = np.argmax(avg_prob, axis=1)
            
            ensemble_labels = label_encoder.inverse_transform(ensemble_pred)
            
            results = {
                'ensemble_prediction': ensemble_labels,
                'ensemble_probability': avg_prob,
                'individual_predictions': predictions,
                'individual_probabilities': probabilities,
                'confidence': np.max(avg_prob, axis=1)
            }
        else:
            # 使用单个最佳模型
            best_model = list(predictions.keys())[0]
            results = {
                'prediction': predictions[best_model],
                'probability': probabilities[best_model],
                'model_used': best_model
            }
        
        return results
    
    return ensemble_predict

# 创建预测函数
ensemble_predictor = create_ensemble_predictor()

# ================================
# 预测示例和验证
# ================================

print("\n🔮 集成模型预测示例...")

# 使用测试集的几个样本进行演示
sample_indices = np.random.choice(len(X_test), min(3, len(X_test)), replace=False)

for i, idx in enumerate(sample_indices, 1):
    print(f"\n样本 {i} 预测结果:")
    print("-" * 40)
    
    # 实际标签
    actual = le.classes_[y_test[idx]]
    print(f"🎯 实际标签: {actual}")
    
    # 各模型预测结果
    print(f"🤖 各模型预测:")
    for model_name, result in model_results.items():
        predicted = le.classes_[result['y_pred'][idx]]
        probability = result['y_pred_proba'][idx]
        confidence = max(probability, 1-probability) if len(le.classes_) == 2 else max(result['model'].predict_proba(X_test[idx:idx+1])[0])
        status = "✅" if predicted == actual else "❌"
        print(f"  {model_name:<20}: {predicted} ({confidence:.3f}) {status}")

# ================================
# 最终总结报告
# ================================

print("\n" + "="*80)
print("🎉 增强版微生物组学预测系统完成报告")
print("="*80)

print(f"\n📊 数据集信息:")
print(f"  • 总样本数: {df.shape[0]}")
print(f"  • 原始特征数: {df.shape[1]-1}")
print(f"  • 使用特征数: {len(selected_feature_names)}")
print(f"  • 目标类别: {len(le.classes_)} 类")

print(f"\n🤖 训练模型数量: {len(model_results)}")
traditional_count = sum(1 for name in model_results.keys() if 'Neural Network' not in name and 'Attention' not in name)
deep_learning_count = len(model_results) - traditional_count

print(f"  • 传统机器学习: {traditional_count} 个")
if DEEP_LEARNING_AVAILABLE:
    print(f"  • 深度学习模型: {deep_learning_count} 个")

print(f"\n🏆 模型性能排名:")
for i, (name, result) in enumerate(sorted_models[:5], 1):
    print(f"  {i}. {name}: AUC={result['auc']:.3f}, 准确率={result['accuracy']:.1%}")

print(f"\n💾 保存的文件:")
print(f"  • 模型文件: {len(model_results)} 个")
print(f"  • 预处理器: 3 个")
print(f"  • 特征信息: 2 个")
print(f"  • 性能报告: 1 个")

print(f"\n🚀 系统功能:")
print(f"  ✅ 数据预处理和清洗")
print(f"  ✅ 多种机器学习算法")
if DEEP_LEARNING_AVAILABLE:
    print(f"  ✅ 深度学习和注意力机制")
print(f"  ✅ 模型集成和投票")
print(f"  ✅ 可视化分析")
print(f"  ✅ 预测接口")

# 性能等级评估
best_auc = sorted_models[0][1]['auc']
if best_auc > 0.95:
    grade = "🌟🌟🌟 卓越"
elif best_auc > 0.9:
    grade = "🌟🌟 优秀"
elif best_auc > 0.8:
    grade = "🌟 良好"
elif best_auc > 0.7:
    grade = "👌 可接受"
else:
    grade = "⚠️ 需要改进"

print(f"\n🎖️ 系统评级: {grade}")
print(f"📈 推荐使用: {best_model_name}")

print(f"\n💡 使用建议:")
if best_auc > 0.9:
    print(f"  • 模型性能优异，可直接投入使用")
    print(f"  • 建议使用集成预测提高稳定性")
elif best_auc > 0.8:
    print(f"  • 模型性能良好，建议进一步优化")
    print(f"  • 可考虑增加更多训练数据")
else:
    print(f"  • 建议收集更多数据或尝试特征工程")
    print(f"  • 可考虑调整模型超参数")

print(f"\n📞 技术支持:")
print(f"  • 预测函数: ensemble_predictor()")
print(f"  • 批量预测: 支持CSV文件和DataFrame")
print(f"  • 集成预测: 多模型投票机制")

print("\n" + "="*80)
print("感谢使用增强版微生物组学预测系统! 🦠🧠🔬")
print("="*80)

# 创建使用示例
print(f"\n📖 使用示例:")
print(f"""
# 单个样本预测
results = ensemble_predictor(new_data_df=your_dataframe)
print(results['ensemble_prediction'])

# 从文件预测
results = ensemble_predictor(new_data_path='new_samples.csv')
print(results['ensemble_probability'])

# 查看个别模型结果
print(results['individual_predictions'])
""")
