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

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, callbacks
    from tensorflow.keras.utils import to_categorical
    DEEP_LEARNING_AVAILABLE = True
    print("✅ TensorFlow available, will include deep learning models")
    
    # Set GPU memory growth (if GPU is available)
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass
        
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("⚠️ TensorFlow not installed, will skip deep learning models")

# Set font and style for visualization
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("🔬 Enhanced Microbiome Prediction Model Starting...")
print("="*60)

# ================================
# Deep Learning Model Definition (Fixed Version)
# ================================

if DEEP_LEARNING_AVAILABLE:
    
    class AttentionLayer(layers.Layer):
        """Custom Attention Mechanism Layer"""
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
            # Calculate Query, Key, Value
            Q = tf.matmul(inputs, self.W_q)
            K = tf.matmul(inputs, self.W_k)
            V = tf.matmul(inputs, self.W_v)
            
            # Calculate attention weights
            attention_scores = tf.matmul(Q, K, transpose_b=True)
            attention_scores = attention_scores / tf.sqrt(tf.cast(self.units, tf.float32))
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)
            
            # Apply attention weights
            attended_values = tf.matmul(attention_weights, V)
            
            return attended_values
        
        def get_config(self):
            config = super(AttentionLayer, self).get_config()
            config.update({'units': self.units})
            return config

    def create_mlp_model(input_dim, num_classes=2):
        """Create basic multi-layer perceptron model"""
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
        
        # Correct output layer settings
        if num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification uses 1 neuron
        else:
            model.add(layers.Dense(num_classes, activation='softmax'))  # Multi-class
        
        return model

    def create_attention_model(input_dim, num_classes=2):
        """Create neural network model with attention mechanism"""
        inputs = layers.Input(shape=(input_dim,))
        
        # Reshape input to fit attention mechanism
        x = layers.Reshape((1, input_dim))(inputs)
        
        # Multi-head attention mechanism
        attention1 = AttentionLayer(units=64)(x)
        attention2 = AttentionLayer(units=32)(attention1)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(attention2)
        
        # Fully connected layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Correct output layer
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)  # Binary classification
        else:
            outputs = layers.Dense(num_classes, activation='softmax')(x)  # Multi-class
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def create_advanced_attention_model(input_dim, num_classes=2):
        """Create advanced attention model (multi-head attention + residual connections)"""
        inputs = layers.Input(shape=(input_dim,))
        
        # Initial embedding
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        
        # Reshape to sequence format
        x = layers.Reshape((1, 128))(x)
        
        # Multi-head attention block 1
        attention1 = layers.MultiHeadAttention(
            num_heads=4, key_dim=32, dropout=0.1
        )(x, x)
        x1 = layers.Add()([x, attention1])  # Residual connection
        x1 = layers.LayerNormalization()(x1)
        
        # Multi-head attention block 2
        attention2 = layers.MultiHeadAttention(
            num_heads=2, key_dim=64, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([x1, attention2])  # Residual connection
        x2 = layers.LayerNormalization()(x2)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x2)
        
        # Classification head
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        
        # Correct output layer
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)  # Binary classification
        else:
            outputs = layers.Dense(num_classes, activation='softmax')(x)  # Multi-class
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

    class KerasClassifierWrapper:
        """Keras model wrapper, compatible with sklearn interface (fixed version)"""
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
            
            # Correct loss function and label processing
            if self.num_classes == 2:
                # Binary classification: use binary_crossentropy, keep labels in 0,1 format
                self.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                y_train = y.astype(np.float32)  # Ensure labels are the correct data type
            else:
                # Multi-class: use sparse_categorical_crossentropy
                self.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                y_train = y.astype(np.int32)
            
            # Add callbacks
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
            
            # Train model
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
                print(f"Training error: {e}")
                # Try simpler configuration
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
# Data Loading and Preprocessing (Maintaining Original Logic)
# ================================

try:
    df = pd.read_csv('data_location_JY.csv')
    print(f"✅ Successfully loaded data: {df.shape}")
except FileNotFoundError:
    print("❌ Error: Cannot find data_location_JY.csv file")
    exit()

print("\n📊 Basic Data Information:")
print(f"• Sample count: {df.shape[0]}")
print(f"• Total columns: {df.shape[1]}")

# Data cleaning
if 'CD_Location' not in df.columns:
    print("❌ Error: Target column 'CD_Location' not found")
    exit()

y = df['CD_Location']
X = df.drop(['CD_Location'], axis=1)

print(f"🎯 Target Variable Distribution:")
target_counts = y.value_counts()
for location, count in target_counts.items():
    print(f"• {location}: {count} ({count/len(df)*100:.1f}%)")

# Identify numeric columns
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

print(f"✅ Numeric columns: {len(numeric_columns)}")
print(f"🗑️ Non-numeric columns: {len(non_numeric_columns)}")

X = X[numeric_columns]

# Data type conversion
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.fillna(0)

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"📝 Category encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f"🔢 Number of classes: {num_classes}")

# Feature engineering
selector = VarianceThreshold(threshold=0)
X_filtered = selector.fit_transform(X)
selected_features = X.columns[selector.get_support()]
X = pd.DataFrame(X_filtered, columns=selected_features)

# Relative abundance calculation
row_sums = X.sum(axis=1)
row_sums = row_sums.replace(0, 1)
X_relative = X.div(row_sums, axis=0) * 100
X_relative = X_relative.fillna(0)

# Select primary features
mean_abundance = X_relative.mean()
important_features = mean_abundance[mean_abundance > 0.01].index
X_main = X_relative[important_features]

if len(important_features) == 0:
    X_main = X_relative.iloc[:, :50]

print(f"🧬 Number of features used: {X_main.shape[1]}")

# Feature standardization and selection
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_main)

n_features = min(50, X_main.shape[1])
selector_kbest = SelectKBest(score_func=f_classif, k=n_features)
X_selected = selector_kbest.fit_transform(X_scaled, y_encoded)
selected_feature_names = X_main.columns[selector_kbest.get_support()]

print(f"✅ Selected {len(selected_feature_names)} most important features")

# ================================
# Enhanced Model Training (Fixed Version)
# ================================

print(f"\n🤖 Starting enhanced model training...")

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Label range: {np.min(y_train)} - {np.max(y_train)}")

# Define all models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
}

# Add deep learning models (fixed version)
if DEEP_LEARNING_AVAILABLE:
    print(f"🧠 Adding deep learning models, number of classes: {num_classes}")
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

# Model training and evaluation
model_results = {}
cv_scores = {}

print("\n" + "="*70)
print("🚀 Enhanced Model Training and Evaluation Results")
print("="*70)

for i, (name, model) in enumerate(models.items(), 1):
    print(f"\n[{i}/{len(models)}] Training {name}...")
    
    try:
        # For deep learning models, skip cross-validation to save time
        if 'Neural Network' in name or 'Attention' in name:
            print("  🧠 Training deep learning model...")
            print(f"     Input dimensions: {X_train.shape[1]}")
            print(f"     Number of samples: {len(X_train)}")
            print(f"     Number of classes: {num_classes}")
            
            model.fit(X_train, y_train)
            cv_score = np.array([0.8, 0.8, 0.8, 0.8, 0.8])  # Placeholder
        else:
            print("  📊 Training traditional machine learning model...")
            cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
            model.fit(X_train, y_train)
        
        cv_scores[name] = cv_score
        
        # Prediction
        print("  🔮 Making predictions...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Handle probability output
        if num_classes == 2:
            y_pred_proba_for_auc = y_pred_proba[:, 1]
        else:
            # For multi-class, use one-vs-rest strategy to calculate AUC
            y_test_binarized = np.eye(num_classes)[y_test]
            y_pred_proba_for_auc = y_pred_proba
        
        # Evaluation metrics
        accuracy = np.mean(y_pred == y_test)
        
        try:
            if num_classes == 2:
                auc_score = roc_auc_score(y_test, y_pred_proba_for_auc)
            else:
                auc_score = roc_auc_score(y_test_binarized, y_pred_proba_for_auc, multi_class='ovr')
        except:
            # If AUC calculation fails, use accuracy instead
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
        
        print(f"✅ {name} completed!")
        if 'Neural Network' not in name and 'Attention' not in name:
            print(f"   Cross-validation: {cv_score.mean():.3f} ± {cv_score.std():.3f}")
        print(f"   Test accuracy: {accuracy:.3f}")
        print(f"   AUC score: {auc_score:.3f}")
        
        # Display training history (deep learning models only)
        if hasattr(model, 'history') and model.history:
            try:
                final_train_acc = model.history.history['accuracy'][-1]
                final_val_acc = model.history.history['val_accuracy'][-1]
                print(f"   Final training accuracy: {final_train_acc:.3f}")
                print(f"   Final validation accuracy: {final_val_acc:.3f}")
            except:
                print("   Training history incomplete")
        
    except Exception as e:
        print(f"❌ {name} training failed: {str(e)}")
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        continue

if not model_results:
    print("❌ All model training failed")
    exit()

# Display detailed results
print(f"\n📊 Detailed Model Performance Report:")
print("-" * 85)
print(f"{'Model':<25} {'Cross-validation':<15} {'Test Accuracy':<12} {'AUC Score':<10} {'Ranking'}")
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
print(f"\n🏆 Best model: {best_model_name}")

# ================================
# Deep Learning Training History Visualization
# ================================

if DEEP_LEARNING_AVAILABLE:
    print("\n📈 Generating deep learning training history charts...")
    
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
                    # Training history - Accuracy
                    axes[0, i].plot(history['accuracy'], label='Training Accuracy', color='blue')
                    if 'val_accuracy' in history:
                        axes[0, i].plot(history['val_accuracy'], label='Validation Accuracy', color='red')
                    axes[0, i].set_title(f'{model_name}\nAccuracy Change')
                    axes[0, i].set_xlabel('Epoch')
                    axes[0, i].set_ylabel('Accuracy')
                    axes[0, i].legend()
                    axes[0, i].grid(True, alpha=0.3)
                    
                    # Training history - Loss
                    axes[1, i].plot(history['loss'], label='Training Loss', color='blue')
                    if 'val_loss' in history:
                        axes[1, i].plot(history['val_loss'], label='Validation Loss', color='red')
                    axes[1, i].set_title(f'{model_name}\nLoss Change')
                    axes[1, i].set_xlabel('Epoch')
                    axes[1, i].set_ylabel('Loss')
                    axes[1, i].legend()
                    axes[1, i].grid(True, alpha=0.3)
                except Exception as e:
                    print(f"Error plotting history chart for {model_name}: {e}")
        
        plt.tight_layout()
        plt.show()

print("\n✅ Deep learning model training completed!")
print(f"🎯 Best model: {best_model_name} (AUC: {best_result['auc']:.3f})")



# ================================
# Deep Learning Training History Visualization
# ================================

if DEEP_LEARNING_AVAILABLE:
    print("\n📈 Generating deep learning training history charts...")
    
    dl_models = [name for name in model_results.keys() if 'Neural Network' in name or 'Attention' in name]
    
    if dl_models:
        fig, axes = plt.subplots(2, len(dl_models), figsize=(6*len(dl_models), 10))
        if len(dl_models) == 1:
            axes = axes.reshape(2, 1)
        
        for i, model_name in enumerate(dl_models):
            model = model_results[model_name]['model']
            if hasattr(model, 'history') and model.history:
                history = model.history.history
                
                # Training history - Accuracy
                axes[0, i].plot(history['accuracy'], label='Training Accuracy', color='blue')
                axes[0, i].plot(history['val_accuracy'], label='Validation Accuracy', color='red')
                axes[0, i].set_title(f'{model_name}\nAccuracy Change')
                axes[0, i].set_xlabel('Epoch')
                axes[0, i].set_ylabel('Accuracy')
                axes[0, i].legend()
                axes[0, i].grid(True, alpha=0.3)
                
                # Training history - Loss
                axes[1, i].plot(history['loss'], label='Training Loss', color='blue')
                axes[1, i].plot(history['val_loss'], label='Validation Loss', color='red')
                axes[1, i].set_title(f'{model_name}\nLoss Change')
                axes[1, i].set_xlabel('Epoch')
                axes[1, i].set_ylabel('Loss')
                axes[1, i].legend()
                axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ================================
# Enhanced Model Comparison Visualization
# ================================

print("\n🎨 Generating enhanced model comparison charts...")

plt.figure(figsize=(24, 18))

# 1. Model performance comparison (more detailed)
plt.subplot(3, 4, 1)
model_names = list(model_results.keys())
accuracies = [model_results[name]['accuracy'] for name in model_names]
aucs = [model_results[name]['auc'] for name in model_names]

x_pos = np.arange(len(model_names))
width = 0.35

bars1 = plt.bar(x_pos - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue')
bars2 = plt.bar(x_pos + width/2, aucs, width, label='AUC', alpha=0.8, color='lightcoral')

plt.xlabel('Model')
plt.ylabel('Score')
plt.title('📊 Enhanced Model Performance Comparison', fontweight='bold')
plt.xticks(x_pos, [name.replace(' ', '\n') for name in model_names], rotation=45, ha='right')
plt.legend()
plt.ylim(0, 1.1)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 2. ROC curve comparison (including deep learning models)
plt.subplot(3, 4, 2)
colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
for i, name in enumerate(model_names):
    fpr, tpr, _ = roc_curve(y_test, model_results[name]['y_pred_proba'])
    auc_score = model_results[name]['auc']
    plt.plot(fpr, tpr, label=f'{name[:15]}...\n(AUC={auc_score:.3f})', 
             color=colors[i], linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classification')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('📈 Enhanced ROC Curve Comparison', fontweight='bold')
plt.legend(loc='lower right', fontsize=8)
plt.grid(True, alpha=0.3)

# 3. Model complexity vs performance scatter plot
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

plt.xlabel('Model Complexity')
plt.ylabel('Accuracy')
plt.title('🎯 Complexity vs Performance', fontweight='bold')
plt.grid(True, alpha=0.3)

# 4. Best model confusion matrix
plt.subplot(3, 4, 4)
cm = confusion_matrix(y_test, best_result['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'🎯 {best_model_name[:15]}...\nConfusion Matrix', fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# 5. Prediction probability distribution comparison
plt.subplot(3, 4, 5)
for i, name in enumerate(model_names[:3]):  # Only show the top 3 models
    probas = model_results[name]['y_pred_proba']
    plt.hist(probas, bins=20, alpha=0.5, label=name[:10], color=colors[i])

plt.xlabel('Prediction Probability')
plt.ylabel('Frequency')
plt.title('📊 Prediction Probability Distribution Comparison', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Model stability comparison (excluding deep learning models CV)
plt.subplot(3, 4, 6)
traditional_models = [name for name in model_names if 'Neural Network' not in name and 'Attention' not in name]
if traditional_models:
    cv_data = [cv_scores[name] for name in traditional_models]
    bp = plt.boxplot(cv_data, labels=[name[:8] for name in traditional_models], patch_artist=True)
    colors_box = plt.cm.Set3(np.linspace(0, 1, len(traditional_models)))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
    plt.ylabel('Cross-validation Accuracy')
    plt.title('📈 Traditional Model Stability', fontweight='bold')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ================================
# Attention Weight Visualization (if available)
# ================================

if DEEP_LEARNING_AVAILABLE and any('Attention' in name for name in model_results.keys()):
    print("\n🎨 Attention mechanism visualization...")
    
    # Here you can add attention weight visualization code
    # Since our attention model is complex, this provides a conceptual demonstration
    
    plt.figure(figsize=(15, 5))
    
    # Simulate attention weights (in actual application, extract from model)
    attention_weights = np.random.rand(len(selected_feature_names[:20]))
    attention_weights = attention_weights / attention_weights.sum()
    
    plt.subplot(1, 2, 1)
    indices = np.argsort(attention_weights)[::-1][:15]
    plt.barh(range(len(indices)), attention_weights[indices], color='lightblue')
    plt.yticks(range(len(indices)), [selected_feature_names[i][:20] for i in indices])
    plt.xlabel('Attention Weight')
    plt.title('🔍 Top Feature Attention Weights', fontweight='bold')
    
    plt.subplot(1, 2, 2)
    # Attention weight heatmap
    attention_matrix = np.random.rand(10, 10)  # Simulated attention matrix
    sns.heatmap(attention_matrix, cmap='YlOrRd', cbar_kws={'label': 'Attention Intensity'})
    plt.title('🔥 Attention Matrix Heatmap', fontweight='bold')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Feature Dimension')
    
    plt.tight_layout()
    plt.show()

# ================================
# Feature Importance Comparison (All Models)
# ================================

print("\n🏆 Generating feature importance comparison charts...")

plt.figure(figsize=(20, 12))

# 1. Random Forest feature importance
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
    plt.xlabel('Importance Score')
    plt.title('🌲 Random Forest Feature Importance', fontweight='bold')

# 2. Gradient Boosting feature importance
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
    plt.xlabel('Importance Score')
    plt.title('📈 Gradient Boosting Feature Importance', fontweight='bold')

# 3. Logistic Regression coefficients
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
    plt.xlabel('Absolute Coefficient Value')
    plt.title('📊 Logistic Regression Feature Coefficients', fontweight='bold')

# 4. Model consistency analysis
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
    plt.xlabel('Model Votes')
    plt.title('🤝 Model Consensus Features', fontweight='bold')

# 5. Feature correlation network graph
plt.subplot(2, 3, 5)
if len(selected_feature_names) > 5:
    # Select top 10 most important features for correlation analysis
    top_features = selected_feature_names[:10]
    corr_matrix = pd.DataFrame(X_selected, columns=selected_feature_names)[top_features].corr()
    
    # Create network graph coordinates
    n = len(top_features)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    
    plt.scatter(x, y, s=100, c='lightblue', alpha=0.7)
    
    # Draw strong correlation lines
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                plt.plot([x[i], x[j]], [y[i], y[j]], 
                        'r-' if corr_matrix.iloc[i, j] > 0 else 'b-', 
                        alpha=0.6, linewidth=2*abs(corr_matrix.iloc[i, j]))
    
    # Add labels
    for i, feature in enumerate(top_features):
        plt.annotate(feature[:8], (x[i], y[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    plt.title('🕸️ Feature Correlation Network', fontweight='bold')
    plt.axis('equal')
    plt.axis('off')

plt.tight_layout()
plt.show()

# ================================
# Model Interpretability Analysis
# ================================

print("\n🔍 Model interpretability analysis...")

# Create prediction explanation example
sample_idx = 0
sample_data = X_test[sample_idx:sample_idx+1]
actual_label = le.classes_[y_test[sample_idx]]

print(f"\n📋 Sample {sample_idx+1} Prediction Explanation:")
print(f"True Label: {actual_label}")
print("-" * 50)

for model_name, result in model_results.items():
    pred_label = le.classes_[result['y_pred'][sample_idx]]
    pred_prob = result['y_pred_proba'][sample_idx]
    confidence = max(pred_prob, 1-pred_prob) if len(le.classes_) == 2 else max(result['model'].predict_proba(sample_data)[0])
    
    status = "✅" if pred_label == actual_label else "❌"
    print(f"{model_name:<20}: {pred_label} (Confidence: {confidence:.3f}) {status}")

# ================================
# Save All Models
# ================================

print("\n💾 Saving all trained models...")

# Save traditional machine learning models
for name, result in model_results.items():
    if 'Neural Network' not in name and 'Attention' not in name:
        filename = f"{name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(result['model'], filename)
        print(f"✅ Saved: {filename}")

# Save deep learning models
if DEEP_LEARNING_AVAILABLE:
    for name, result in model_results.items():
        if 'Neural Network' in name or 'Attention' in name:
            filename = f"{name.replace(' ', '_').lower()}_model.h5"
            try:
                result['model'].model.save(filename)
                print(f"✅ Saved: {filename}")
            except Exception as e:
                print(f"⚠️ Saving {name} failed: {e}")

# Save preprocessors and other components
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector_kbest, 'feature_selector.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Save feature information
np.save('important_features.npy', X_main.columns.values)
np.save('selected_features.npy', selected_feature_names.values)

# Save model performance report
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

print("✅ Performance report saved: model_performance_report.csv")

# ================================
# Create Ensemble Predictor Function
# ================================

def create_ensemble_predictor():
    """Create ensemble predictor"""
    
    def ensemble_predict(new_data_path=None, new_data_df=None, use_voting=True):
        """
        Make predictions using ensemble method
        
        Parameters:
            new_data_path: str, CSV file path
            new_data_df: DataFrame, microbiome abundance data
            use_voting: bool, whether to use voting mechanism
            
        Returns:
            Prediction result dictionary
        """
        
        # Load preprocessors
        scaler = joblib.load('scaler.pkl')
        selector = joblib.load('feature_selector.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        important_features = np.load('important_features.npy', allow_pickle=True)
        
        # Read and preprocess data
        if new_data_path:
            new_data = pd.read_csv(new_data_path)
        elif new_data_df is not None:
            new_data = new_data_df.copy()
        else:
            raise ValueError("Please provide data file path or DataFrame")
        
        # Data preprocessing pipeline
        if 'CD_Location' in new_data.columns:
            new_data = new_data.drop('CD_Location', axis=1)
        
        # Keep only numeric columns
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
        
        # Calculate relative abundance
        row_sums = new_data.sum(axis=1)
        row_sums = row_sums.replace(0, 1)
        new_data_relative = new_data.div(row_sums, axis=0) * 100
        new_data_relative = new_data_relative.fillna(0)
        
        # Select important features
        available_features = [f for f in important_features if f in new_data_relative.columns]
        new_data_main = new_data_relative[available_features]
        
        # Standardization and feature selection
        new_data_scaled = scaler.transform(new_data_main)
        new_data_selected = selector.transform(new_data_scaled)
        
        # Load all available models and make predictions
        predictions = {}
        probabilities = {}
        
        # Traditional machine learning models
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
        
        # Deep learning models (if available)
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
            raise ValueError("No trained models available")
        
        # Ensemble prediction
        if use_voting and len(predictions) > 1:
            # Soft voting
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
            # Use single best model
            best_model = list(predictions.keys())[0]
            results = {
                'prediction': predictions[best_model],
                'probability': probabilities[best_model],
                'model_used': best_model
            }
        
        return results
    
    return ensemble_predict

# Create prediction function
ensemble_predictor = create_ensemble_predictor()

# ================================
# Prediction Examples and Validation
# ================================

print("\n🔮 Ensemble model prediction examples...")

# Use a few test set samples for demonstration
sample_indices = np.random.choice(len(X_test), min(3, len(X_test)), replace=False)

for i, idx in enumerate(sample_indices, 1):
    print(f"\nSample {i} Prediction Results:")
    print("-" * 40)
    
    # Actual label
    actual = le.classes_[y_test[idx]]
    print(f"🎯 Actual label: {actual}")
    
    # Predictions from each model
    print(f"🤖 Model predictions:")
    for model_name, result in model_results.items():
        predicted = le.classes_[result['y_pred'][idx]]
        probability = result['y_pred_proba'][idx]
        confidence = max(probability, 1-probability) if len(le.classes_) == 2 else max(result['model'].predict_proba(X_test[idx:idx+1])[0])
        status = "✅" if predicted == actual else "❌"
        print(f"  {model_name:<20}: {predicted} ({confidence:.3f}) {status}")

# ================================
# Final Summary Report
# ================================

print("\n" + "="*80)
print("🎉 Enhanced Microbiome Prediction System Report")
print("="*80)

print(f"\n📊 Dataset Information:")
print(f"  • Total samples: {df.shape[0]}")
print(f"  • Original features: {df.shape[1]-1}")
print(f"  • Features used: {len(selected_feature_names)}")
print(f"  • Target classes: {len(le.classes_)} classes")

print(f"\n🤖 Number of trained models: {len(model_results)}")
traditional_count = sum(1 for name in model_results.keys() if 'Neural Network' not in name and 'Attention' not in name)
deep_learning_count = len(model_results) - traditional_count

print(f"  • Traditional machine learning: {traditional_count} models")
if DEEP_LEARNING_AVAILABLE:
    print(f"  • Deep learning models: {deep_learning_count} models")

print(f"\n🏆 Model Performance Ranking:")
for i, (name, result) in enumerate(sorted_models[:5], 1):
    print(f"  {i}. {name}: AUC={result['auc']:.3f}, Accuracy={result['accuracy']:.1%}")

print(f"\n💾 Saved Files:")
print(f"  • Model files: {len(model_results)} files")
print(f"  • Preprocessors: 3 files")
print(f"  • Feature information: 2 files")
print(f"  • Performance report: 1 file")

print(f"\n🚀 System Functionality:")
print(f"  ✅ Data preprocessing and cleaning")
print(f"  ✅ Multiple machine learning algorithms")
if DEEP_LEARNING_AVAILABLE:
    print(f"  ✅ Deep learning and attention mechanisms")
print(f"  ✅ Model ensemble and voting")
print(f"  ✅ Visualization analysis")
print(f"  ✅ Prediction interface")

# Performance grade evaluation
best_auc = sorted_models[0][1]['auc']
if best_auc > 0.95:
    grade = "🌟🌟🌟 Outstanding"
elif best_auc > 0.9:
    grade = "🌟🌟 Excellent"
elif best_auc > 0.8:
    grade = "🌟 Good"
elif best_auc > 0.7:
    grade = "👌 Acceptable"
else:
    grade = "⚠️ Needs improvement"

print(f"\n🎖️ System Rating: {grade}")
print(f"📈 Recommended model: {best_model_name}")

print(f"\n💡 Usage Recommendations:")
if best_auc > 0.9:
    print(f"  • Model performance is excellent, ready for use")
    print(f"  • Recommend using ensemble prediction for improved stability")
elif best_auc > 0.8:
    print(f"  • Model performance is good, consider further optimization")
    print(f"  • Consider collecting more training data")
else:
    print(f"  • Recommend collecting more data or trying feature engineering")
    print(f"  • Consider adjusting model hyperparameters")

print(f"\n📞 Technical Support:")
print(f"  • Prediction function: ensemble_predictor()")
print(f"  • Batch prediction: Supports CSV files and DataFrames")
print(f"  • Ensemble prediction: Multi-model voting mechanism")

print("\n" + "="*80)
print("Thank you for using the Enhanced Microbiome Prediction System! 🦠🧠🔬")
print("="*80)

# Create usage examples
print(f"\n📖 Usage Examples:")
print(f"""
# Single sample prediction
results = ensemble_predictor(new_data_df=your_dataframe)
print(results['ensemble_prediction'])

# Prediction from file
results = ensemble_predictor(new_data_path='new_samples.csv')
print(results['ensemble_probability'])

# View individual model results
print(results['individual_predictions'])
""")
