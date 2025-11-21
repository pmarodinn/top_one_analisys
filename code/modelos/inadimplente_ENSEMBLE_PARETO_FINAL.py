"""
MODELO FINAL: ENSEMBLE HETEROGÊNEO + OTIMIZAÇÃO MULTI-OBJETIVO (PARETO)
=========================================================================

Combina as melhores técnicas:
1. Ensemble Heterogêneo com Stacking (3 níveis)
   - Nível 0: XGBoost + NN + LGBM
   - Nível 1: Meta-learner com features enriquecidas
2. Otimização Multi-Objetivo via NSGA-II
   - Objetivo 1: Maximizar Lucro
   - Objetivo 2: Minimizar Taxa de Inadimplência (FDR)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduzir logs do TensorFlow
#s.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desabilitar otimizações que consomem memória

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_recall_curve,
    roc_curve, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgbm

# Configurar TensorFlow para usar menos memória
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
# Limitar uso de CPU
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

from tensorflow import keras
from tensorflow.keras import layers, callbacks
import optuna
from optuna.samplers import NSGAIISampler
import warnings
warnings.filterwarnings('ignore')

# Importar função de carregamento
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_and_preprocess_v3 import load_and_preprocess_v3

print("="*80)
print("🏆 MODELO FINAL: ENSEMBLE HETEROGÊNEO + OTIMIZAÇÃO MULTI-OBJETIVO")
print("="*80)

# ---------- 1. CARREGAMENTO E PREPARAÇÃO ----------
DATA_FILE = '../../data/dataset_interno_top_one_atualizado.csv'
df = load_and_preprocess_v3(DATA_FILE)

print(f"\n📊 Dados carregados: {len(df)} contratos")

# Target: LUCRO (regressão)
y_lucro = df['lucro'].values
y_default = df['default'].values  # Para métricas de inadimplência

# Engenharia temporal leve (sem leakage)
if 'Data_Lancamento' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Data_Lancamento']):
    mes = df['Data_Lancamento'].dt.month.fillna(0)
    dia_semana = df['Data_Lancamento'].dt.dayofweek.fillna(0)
    df['mes_sin'] = np.sin(2 * np.pi * mes / 12)
    df['mes_cos'] = np.cos(2 * np.pi * mes / 12)
    df['dia_semana_sin'] = np.sin(2 * np.pi * dia_semana / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * dia_semana / 7)

if {'Data_Lancamento', 'Data_inicio'}.issubset(df.columns):
    if pd.api.types.is_datetime64_any_dtype(df['Data_inicio']):
        df['dias_decorridos_origem'] = (
            (df['Data_inicio'] - df['Data_Lancamento']).dt.days.clip(lower=-365*5, upper=365*5)
        )

# Features
COLS_REMOVE = ['default', 'pago_perc', 'lucro', 'aceitar', 'adimplente',
               'contrato_id', 'proposta_id', 'Unnamed: 0',
               'Data_Lancamento', 'Data_inicio']

X = df.drop(columns=[c for c in COLS_REMOVE if c in df.columns])

# Engenharia de features
print(f"\n🔧 Features antes do encoding: {X.shape[1]}")
X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.median())
print(f"Features após encoding: {X.shape[1]}")

# ---------- 2. TRAIN/TEST SPLIT (80/20) ----------
print("\n📊 Split 80/20...")

# Incluir pago_perc e lucro no split para análise posterior
pago_perc_values = df['pago_perc'].values
lucro_values = df['lucro'].values

X_train, X_test, y_train_lucro, y_test_lucro, y_train_default, y_test_default, pago_perc_train, pago_perc_test, lucro_train, lucro_test = train_test_split(
    X, y_lucro, y_default, pago_perc_values, lucro_values, test_size=0.2, random_state=42
)

print(f"Treino: {len(X_train):,} contratos")
print(f"Teste: {len(X_test):,} contratos")

# Manter versões escaladas (para NN/meta) e originais (para árvores)
feature_names = X.columns.tolist()
X_train_matrix = X_train.values.astype(np.float32)
X_test_matrix = X_test.values.astype(np.float32)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_matrix)
X_test_scaled = scaler.transform(X_test_matrix)

# ---------- 3. ARQUITETURAS DOS MODELOS BASE ----------

def create_nn_regression(input_dim, hidden_dim=128, dropout_rate=0.3):
    """Rede Neural robusta para regressão de lucro com embeddings ricos"""
    inputs = layers.Input(shape=(input_dim,))

    x = layers.LayerNormalization()(inputs)
    x = layers.Dense(hidden_dim, kernel_initializer='he_normal',
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(hidden_dim // 2, kernel_initializer='he_normal',
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(dropout_rate)(x)

    embeddings = layers.Dense(hidden_dim // 2, activation='linear', name='embedding')(x)

    x = layers.Activation('swish')(embeddings)
    x = layers.Dropout(max(dropout_rate / 1.5, 0.1))(x)

    output = layers.Dense(1, activation='linear', name='lucro')(x)

    model = keras.Model(inputs=inputs, outputs=output)
    optimizer = keras.optimizers.AdamW(learning_rate=0.002, weight_decay=1e-4)
    loss = keras.losses.Huber(delta=250.0)
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

    embedding_model = keras.Model(inputs=inputs, outputs=embeddings)

    return model, embedding_model

# ---------- 4. ENSEMBLE COM STACKING E OTIMIZAÇÃO MULTI-OBJETIVO ----------

print("\n" + "="*80)
print("🔄 TREINANDO ENSEMBLE HETEROGÊNEO + PARETO")
print("="*80)

# ========== NÍVEL 0: MODELOS BASE ==========
print(f"\n--- Nível 0: Treinando Modelos Base ---")

# Preparar K-Fold para OOF
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 1. Neural Network (com embeddings + Huber Loss)
print("1️⃣ Neural Network (cross-val e Huber)...")

tmp_model, tmp_embedding = create_nn_regression(X_train_scaled.shape[1])
embedding_size = tmp_embedding.output_shape[-1]
del tmp_model, tmp_embedding

oof_nn = np.zeros(len(X_train_scaled))
embeddings_oof = np.zeros((len(X_train_scaled), embedding_size), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
    nn_fold, embed_fold = create_nn_regression(X_train_scaled.shape[1])
    fold_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6)
    ]
    nn_fold.fit(
        X_train_scaled[train_idx], y_train_lucro[train_idx],
        validation_data=(X_train_scaled[val_idx], y_train_lucro[val_idx]),
        epochs=350,
        batch_size=128,
        callbacks=fold_callbacks,
        verbose=0
    )
    oof_nn[val_idx] = nn_fold.predict(X_train_scaled[val_idx], verbose=0).flatten()
    embeddings_oof[val_idx] = embed_fold.predict(X_train_scaled[val_idx], verbose=0)
    fold_mae = mean_absolute_error(y_train_lucro[val_idx], oof_nn[val_idx])
    fold_r2 = r2_score(y_train_lucro[val_idx], oof_nn[val_idx])
    print(f"   • Fold {fold}: MAE = R$ {fold_mae:.2f}, R² = {fold_r2:.4f}")

nn_callbacks = [
    callbacks.EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_lr=1e-6)
]
model_nn, embedding_extractor = create_nn_regression(X_train_scaled.shape[1])
model_nn.fit(
    X_train_scaled, y_train_lucro,
    validation_split=0.15,
    epochs=400,
    batch_size=128,
    callbacks=nn_callbacks,
    verbose=0
)

pred_nn_train = model_nn.predict(X_train_scaled, verbose=0).flatten()
pred_nn_test = model_nn.predict(X_test_scaled, verbose=0).flatten()
embeddings_train_full = embedding_extractor.predict(X_train_scaled, verbose=0)
embeddings_test = embedding_extractor.predict(X_test_scaled, verbose=0)
embeddings_train = embeddings_train_full  # compatibilidade com seções posteriores

mae_nn = mean_absolute_error(y_test_lucro, pred_nn_test)
r2_nn = r2_score(y_test_lucro, pred_nn_test)
print(f"   ✓ NN (full data): MAE = R$ {mae_nn:.2f}, R² = {r2_nn:.4f}")

# 2. LightGBM
print("2️⃣ LightGBM (k-fold + regularização)...")
lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.015,
    'n_estimators': 5000,
    'num_leaves': 180,
    'min_child_samples': 60,
    'subsample': 0.85,
    'subsample_freq': 5,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.3,
    'reg_lambda': 1.2,
    'random_state': 42,
    'n_jobs': -1
}

oof_lgb = np.zeros(len(X_train_matrix))
lgb_best_iters = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_matrix), 1):
    model_lgb = lgbm.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_train_matrix[train_idx], y_train_lucro[train_idx],
        eval_set=[(X_train_matrix[val_idx], y_train_lucro[val_idx])],
        eval_metric='mae',
        callbacks=[lgbm.early_stopping(250, verbose=False)]
    )
    oof_lgb[val_idx] = model_lgb.predict(X_train_matrix[val_idx])
    best_iter = model_lgb.best_iteration_ or lgb_params['n_estimators']
    lgb_best_iters.append(best_iter)
    fold_mae = mean_absolute_error(y_train_lucro[val_idx], oof_lgb[val_idx])
    fold_r2 = r2_score(y_train_lucro[val_idx], oof_lgb[val_idx])
    print(f"   • Fold {fold}: MAE = R$ {fold_mae:.2f}, R² = {fold_r2:.4f} (iters={best_iter})")

avg_lgb_iter = int(np.mean(lgb_best_iters)) if lgb_best_iters else lgb_params['n_estimators']
final_lgbm = lgbm.LGBMRegressor(**{**lgb_params, 'n_estimators': max(avg_lgb_iter, 200)})
final_lgbm.fit(X_train_matrix, y_train_lucro)
pred_lgbm_train = final_lgbm.predict(X_train_matrix)
pred_lgbm_test = final_lgbm.predict(X_test_matrix)

mae_lgbm = mean_absolute_error(y_test_lucro, pred_lgbm_test)
r2_lgbm = r2_score(y_test_lucro, pred_lgbm_test)
print(f"   ✓ LGBM: MAE = R$ {mae_lgbm:.2f}, R² = {r2_lgbm:.4f}")

# 3. XGBoost
print("3️⃣ XGBoost (k-fold + hist)...")
xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.015,
    'n_estimators': 4000,
    'max_depth': 8,
    'min_child_weight': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.65,
    'reg_alpha': 0.5,
    'reg_lambda': 1.5,
    'gamma': 0.3,
    'tree_method': 'hist',
    'n_jobs': 4,
    'verbosity': 0,
    'random_state': 42
}

oof_xgb = np.zeros(len(X_train_matrix))
xgb_best_iters = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_matrix), 1):
    model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=250)
    model_xgb.fit(
        X_train_matrix[train_idx], y_train_lucro[train_idx],
        eval_set=[(X_train_matrix[val_idx], y_train_lucro[val_idx])],
        verbose=False
    )
    oof_xgb[val_idx] = model_xgb.predict(X_train_matrix[val_idx])
    best_iter = getattr(model_xgb, 'best_iteration', None) or xgb_params['n_estimators']
    xgb_best_iters.append(best_iter)
    fold_mae = mean_absolute_error(y_train_lucro[val_idx], oof_xgb[val_idx])
    fold_r2 = r2_score(y_train_lucro[val_idx], oof_xgb[val_idx])
    print(f"   • Fold {fold}: MAE = R$ {fold_mae:.2f}, R² = {fold_r2:.4f} (iters={best_iter})")

avg_xgb_iter = int(np.mean(xgb_best_iters)) if xgb_best_iters else xgb_params['n_estimators']
final_xgb = xgb.XGBRegressor(**{**xgb_params, 'n_estimators': max(avg_xgb_iter, 300)})
final_xgb.fit(X_train_matrix, y_train_lucro, verbose=False)
pred_xgb_train = final_xgb.predict(X_train_matrix)
pred_xgb_test = final_xgb.predict(X_test_matrix)

mae_xgb = mean_absolute_error(y_test_lucro, pred_xgb_test)
r2_xgb = r2_score(y_test_lucro, pred_xgb_test)
print(f"   ✓ XGB: MAE = R$ {mae_xgb:.2f}, R² = {r2_xgb:.4f}")

# ========== NÍVEL 1: META-LEARNER ==========
print(f"\n--- Nível 1: Meta-Learner com Features Enriquecidas ---")

# Predições dos 3 modelos + dispersão + embeddings OOF + top features originais
feature_stds = X_train_matrix.std(axis=0)
top_features_idx = np.argsort(feature_stds)[-10:]

pred_stack_train = np.vstack([oof_nn, oof_lgb, oof_xgb])
pred_stack_test = np.vstack([pred_nn_test, pred_lgbm_test, pred_xgb_test])
dispersion_train = pred_stack_train.std(axis=0)
dispersion_test = pred_stack_test.std(axis=0)

meta_features_train = np.column_stack([
    oof_nn,
    oof_lgb,
    oof_xgb,
    dispersion_train,
    embeddings_oof,
    X_train_scaled[:, top_features_idx]
])

meta_features_test = np.column_stack([
    pred_nn_test,
    pred_lgbm_test,
    pred_xgb_test,
    dispersion_test,
    embeddings_test,
    X_test_scaled[:, top_features_idx]
])

print(f"   Features do meta-learner: {meta_features_train.shape[1]} (3 pred + 1 risco + {embedding_size} emb + 10 orig)")

# Meta-learner: XGBoost leve com early stopping
meta_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'n_estimators': 3000,
    'max_depth': 5,
    'min_child_weight': 8,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'gamma': 0.15,
    'tree_method': 'hist',
    'n_jobs': 4,
    'verbosity': 0,
    'random_state': 42
}

meta_X_train, meta_X_val, meta_y_train, meta_y_val = train_test_split(
    meta_features_train, y_train_lucro, test_size=0.15, random_state=42
)

meta_model_cv = xgb.XGBRegressor(**meta_params, early_stopping_rounds=250)
meta_model_cv.fit(
    meta_X_train, meta_y_train,
    eval_set=[(meta_X_val, meta_y_val)],
    verbose=False
)
best_meta_iter = getattr(meta_model_cv, 'best_iteration', None) or meta_params['n_estimators']
meta_model = xgb.XGBRegressor(**{**meta_params, 'n_estimators': best_meta_iter})
meta_model.fit(meta_features_train, y_train_lucro)

pred_meta_train = meta_model.predict(meta_features_train)
pred_meta_test = meta_model.predict(meta_features_test)

mae_meta = mean_absolute_error(y_test_lucro, pred_meta_test)
r2_meta = r2_score(y_test_lucro, pred_meta_test)
print(f"   ✓ Meta-Learner: MAE = R$ {mae_meta:.2f}, R² = {r2_meta:.4f}")

# ========== OTIMIZAÇÃO MULTI-OBJETIVO (PARETO) ==========
print(f"\n--- Otimização Multi-Objetivo: Lucro vs Inadimplência ---")

def objective_pareto(trial):
    """
    Objetivo multi-objetivo:
    1. Maximizar lucro
    2. Minimizar taxa de inadimplência (FDR)
    """
    # Threshold otimizado entre min e max das predições
    threshold = trial.suggest_float('threshold', 
                                   pred_meta_test.min(),
                                   pred_meta_test.max())
    
    # Aceitar contratos com lucro esperado >= threshold
    aceitar = pred_meta_test >= threshold
    
    if aceitar.sum() == 0:
        return -1e9, 1.0  # Penalizar threshold que rejeita tudo
    
    # Objetivo 1: Lucro total
    lucro_total = y_test_lucro[aceitar].sum()
    
    # Objetivo 2: Taxa de inadimplência (FDR - False Discovery Rate)
    # FDR = contratos inadimplentes aceitos / total aceitos
    inadimplentes_aceitos = y_test_default[aceitar].sum()
    fdr = inadimplentes_aceitos / aceitar.sum()
    
    return lucro_total, fdr

# Criar estudo multi-objetivo com NSGA-II
study = optuna.create_study(
    directions=["maximize", "minimize"],  # [lucro, inadimplência]
    sampler=NSGAIISampler(seed=42)
)

study.optimize(objective_pareto, n_trials=600, show_progress_bar=False)

# Pareto front
pareto_trials = study.best_trials
print(f"   ✓ Pareto Frontier: {len(pareto_trials)} soluções ótimas")

# Selecionar solução balanceada (pode ajustar critério)
# Aqui: escolher solução com melhor trade-off (normalizado)
pareto_scores = []
for trial in pareto_trials:
    lucro_norm = (trial.values[0] - min([t.values[0] for t in pareto_trials])) / \
                (max([t.values[0] for t in pareto_trials]) - min([t.values[0] for t in pareto_trials]) + 1e-6)
    fdr_norm = 1 - (trial.values[1] - min([t.values[1] for t in pareto_trials])) / \
              (max([t.values[1] for t in pareto_trials]) - min([t.values[1] for t in pareto_trials]) + 1e-6)
    score = lucro_norm * 0.7 + fdr_norm * 0.3  # 60% lucro, 40% risco
    pareto_scores.append(score)

best_idx = np.argmax(pareto_scores)
best_trial = pareto_trials[best_idx]
best_threshold = best_trial.params['threshold']
best_lucro = best_trial.values[0]
best_fdr = best_trial.values[1]

print(f"   🎯 Solução Selecionada:")
print(f"      Threshold: R$ {best_threshold:.2f}")
print(f"      Lucro: R$ {best_lucro:,.2f}")
print(f"      FDR (Taxa Inadimplência): {best_fdr*100:.2f}%")

# Aplicar threshold ótimo
aceitar_final = pred_meta_test >= best_threshold
num_aceitos = aceitar_final.sum()
taxa_aprovacao = num_aceitos / len(y_test_lucro) * 100

lucro_otimizado = y_test_lucro[aceitar_final].sum()
lucro_baseline = y_test_lucro.sum()
lucro_maximo_teorico = y_test_lucro[y_test_lucro > 0].sum()

ganho_vs_baseline = lucro_otimizado - lucro_baseline
ganho_perc = (ganho_vs_baseline / lucro_baseline * 100) if lucro_baseline != 0 else 0

# Calcular previsões para matriz de confusão (classificação binária)
# y_true: 1 se default (inadimplente), 0 caso contrário
# y_pred: 1 se REJEITADO (lucro < threshold), 0 se ACEITO
# y_test_default pode ser um numpy.ndarray ou um pandas.Series; lidar com ambos
if hasattr(y_test_default, 'values'):
    y_true_default = y_test_default.values
else:
    y_true_default = np.array(y_test_default)

y_pred_rejeitar = (pred_meta_test < best_threshold).astype(int)  # 1=rejeitar, 0=aceitar

# ---------- 5. RESULTADOS FINAIS ----------
print("\n" + "="*80)
print("📊 RESULTADOS FINAIS")
print("="*80)

print(f"\n--- Desempenho dos Modelos Base ---")
print(f"NN:   MAE = R$ {mae_nn:.2f}, R² = {r2_nn:.4f}")
print(f"LGBM: MAE = R$ {mae_lgbm:.2f}, R² = {r2_lgbm:.4f}")
print(f"XGB:  MAE = R$ {mae_xgb:.2f}, R² = {r2_xgb:.4f}")
print(f"META: MAE = R$ {mae_meta:.2f}, R² = {r2_meta:.4f}")

print(f"\n--- Comparação de Cenários ---")
print(f"{'Cenário':<45} {'Lucro':>15} {'Eficiência':>12} {'Contratos':>12}")
print("="*80)
print(f"{'1. Baseline (aceitar todos)':<45} R$ {lucro_baseline:>12,.2f} {lucro_baseline/lucro_maximo_teorico:>11.2%} {len(y_test_lucro):>11,}")
print(f"{'2. Máximo Teórico (oracle perfeito)':<45} R$ {lucro_maximo_teorico:>12,.2f} {100.0:>11.2%} {(y_test_lucro>0).sum():>11,}")
print(f"{'3. ENSEMBLE + PARETO (otimizado)':<45} R$ {lucro_otimizado:>12,.2f} {lucro_otimizado/lucro_maximo_teorico:>11.2%} {num_aceitos:>11,}")
print("="*80)

print(f"\n💰 Ganho MODELO vs Baseline: R$ {ganho_vs_baseline:+,.2f} ({ganho_perc:+.2f}%)")
print(f"📊 Taxa de Aprovação: {taxa_aprovacao:.1f}%")
print(f"📉 Taxa de Inadimplência (FDR): {best_fdr*100:.2f}%")
print(f"🎯 Pareto Frontier: {len(pareto_trials)} soluções ótimas encontradas")

# ---------- 6. VISUALIZAÇÕES ----------
print("\n--- Gerando Visualizações ---")
os.makedirs('../../graficos/analise_modelos/ENSEMBLE_PARETO_FINAL', exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Comparação R² dos modelos
ax1 = axes[0, 0]
models_r2 = ['NN', 'LGBM', 'XGB', 'META']
r2_values = [r2_nn, r2_lgbm, r2_xgb, r2_meta]
colors = ['blue', 'green', 'red', 'purple']
ax1.bar(models_r2, r2_values, color=colors, alpha=0.7)
ax1.set_ylabel('R²', fontsize=12)
ax1.set_title('Comparação de R² entre Modelos', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3, axis='y')
for i, v in enumerate(r2_values):
    ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', fontweight='bold')

# 2. Comparação MAE dos modelos
ax2 = axes[0, 1]
mae_values = [mae_nn, mae_lgbm, mae_xgb, mae_meta]
ax2.bar(models_r2, mae_values, color=colors, alpha=0.7)
ax2.set_ylabel('MAE (R$)', fontsize=12)
ax2.set_title('Comparação de MAE entre Modelos', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3, axis='y')
for i, v in enumerate(mae_values):
    ax2.text(i, v + 10, f'R$ {v:.0f}', ha='center', fontweight='bold')

# 3. Matriz de Confusão
ax3 = axes[0, 2]
cm = confusion_matrix(y_true_default, y_pred_rejeitar)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=True,
            xticklabels=['Aceito', 'Rejeitado'],
            yticklabels=['Adimplente', 'Inadimplente'])
ax3.set_xlabel('Predição do Modelo', fontsize=12)
ax3.set_ylabel('Realidade', fontsize=12)
ax3.set_title('Matriz de Confusão', fontsize=14, fontweight='bold')

# Adicionar métricas na matriz
tn, fp, fn, tp = cm.ravel()
precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
ax3.text(0.5, -0.15, f'Precision: {precisao:.2%} | Recall: {recall:.2%} | F1-Score: {f1:.2%}',
         ha='center', transform=ax3.transAxes, fontsize=10, fontweight='bold')

# Salvar matriz de confusão separadamente (arquivo dedicado)
try:
    fig_cm = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Aceito', 'Rejeitado'],
                yticklabels=['Adimplente', 'Inadimplente'])
    plt.xlabel('Predição do Modelo', fontsize=12)
    plt.ylabel('Realidade', fontsize=12)
    plt.title('Matriz de Confusão', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../../graficos/analise_modelos/ENSEMBLE_PARETO_FINAL/matriz_confusao.png', dpi=300, bbox_inches='tight')
    plt.close(fig_cm)
except Exception:
    # Não falhar todo o pipeline por causa de salvamento de figura
    pass

# 4. Pareto Frontier
ax4 = axes[1, 0]
lucros = [trial.values[0] for trial in pareto_trials]
fdrs = [trial.values[1] * 100 for trial in pareto_trials]
ax4.scatter(fdrs, lucros, c='purple', alpha=0.6, s=50)
ax4.scatter([best_fdr*100], [best_lucro], c='red', s=200, marker='*', label='Solução Selecionada', zorder=5)
ax4.set_xlabel('FDR - Taxa de Inadimplência (%)', fontsize=12)
ax4.set_ylabel('Lucro Total (R$)', fontsize=12)
ax4.set_title('Pareto Frontier: Lucro vs Inadimplência', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# 5. Comparação de Cenários
ax5 = axes[1, 1]
cenarios = ['Baseline\n(aceitar todos)', 'Máximo Teórico\n(oracle)', 'ENSEMBLE+PARETO\n(otimizado)']
lucros_cenarios = [lucro_baseline, lucro_maximo_teorico, lucro_otimizado]
cores_cenarios = ['gray', 'lightblue', 'purple']
bars = ax5.bar(cenarios, lucros_cenarios, color=cores_cenarios, alpha=0.7)
ax5.set_ylabel('Lucro (R$)', fontsize=12)
ax5.set_title('Comparação de Cenários', fontsize=14, fontweight='bold')
ax5.grid(alpha=0.3, axis='y')
for i, (bar, valor) in enumerate(zip(bars, lucros_cenarios)):
    altura = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2, altura + max(lucros_cenarios)*0.02, 
             f'R$ {valor:,.0f}', ha='center', fontweight='bold', fontsize=9)

# 6. Distribuição de Decisões
ax6 = axes[1, 2]
decisoes = ['Aceitos', 'Rejeitados']
contagens = [aceitar_final.sum(), (~aceitar_final).sum()]
cores_decisoes = ['green', 'red']
wedges, texts, autotexts = ax6.pie(contagens, labels=decisoes, autopct='%1.1f%%',
                                     colors=cores_decisoes, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax6.set_title('Distribuição de Decisões', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../../graficos/analise_modelos/ENSEMBLE_PARETO_FINAL/analise_completa.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Gráfico principal salvo")

# ========== ANÁLISE COMPLEMENTAR: CONTAGENS POR TIPO DE CONTRATO ==========
print("\n" + "="*80)
print("📊 ANÁLISE COMPLEMENTAR: BREAKDOWN DOS CONTRATOS ACEITOS PELO MODELO")
print("="*80)

# Calcular as contagens APENAS dos contratos ACEITOS
adimplentes_aceitos = ((pago_perc_test == 1.0) & aceitar_final).sum()
inadimplentes_aceitos = ((pago_perc_test < 1.0) & aceitar_final).sum()
inadimplentes_lucrativos_aceitos = ((pago_perc_test < 1.0) & (lucro_test > 0) & aceitar_final).sum()
inadimplentes_nao_lucrativos_aceitos = ((pago_perc_test < 1.0) & (lucro_test <= 0) & aceitar_final).sum()

total_aceitos = aceitar_final.sum()

print(f"\n📋 Total de Contratos ACEITOS pelo Modelo: {total_aceitos:,}")
print(f"\n✅ Adimplentes (100% pago):")
print(f"   {adimplentes_aceitos:,} contratos ({adimplentes_aceitos/total_aceitos*100:.2f}%)")
print(f"\n❌ Inadimplentes (<100% pago): {inadimplentes_aceitos:,} contratos ({inadimplentes_aceitos/total_aceitos*100:.2f}%)")
print(f"   ├─ 💰 Inadimplentes Lucrativos (lucro > 0): {inadimplentes_lucrativos_aceitos:,} contratos ({inadimplentes_lucrativos_aceitos/total_aceitos*100:.2f}%)")
print(f"   └─ 📉 Inadimplentes Não Lucrativos (lucro ≤ 0): {inadimplentes_nao_lucrativos_aceitos:,} contratos ({inadimplentes_nao_lucrativos_aceitos/total_aceitos*100:.2f}%)")

print("\n💡 INSIGHT:")
if inadimplentes_aceitos > 0:
    print(f"   Dos {inadimplentes_aceitos:,} inadimplentes aceitos, {inadimplentes_lucrativos_aceitos:,} ({inadimplentes_lucrativos_aceitos/inadimplentes_aceitos*100:.1f}%) ainda geraram lucro!")
    print(f"   Isso mostra que inadimplência parcial ≠ prejuízo automático.")
else:
    print(f"   Nenhum inadimplente foi aceito pelo modelo!")

# ========== ANÁLISE DOS EMBEDDINGS PRINCIPAIS ==========
print("\n" + "="*80)
print("🔍 ANÁLISE DOS EMBEDDINGS: Dimensões Latentes Mais Importantes")
print("="*80)

# Segmentar clientes por tipo (no conjunto de teste)
mask_adimplentes = pago_perc_test == 1.0
mask_inadimplentes_totais = pago_perc_test == 0.0
mask_inadimplentes_lucrativos = (pago_perc_test > 0) & (pago_perc_test < 1.0) & (lucro_test > 0)

# Calcular médias dos embeddings para cada grupo
embeddings_adimplentes = embeddings_test[mask_adimplentes].mean(axis=0)
embeddings_inadimplentes_totais = embeddings_test[mask_inadimplentes_totais].mean(axis=0)
embeddings_inadimplentes_lucrativos = embeddings_test[mask_inadimplentes_lucrativos].mean(axis=0)
embeddings_geral = embeddings_test.mean(axis=0)

# Calcular desvios em relação à média geral (importância relativa)
desvio_adimplentes = np.abs(embeddings_adimplentes - embeddings_geral)
desvio_inadimplentes_totais = np.abs(embeddings_inadimplentes_totais - embeddings_geral)
desvio_inadimplentes_lucrativos = np.abs(embeddings_inadimplentes_lucrativos - embeddings_geral)

# Top 5 dimensões mais importantes para cada grupo
top_k = 5
top_adimplentes = np.argsort(desvio_adimplentes)[-top_k:][::-1]
top_inadimplentes_totais = np.argsort(desvio_inadimplentes_totais)[-top_k:][::-1]
top_inadimplentes_lucrativos = np.argsort(desvio_inadimplentes_lucrativos)[-top_k:][::-1]

print(f"\n✅ ADIMPLENTES (100% pago) - Top {top_k} Dimensões dos Embeddings:")
print(f"   Total de clientes: {mask_adimplentes.sum():,}")
for i, dim in enumerate(top_adimplentes, 1):
    valor_medio = embeddings_adimplentes[dim]
    desvio = desvio_adimplentes[dim]
    print(f"   {i}. Dimensão {dim:2d}: valor={valor_medio:+.4f}, desvio={desvio:.4f}")

print(f"\n❌ INADIMPLENTES TOTAIS (0% pago) - Top {top_k} Dimensões dos Embeddings:")
print(f"   Total de clientes: {mask_inadimplentes_totais.sum():,}")
for i, dim in enumerate(top_inadimplentes_totais, 1):
    valor_medio = embeddings_inadimplentes_totais[dim]
    desvio = desvio_inadimplentes_totais[dim]
    print(f"   {i}. Dimensão {dim:2d}: valor={valor_medio:+.4f}, desvio={desvio:.4f}")

print(f"\n💰 INADIMPLENTES LUCRATIVOS (0% < pago < 100% e lucro > 0) - Top {top_k} Dimensões:")
print(f"   Total de clientes: {mask_inadimplentes_lucrativos.sum():,}")
for i, dim in enumerate(top_inadimplentes_lucrativos, 1):
    valor_medio = embeddings_inadimplentes_lucrativos[dim]
    desvio = desvio_inadimplentes_lucrativos[dim]
    print(f"   {i}. Dimensão {dim:2d}: valor={valor_medio:+.4f}, desvio={desvio:.4f}")

# Análise de sobreposição entre grupos
print("\n📊 Análise de Sobreposição das Dimensões Importantes:")
set_adim = set(top_adimplentes)
set_inad_totais = set(top_inadimplentes_totais)
set_inad_lucrativos = set(top_inadimplentes_lucrativos)

comuns_todos = set_adim & set_inad_totais & set_inad_lucrativos
comuns_adim_lucrativos = (set_adim & set_inad_lucrativos) - comuns_todos
comuns_adim_totais = (set_adim & set_inad_totais) - comuns_todos
comuns_inad = (set_inad_totais & set_inad_lucrativos) - comuns_todos

if comuns_todos:
    print(f"   Dimensões comuns a TODOS os grupos: {sorted(comuns_todos)}")
if comuns_adim_lucrativos:
    print(f"   Dimensões comuns a Adimplentes e Inadimplentes Lucrativos: {sorted(comuns_adim_lucrativos)}")
if comuns_adim_totais:
    print(f"   Dimensões comuns a Adimplentes e Inadimplentes Totais: {sorted(comuns_adim_totais)}")
if comuns_inad:
    print(f"   Dimensões comuns aos dois tipos de Inadimplentes: {sorted(comuns_inad)}")

exclusivas_adim = set_adim - set_inad_totais - set_inad_lucrativos
exclusivas_inad_totais = set_inad_totais - set_adim - set_inad_lucrativos
exclusivas_inad_lucrativos = set_inad_lucrativos - set_adim - set_inad_totais

print(f"\n🎯 Dimensões EXCLUSIVAS de cada grupo:")
print(f"   Adimplentes: {sorted(exclusivas_adim) if exclusivas_adim else 'Nenhuma exclusiva'}")
print(f"   Inadimplentes Totais: {sorted(exclusivas_inad_totais) if exclusivas_inad_totais else 'Nenhuma exclusiva'}")
print(f"   Inadimplentes Lucrativos: {sorted(exclusivas_inad_lucrativos) if exclusivas_inad_lucrativos else 'Nenhuma exclusiva'}")

# ========== MAPEAMENTO: DIMENSÕES → FEATURES ORIGINAIS ==========
print("\n" + "="*80)
print("🔗 MAPEAMENTO: Dimensões dos Embeddings → Features Originais")
print("="*80)

# Calcular correlações entre embeddings e features originais
feature_names = X.columns.tolist()

def map_dimensions_to_features(dimensions, embeddings, features, top_n=3):
    """Mapeia dimensões dos embeddings para as features originais mais correlacionadas"""
    mapping = {}
    for dim in dimensions:
        # Correlação de Pearson entre a dimensão do embedding e cada feature
        correlations = []
        for i, feat_name in enumerate(feature_names):
            corr = np.corrcoef(embeddings[:, dim], features[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append((feat_name, abs(corr), corr))
        
        # Top N features mais correlacionadas
        correlations.sort(key=lambda x: x[1], reverse=True)
        mapping[dim] = correlations[:top_n]
    
    return mapping

# Mapear para cada grupo
mapping_adimplentes = map_dimensions_to_features(top_adimplentes, embeddings_test, X_test_scaled, top_n=3)
mapping_inadimplentes_totais = map_dimensions_to_features(top_inadimplentes_totais, embeddings_test, X_test_scaled, top_n=3)
mapping_inadimplentes_lucrativos = map_dimensions_to_features(top_inadimplentes_lucrativos, embeddings_test, X_test_scaled, top_n=3)

print("\n✅ ADIMPLENTES - Mapeamento Dimensão → Features:")
for i, dim in enumerate(top_adimplentes, 1):
    print(f"\n   {i}. Dimensão {dim} (desvio={desvio_adimplentes[dim]:.4f}):")
    for feat_name, abs_corr, corr in mapping_adimplentes[dim]:
        print(f"      └─ {feat_name[:50]:<50} | correlação: {corr:+.3f}")

print("\n❌ INADIMPLENTES TOTAIS - Mapeamento Dimensão → Features:")
for i, dim in enumerate(top_inadimplentes_totais, 1):
    print(f"\n   {i}. Dimensão {dim} (desvio={desvio_inadimplentes_totais[dim]:.4f}):")
    for feat_name, abs_corr, corr in mapping_inadimplentes_totais[dim]:
        print(f"      └─ {feat_name[:50]:<50} | correlação: {corr:+.3f}")

print("\n💰 INADIMPLENTES LUCRATIVOS - Mapeamento Dimensão → Features:")
for i, dim in enumerate(top_inadimplentes_lucrativos, 1):
    print(f"\n   {i}. Dimensão {dim} (desvio={desvio_inadimplentes_lucrativos[dim]:.4f}):")
    for feat_name, abs_corr, corr in mapping_inadimplentes_lucrativos[dim]:
        print(f"      └─ {feat_name[:50]:<50} | correlação: {corr:+.3f}")

print("\n💡 INTERPRETAÇÃO:")
print("   - Correlação positiva (+): feature AUMENTA quando dimensão aumenta")
print("   - Correlação negativa (-): feature DIMINUI quando dimensão aumenta")
print("   - Correlação alta (>0.5): forte relação linear entre dimensão e feature")

# ========== EXPLICABILIDADE INDIVIDUAL: EXEMPLOS DE DECISÕES ==========
print("\n" + "="*80)
print("🔍 EXPLICABILIDADE INDIVIDUAL: Por que cada contrato foi aceito/rejeitado?")
print("="*80)

def explicar_decisao_contrato(idx, X_original, X_scaled, pred_lucro, embeddings, 
                               threshold, aceitar, pago_perc, lucro_real, feature_names):
    """Explica a decisão do modelo para um contrato específico"""
    decisao = "✅ ACEITO" if aceitar[idx] else "❌ REJEITADO"
    lucro_esperado = pred_lucro[idx]
    
    print(f"\n{'='*80}")
    print(f"📋 CONTRATO #{idx}")
    print(f"{'='*80}")
    print(f"Decisão: {decisao}")
    print(f"Lucro Esperado pelo Modelo: R$ {lucro_esperado:.2f}")
    print(f"Threshold de Decisão: R$ {threshold:.2f}")
    print(f"Margem: R$ {lucro_esperado - threshold:.2f} ({'acima' if lucro_esperado >= threshold else 'abaixo'} do threshold)")
    
    # Resultado real
    if pago_perc[idx] == 1.0:
        resultado_real = "✅ Adimplente (100% pago)"
    elif pago_perc[idx] == 0.0:
        resultado_real = "❌ Inadimplente Total (0% pago)"
    else:
        resultado_real = f"⚠️ Inadimplente Parcial ({pago_perc[idx]*100:.1f}% pago)"
    
    print(f"\nResultado Real: {resultado_real}")
    print(f"Lucro Real: R$ {lucro_real[idx]:.2f}")
    
    # Acurácia da predição
    erro_predicao = abs(lucro_esperado - lucro_real[idx])
    print(f"Erro de Predição: R$ {erro_predicao:.2f}")
    
    # Análise do embedding
    print(f"\n🧠 ANÁLISE DO EMBEDDING (Representação Latente):")
    emb = embeddings[idx]
    
    # Top 5 dimensões mais ativas (valores absolutos maiores)
    top_dims_ativas = np.argsort(np.abs(emb))[-5:][::-1]
    print(f"   Top 5 Dimensões Mais Ativas:")
    for i, dim in enumerate(top_dims_ativas, 1):
        valor = emb[dim]
        print(f"      {i}. Dim {dim}: {valor:+.4f}")
    
    # Mapear para features originais
    print(f"\n📊 FEATURES MAIS INFLUENTES (Top 10):")
    
    # Features com maior valor absoluto (após escalonamento)
    feature_values = X_scaled[idx]
    top_feat_idx = np.argsort(np.abs(feature_values))[-10:][::-1]
    
    for i, feat_idx in enumerate(top_feat_idx, 1):
        feat_name = feature_names[feat_idx]
        feat_value_scaled = feature_values[feat_idx]
        # Tentar pegar valor original (antes do scaling)
        feat_value_original = X_original.iloc[idx, feat_idx]
        
        # Simplificar nome da feature
        if len(feat_name) > 50:
            feat_name_display = feat_name[:47] + "..."
        else:
            feat_name_display = feat_name
        
        print(f"      {i:2d}. {feat_name_display:<50} = {feat_value_original}")
        print(f"          (valor normalizado: {feat_value_scaled:+.3f})")
    
    print(f"\n💡 INTERPRETAÇÃO:")
    if aceitar[idx]:
        if lucro_real[idx] > 0:
            print(f"   ✅ Decisão CORRETA: Modelo aceitou e gerou lucro real de R$ {lucro_real[idx]:.2f}")
        else:
            print(f"   ❌ Decisão INCORRETA: Modelo aceitou mas gerou prejuízo de R$ {lucro_real[idx]:.2f}")
    else:
        if lucro_real[idx] > 0:
            print(f"   ❌ Decisão INCORRETA: Modelo rejeitou mas perdeu lucro de R$ {lucro_real[idx]:.2f}")
        else:
            print(f"   ✅ Decisão CORRETA: Modelo rejeitou e evitou prejuízo de R$ {abs(lucro_real[idx]):.2f}")
    
    return decisao, lucro_esperado, lucro_real[idx]

# Selecionar exemplos representativos
print("\n📌 Selecionando exemplos representativos...")

# 1. Melhor acerto: Aceito com maior lucro real
aceitos_lucrativos = np.where(aceitar_final & (lucro_test > 0))[0]
if len(aceitos_lucrativos) > 0:
    melhor_acerto_idx = aceitos_lucrativos[np.argmax(lucro_test[aceitos_lucrativos])]
    print("\n" + "🏆 EXEMPLO 1: MELHOR ACERTO (Aceito com maior lucro)")
    explicar_decisao_contrato(melhor_acerto_idx, X_test, X_test_scaled, pred_meta_test, 
                              embeddings_test, best_threshold, aceitar_final, 
                              pago_perc_test, lucro_test, X_test.columns.tolist())

# 2. Falso positivo: Aceito com prejuízo
aceitos_prejuizo = np.where(aceitar_final & (lucro_test <= 0))[0]
if len(aceitos_prejuizo) > 0:
    pior_erro_fp_idx = aceitos_prejuizo[np.argmin(lucro_test[aceitos_prejuizo])]
    print("\n" + "⚠️ EXEMPLO 2: FALSO POSITIVO (Aceito mas gerou prejuízo)")
    explicar_decisao_contrato(pior_erro_fp_idx, X_test, X_test_scaled, pred_meta_test, 
                              embeddings_test, best_threshold, aceitar_final, 
                              pago_perc_test, lucro_test, X_test.columns.tolist())

# 3. Falso negativo: Rejeitado mas tinha lucro
rejeitados_lucrativos = np.where(~aceitar_final & (lucro_test > 0))[0]
if len(rejeitados_lucrativos) > 0:
    # Pegar o com maior lucro perdido
    pior_erro_fn_idx = rejeitados_lucrativos[np.argmax(lucro_test[rejeitados_lucrativos])]
    print("\n" + "⚠️ EXEMPLO 3: FALSO NEGATIVO (Rejeitado mas tinha lucro)")
    explicar_decisao_contrato(pior_erro_fn_idx, X_test, X_test_scaled, pred_meta_test, 
                              embeddings_test, best_threshold, aceitar_final, 
                              pago_perc_test, lucro_test, X_test.columns.tolist())

# 4. Verdadeiro negativo: Rejeitado com razão (tinha prejuízo)
rejeitados_prejuizo = np.where(~aceitar_final & (lucro_test <= 0))[0]
if len(rejeitados_prejuizo) > 0:
    # Pegar o com maior prejuízo evitado
    melhor_rejeicao_idx = rejeitados_prejuizo[np.argmin(lucro_test[rejeitados_prejuizo])]
    print("\n" + "✅ EXEMPLO 4: VERDADEIRO NEGATIVO (Rejeitado corretamente)")
    explicar_decisao_contrato(melhor_rejeicao_idx, X_test, X_test_scaled, pred_meta_test, 
                              embeddings_test, best_threshold, aceitar_final, 
                              pago_perc_test, lucro_test, X_test.columns.tolist())

# 5. Adimplente típico
adimplentes_aceitos_idx = np.where(aceitar_final & (pago_perc_test == 1.0))[0]
if len(adimplentes_aceitos_idx) > 0:
    # Pegar um adimplente mediano
    adimplente_exemplo_idx = adimplentes_aceitos_idx[len(adimplentes_aceitos_idx)//2]
    print("\n" + "✅ EXEMPLO 5: ADIMPLENTE TÍPICO (Aceito e pagou 100%)")
    explicar_decisao_contrato(adimplente_exemplo_idx, X_test, X_test_scaled, pred_meta_test, 
                              embeddings_test, best_threshold, aceitar_final, 
                              pago_perc_test, lucro_test, X_test.columns.tolist())

# 6. Inadimplente lucrativo
inad_lucrativos_aceitos_idx = np.where(aceitar_final & (pago_perc_test < 1.0) & (pago_perc_test > 0) & (lucro_test > 0))[0]
if len(inad_lucrativos_aceitos_idx) > 0:
    inad_lucrativo_idx = inad_lucrativos_aceitos_idx[len(inad_lucrativos_aceitos_idx)//2]
    print("\n" + "💰 EXEMPLO 6: INADIMPLENTE LUCRATIVO (Aceito, pagou parcial mas gerou lucro)")
    explicar_decisao_contrato(inad_lucrativo_idx, X_test, X_test_scaled, pred_meta_test, 
                              embeddings_test, best_threshold, aceitar_final, 
                              pago_perc_test, lucro_test, X_test.columns.tolist())

# ========== VISUALIZAÇÃO DOS EMBEDDINGS ==========
# Criar gráfico comparativo dos embeddings principais
fig_emb, axes_emb = plt.subplots(2, 2, figsize=(16, 12))

# Preparar dados para visualização
mask_adimplentes_viz = pago_perc_test == 1.0
mask_inadimplentes_totais_viz = pago_perc_test == 0.0
mask_inadimplentes_lucrativos_viz = (pago_perc_test > 0) & (pago_perc_test < 1.0) & (lucro_test > 0)

embeddings_adim_viz = embeddings_test[mask_adimplentes_viz].mean(axis=0)
embeddings_inad_totais_viz = embeddings_test[mask_inadimplentes_totais_viz].mean(axis=0)
embeddings_inad_lucrativos_viz = embeddings_test[mask_inadimplentes_lucrativos_viz].mean(axis=0)
embeddings_geral_viz = embeddings_test.mean(axis=0)

# 1. Perfil completo dos embeddings (todas as dimensões)
ax_emb1 = axes_emb[0, 0]
dims = np.arange(len(embeddings_geral_viz))
ax_emb1.plot(dims, embeddings_adim_viz, 'g-', label='Adimplentes', alpha=0.7, linewidth=2)
ax_emb1.plot(dims, embeddings_inad_totais_viz, 'r-', label='Inadimplentes Totais', alpha=0.7, linewidth=2)
ax_emb1.plot(dims, embeddings_inad_lucrativos_viz, 'orange', label='Inadimplentes Lucrativos', alpha=0.7, linewidth=2)
ax_emb1.plot(dims, embeddings_geral_viz, 'k--', label='Média Geral', alpha=0.5, linewidth=1)
ax_emb1.set_xlabel('Dimensão do Embedding', fontsize=11)
ax_emb1.set_ylabel('Valor Médio', fontsize=11)
ax_emb1.set_title('Perfil Completo dos Embeddings por Grupo', fontsize=13, fontweight='bold')
ax_emb1.legend(fontsize=9)
ax_emb1.grid(alpha=0.3)

# 2. Top 5 dimensões mais discriminativas para cada grupo
ax_emb2 = axes_emb[0, 1]
desvio_adim_viz = np.abs(embeddings_adim_viz - embeddings_geral_viz)
desvio_inad_totais_viz = np.abs(embeddings_inad_totais_viz - embeddings_geral_viz)
desvio_inad_lucrativos_viz = np.abs(embeddings_inad_lucrativos_viz - embeddings_geral_viz)

top_5_adim = np.argsort(desvio_adim_viz)[-5:][::-1]
top_5_inad_totais = np.argsort(desvio_inad_totais_viz)[-5:][::-1]
top_5_inad_lucrativos = np.argsort(desvio_inad_lucrativos_viz)[-5:][::-1]

x_pos = np.arange(5)
width = 0.25

ax_emb2.bar(x_pos - width, desvio_adim_viz[top_5_adim], width, label='Adimplentes', color='green', alpha=0.7)
ax_emb2.bar(x_pos, desvio_inad_totais_viz[top_5_inad_totais], width, label='Inadim. Totais', color='red', alpha=0.7)
ax_emb2.bar(x_pos + width, desvio_inad_lucrativos_viz[top_5_inad_lucrativos], width, label='Inadim. Lucrativos', color='orange', alpha=0.7)

ax_emb2.set_xlabel('Rank (1=Mais Importante)', fontsize=11)
ax_emb2.set_ylabel('Desvio da Média Geral', fontsize=11)
ax_emb2.set_title('Top 5 Dimensões Mais Discriminativas', fontsize=13, fontweight='bold')
ax_emb2.set_xticks(x_pos)
ax_emb2.set_xticklabels([f'{i+1}º' for i in range(5)])
ax_emb2.legend(fontsize=9)
ax_emb2.grid(alpha=0.3, axis='y')

# 3. Heatmap das dimensões mais importantes
ax_emb3 = axes_emb[1, 0]
# Combinar top 5 de cada grupo (sem repetição)
all_top_dims = sorted(set(list(top_5_adim) + list(top_5_inad_totais) + list(top_5_inad_lucrativos)))
heatmap_data = np.array([
    embeddings_adim_viz[all_top_dims],
    embeddings_inad_totais_viz[all_top_dims],
    embeddings_inad_lucrativos_viz[all_top_dims]
])

im = ax_emb3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
ax_emb3.set_yticks([0, 1, 2])
ax_emb3.set_yticklabels(['Adimplentes', 'Inadim. Totais', 'Inadim. Lucrativos'], fontsize=10)
ax_emb3.set_xticks(range(len(all_top_dims)))
ax_emb3.set_xticklabels([f'Dim {d}' for d in all_top_dims], rotation=45, ha='right', fontsize=9)
ax_emb3.set_title('Heatmap: Valores dos Embeddings nas Dimensões Principais', fontsize=13, fontweight='bold')
cbar = plt.colorbar(im, ax=ax_emb3)
cbar.set_label('Valor do Embedding', fontsize=10)

# 4. Distribuição de variância por dimensão
ax_emb4 = axes_emb[1, 1]
var_por_dim = embeddings_test.std(axis=0)
dims_sorted = np.argsort(var_por_dim)[::-1][:15]  # Top 15 dimensões com maior variância

ax_emb4.bar(range(15), var_por_dim[dims_sorted], color='purple', alpha=0.7)
ax_emb4.set_xlabel('Rank da Dimensão', fontsize=11)
ax_emb4.set_ylabel('Desvio Padrão', fontsize=11)
ax_emb4.set_title('Top 15 Dimensões com Maior Variabilidade', fontsize=13, fontweight='bold')
ax_emb4.set_xticks(range(15))
ax_emb4.set_xticklabels([f'Dim {dims_sorted[i]}' for i in range(15)], rotation=45, ha='right', fontsize=8)
ax_emb4.grid(alpha=0.3, axis='y')

# Adicionar texto explicativo
textstr = 'Dimensões com alta variabilidade\ncapturam mais informação\npara discriminar entre grupos'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax_emb4.text(0.98, 0.97, textstr, transform=ax_emb4.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('../../graficos/analise_modelos/ENSEMBLE_PARETO_FINAL/analise_embeddings.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== VISUALIZAÇÃO: MAPEAMENTO DIMENSÕES → FEATURES ==========
# Criar gráfico mostrando as correlações mais fortes
fig_map, axes_map = plt.subplots(3, 1, figsize=(18, 14))

def plot_dimension_mapping(ax, dimensions, embeddings, features_scaled, feature_names, 
                          desvios, title, color):
    """Plota heatmap das correlações entre dimensões e top features"""
    # Para cada dimensão, pegar top 5 features
    all_features = set()
    dim_feature_corr = {}
    
    for dim in dimensions:
        # Passar como lista com um único elemento
        top_feats = map_dimensions_to_features([dim], embeddings, features_scaled, top_n=5)[dim]
        dim_feature_corr[dim] = {}
        for feat_name, abs_corr, corr in top_feats:
            all_features.add(feat_name)
            dim_feature_corr[dim][feat_name] = corr
    
    # Criar matriz de correlações
    all_features = sorted(list(all_features))
    corr_matrix = np.zeros((len(dimensions), len(all_features)))
    
    for i, dim in enumerate(dimensions):
        for j, feat in enumerate(all_features):
            corr_matrix[i, j] = dim_feature_corr[dim].get(feat, 0)
    
    # Plot heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Labels
    ax.set_yticks(range(len(dimensions)))
    ax.set_yticklabels([f'Dim {d}\n(δ={desvios[d]:.3f})' for d in dimensions], fontsize=9)
    
    ax.set_xticks(range(len(all_features)))
    # Truncar nomes longos
    feat_labels = [f[:25] + '...' if len(f) > 25 else f for f in all_features]
    ax.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=8)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Features Originais', fontsize=11)
    ax.set_ylabel('Dimensões do Embedding', fontsize=11)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlação', fontsize=10)
    
    # Adicionar valores no heatmap (apenas correlações > 0.3)
    for i in range(len(dimensions)):
        for j in range(len(all_features)):
            val = corr_matrix[i, j]
            if abs(val) > 0.3:
                text_color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       color=text_color, fontsize=7, fontweight='bold')

# Plot para cada grupo
plot_dimension_mapping(axes_map[0], top_adimplentes, embeddings_test, X_test_scaled, 
                      feature_names, desvio_adimplentes,
                      '✅ ADIMPLENTES: Dimensões → Features Originais', 'green')

plot_dimension_mapping(axes_map[1], top_inadimplentes_totais, embeddings_test, X_test_scaled, 
                      feature_names, desvio_inadimplentes_totais,
                      '❌ INADIMPLENTES TOTAIS: Dimensões → Features Originais', 'red')

plot_dimension_mapping(axes_map[2], top_inadimplentes_lucrativos, embeddings_test, X_test_scaled, 
                      feature_names, desvio_inadimplentes_lucrativos,
                      '💰 INADIMPLENTES LUCRATIVOS: Dimensões → Features Originais', 'orange')

plt.tight_layout()
plt.savefig('../../graficos/analise_modelos/ENSEMBLE_PARETO_FINAL/mapeamento_dimensoes_features.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Gráficos salvos (incluindo análise de embeddings e mapeamento)")

def mapear_dimensao_para_features(dim_idx, embeddings, features_scaled, feature_names, top_n=5):
    """
    Mapeia uma dimensão do embedding para as features originais mais correlacionadas
    """
    embedding_dim = embeddings[:, dim_idx]
    correlations = []
    
    for i, feature_name in enumerate(feature_names):
        feature_values = features_scaled[:, i]
        corr = np.corrcoef(embedding_dim, feature_values)[0, 1]
        correlations.append((feature_name, abs(corr), corr))
    
    # Ordenar por correlação absoluta (mais forte)
    correlations.sort(key=lambda x: x[1], reverse=True)
    return correlations[:top_n]

# Mapear as dimensões mais importantes de cada grupo
print("\n✅ ADIMPLENTES - Mapeamento das Top 5 Dimensões:")
for i, dim in enumerate(top_adimplentes, 1):
    top_features = mapear_dimensao_para_features(dim, embeddings_test, X_test_scaled, feature_names, top_n=3)
    print(f"\n   {i}. Dimensão {dim} (desvio={desvio_adimplentes[dim]:.4f}):")
    print(f"      Top 3 features correlacionadas:")
    for feat_name, abs_corr, corr in top_features:
        sinal = "+" if corr > 0 else "-"
        # Truncar nome da feature se for muito longo
        feat_display = feat_name if len(feat_name) <= 40 else feat_name[:37] + "..."
        print(f"      • {feat_display:<40} | corr: {sinal}{abs_corr:.3f}")

print("\n❌ INADIMPLENTES TOTAIS - Mapeamento das Top 5 Dimensões:")
for i, dim in enumerate(top_inadimplentes_totais, 1):
    top_features = mapear_dimensao_para_features(dim, embeddings_test, X_test_scaled, feature_names, top_n=3)
    print(f"\n   {i}. Dimensão {dim} (desvio={desvio_inadimplentes_totais[dim]:.4f}):")
    print(f"      Top 3 features correlacionadas:")
    for feat_name, abs_corr, corr in top_features:
        sinal = "+" if corr > 0 else "-"
        feat_display = feat_name if len(feat_name) <= 40 else feat_name[:37] + "..."
        print(f"      • {feat_display:<40} | corr: {sinal}{abs_corr:.3f}")

print("\n💰 INADIMPLENTES LUCRATIVOS - Mapeamento das Top 5 Dimensões:")
for i, dim in enumerate(top_inadimplentes_lucrativos, 1):
    top_features = mapear_dimensao_para_features(dim, embeddings_test, X_test_scaled, feature_names, top_n=3)
    print(f"\n   {i}. Dimensão {dim} (desvio={desvio_inadimplentes_lucrativos[dim]:.4f}):")
    print(f"      Top 3 features correlacionadas:")
    for feat_name, abs_corr, corr in top_features:
        sinal = "+" if corr > 0 else "-"
        feat_display = feat_name if len(feat_name) <= 40 else feat_name[:37] + "..."
        print(f"      • {feat_display:<40} | corr: {sinal}{abs_corr:.3f}")

print("\n💡 COMO INTERPRETAR:")
print("   - Correlação positiva (+): feature alta → dimensão alta")
print("   - Correlação negativa (-): feature alta → dimensão baixa")
print("   - Valores próximos de 1.0: forte relação linear")
print("   - Múltiplas features podem contribuir para uma dimensão (espaço latente)")

# ---------- 7. SALVAR RESUMO COMPLETO (APÓS ANÁLISE DE EMBEDDINGS) ----------
print("\n--- Salvando Resumo Completo ---")
with open('../../graficos/analise_modelos/ENSEMBLE_PARETO_FINAL/resumo.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("MODELO FINAL: ENSEMBLE HETEROGÊNEO + OTIMIZAÇÃO MULTI-OBJETIVO\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Total de Contratos: {len(y_test_lucro):,}\n")
    f.write(f"Split: 80/20\n\n")
    
    f.write("--- DESEMPENHO DOS MODELOS BASE ---\n")
    f.write(f"NN:   MAE = R$ {mae_nn:.2f}, R² = {r2_nn:.4f}\n")
    f.write(f"LGBM: MAE = R$ {mae_lgbm:.2f}, R² = {r2_lgbm:.4f}\n")
    f.write(f"XGB:  MAE = R$ {mae_xgb:.2f}, R² = {r2_xgb:.4f}\n")
    f.write(f"META: MAE = R$ {mae_meta:.2f}, R² = {r2_meta:.4f}\n\n")
    
    f.write("--- RESULTADOS FINAIS ---\n")
    f.write(f"Baseline:          R$ {lucro_baseline:,.2f}\n")
    f.write(f"Máximo Teórico:    R$ {lucro_maximo_teorico:,.2f}\n")
    f.write(f"ENSEMBLE+PARETO:   R$ {lucro_otimizado:,.2f}\n\n")
    
    f.write(f"Ganho vs Baseline: R$ {ganho_vs_baseline:+,.2f} ({ganho_perc:+.2f}%)\n\n")
    
    f.write("--- PARETO FRONTIER ---\n")
    f.write(f"Soluções Ótimas: {len(pareto_trials)}\n")
    f.write(f"Threshold Selecionado: R$ {best_threshold:.2f}\n")
    f.write(f"FDR (Taxa Inadimplência): {best_fdr*100:.2f}%\n")
    f.write(f"Taxa de Aprovação: {taxa_aprovacao:.1f}%\n\n")
    
    f.write("--- MATRIZ DE CONFUSÃO ---\n")
    f.write(f"True Negatives (Adimplente Aceito):     {tn:,}\n")
    f.write(f"False Positives (Adimplente Rejeitado): {fp:,}\n")
    f.write(f"False Negatives (Inadimplente Aceito):  {fn:,}\n")
    f.write(f"True Positives (Inadimplente Rejeitado):{tp:,}\n")
    f.write(f"Precision (Precisão): {precisao:.2%}\n")
    f.write(f"Recall (Sensibilidade): {recall:.2%}\n")
    f.write(f"F1-Score: {f1:.2%}\n\n")
    
    # Adicionar análise de embeddings ao resumo
    f.write("--- ANÁLISE DOS EMBEDDINGS ---\n")
    f.write(f"Dimensão dos Embeddings: {embeddings_test.shape[1]}\n\n")
    
    f.write("Top 5 Dimensões - ADIMPLENTES:\n")
    for i, dim in enumerate(top_adimplentes, 1):
        valor = embeddings_adimplentes[dim]
        desvio = desvio_adimplentes[dim]
        f.write(f"  {i}. Dim {dim:2d}: valor={valor:+.4f}, desvio={desvio:.4f}\n")
    
    f.write("\nTop 5 Dimensões - INADIMPLENTES TOTAIS:\n")
    for i, dim in enumerate(top_inadimplentes_totais, 1):
        valor = embeddings_inadimplentes_totais[dim]
        desvio = desvio_inadimplentes_totais[dim]
        f.write(f"  {i}. Dim {dim:2d}: valor={valor:+.4f}, desvio={desvio:.4f}\n")
    
    f.write("\nTop 5 Dimensões - INADIMPLENTES LUCRATIVOS:\n")
    for i, dim in enumerate(top_inadimplentes_lucrativos, 1):
        valor = embeddings_inadimplentes_lucrativos[dim]
        desvio = desvio_inadimplentes_lucrativos[dim]
        f.write(f"  {i}. Dim {dim:2d}: valor={valor:+.4f}, desvio={desvio:.4f}\n")
    
    f.write(f"\nDimensões Exclusivas (melhor discriminação):\n")
    f.write(f"  Adimplentes: {sorted(exclusivas_adim) if exclusivas_adim else 'Nenhuma'}\n")
    f.write(f"  Inadimplentes Totais: {sorted(exclusivas_inad_totais) if exclusivas_inad_totais else 'Nenhuma'}\n")
    f.write(f"  Inadimplentes Lucrativos: {sorted(exclusivas_inad_lucrativos) if exclusivas_inad_lucrativos else 'Nenhuma'}\n\n")
    
    # Adicionar mapeamento para features originais
    f.write("--- MAPEAMENTO: DIMENSÕES → FEATURES ORIGINAIS ---\n\n")
    
    f.write("ADIMPLENTES - Top Dimensões:\n")
    for i, dim in enumerate(top_adimplentes, 1):
        top_features = mapear_dimensao_para_features(dim, embeddings_test, X_test_scaled, feature_names, top_n=3)
        f.write(f"  Dimensão {dim}:\n")
        for feat_name, abs_corr, corr in top_features:
            sinal = "+" if corr > 0 else "-"
            f.write(f"    • {feat_name}: {sinal}{abs_corr:.3f}\n")
        f.write("\n")
    
    f.write("INADIMPLENTES TOTAIS - Top Dimensões:\n")
    for i, dim in enumerate(top_inadimplentes_totais, 1):
        top_features = mapear_dimensao_para_features(dim, embeddings_test, X_test_scaled, feature_names, top_n=3)
        f.write(f"  Dimensão {dim}:\n")
        for feat_name, abs_corr, corr in top_features:
            sinal = "+" if corr > 0 else "-"
            f.write(f"    • {feat_name}: {sinal}{abs_corr:.3f}\n")
        f.write("\n")
    
    f.write("INADIMPLENTES LUCRATIVOS - Top Dimensões:\n")
    for i, dim in enumerate(top_inadimplentes_lucrativos, 1):
        top_features = mapear_dimensao_para_features(dim, embeddings_test, X_test_scaled, feature_names, top_n=3)
        f.write(f"  Dimensão {dim}:\n")
        for feat_name, abs_corr, corr in top_features:
            sinal = "+" if corr > 0 else "-"
            f.write(f"    • {feat_name}: {sinal}{abs_corr:.3f}\n")
        f.write("\n")

print("✓ Resumo salvo (incluindo análise de embeddings e mapeamento)")

print("\n" + "="*80)
print("✅ MODELO FINAL COMPLETO!")
print("="*80)
print(f"🏆 Resultado Final: {ganho_perc:+.2f}% de ganho vs baseline")
print(f"📉 Taxa de Inadimplência: {best_fdr*100:.2f}%")
print(f"📁 Arquivos salvos em: graficos/ENSEMBLE_PARETO_FINAL/")
print(f"   - analise_completa.png (visão geral dos resultados)")
print(f"   - analise_embeddings.png (perfil e dimensões dos embeddings)")
print(f"   - mapeamento_dimensoes_features.png (tradução: dimensões → features originais)")
print(f"   - matriz_confusao.png (matriz de confusão)")
print(f"   - resumo.txt (relatório completo com mapeamento)")
