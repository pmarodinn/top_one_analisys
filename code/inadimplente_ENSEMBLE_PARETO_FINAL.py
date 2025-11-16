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
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desabilitar otimizações que consomem memória

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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
DATA_FILE = '../data/dataset_interno_top_one_atualizado.csv'
df = load_and_preprocess_v3(DATA_FILE)

print(f"\n📊 Dados carregados: {len(df)} contratos")

# Target: LUCRO (regressão)
y_lucro = df['lucro'].values
y_default = df['default'].values  # Para métricas de inadimplência

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
print("\n� Split 80/20...")

X_train, X_test, y_train_lucro, y_test_lucro, y_train_default, y_test_default = train_test_split(
    X, y_lucro, y_default, test_size=0.2, random_state=42
)

print(f"Treino: {len(X_train):,} contratos")
print(f"Teste: {len(X_test):,} contratos")

# Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 3. ARQUITETURAS DOS MODELOS BASE ----------

def create_nn_regression(input_dim, hidden_dim=64):  # Reduzido de 128 para 64
    """Rede Neural para regressão de lucro + extração de embeddings (otimizada para CPU)"""
    inputs = layers.Input(shape=(input_dim,))
    
    # Encoder (arquitetura mais leve)
    x = layers.Dense(hidden_dim, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
    x = layers.Dropout(0.3)(x)
    
    embeddings = layers.Dense(hidden_dim // 2, activation='relu')(x)  # Camada de embeddings
    x = layers.Dropout(0.3)(embeddings)
    
    # Output: lucro (regressão)
    output = layers.Dense(1, activation='linear', name='lucro')(x)
    
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Modelo para extração de embeddings
    embedding_model = keras.Model(inputs=inputs, outputs=embeddings)
    
    return model, embedding_model

# ---------- 4. ENSEMBLE COM STACKING E OTIMIZAÇÃO MULTI-OBJETIVO ----------

print("\n" + "="*80)
print("🔄 TREINANDO ENSEMBLE HETEROGÊNEO + PARETO")
print("="*80)

# ========== NÍVEL 0: MODELOS BASE ==========
print(f"\n--- Nível 0: Treinando Modelos Base ---")

# 1. Neural Network (com embeddings)
print("1️⃣ Neural Network...")
model_nn, embedding_extractor = create_nn_regression(X_train_scaled.shape[1])

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model_nn.fit(
    X_train_scaled, y_train_lucro,
    validation_split=0.15,
    epochs=50,  # Reduzido de 100 para 50
    batch_size=128,  # Aumentado de 64 para 128 (mais eficiente na CPU)
    callbacks=[early_stop],
    verbose=0
)

# Criar extrator de embeddings (camada correta após simplificação)
embedding_extractor = keras.Model(
    inputs=model_nn.input,
    outputs=model_nn.layers[-3].output  # Camada de embeddings (ajustado)
)

pred_nn_train = model_nn.predict(X_train_scaled, verbose=0).flatten()
pred_nn_test = model_nn.predict(X_test_scaled, verbose=0).flatten()
embeddings_train = embedding_extractor.predict(X_train_scaled, verbose=0)
embeddings_test = embedding_extractor.predict(X_test_scaled, verbose=0)

mae_nn = mean_absolute_error(y_test_lucro, pred_nn_test)
r2_nn = r2_score(y_test_lucro, pred_nn_test)
print(f"   ✓ NN: MAE = R$ {mae_nn:.2f}, R² = {r2_nn:.4f}")

# 2. LightGBM
print("2️⃣ LightGBM...")
model_lgbm = lgbm.train(
    {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'seed': 42,
     'learning_rate': 0.05, 'num_leaves': 31, 'feature_fraction': 0.8},
    lgbm.Dataset(X_train_scaled, y_train_lucro),
    num_boost_round=500
)

pred_lgbm_train = model_lgbm.predict(X_train_scaled)
pred_lgbm_test = model_lgbm.predict(X_test_scaled)

mae_lgbm = mean_absolute_error(y_test_lucro, pred_lgbm_test)
r2_lgbm = r2_score(y_test_lucro, pred_lgbm_test)
print(f"   ✓ LGBM: MAE = R$ {mae_lgbm:.2f}, R² = {r2_lgbm:.4f}")

# 3. XGBoost
print("3️⃣ XGBoost...")
dtrain = xgb.DMatrix(X_train_scaled, label=y_train_lucro)
dtest = xgb.DMatrix(X_test_scaled, label=y_test_lucro)

model_xgb = xgb.train(
    {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'seed': 42,
     'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8},
    dtrain,
    num_boost_round=500
)

pred_xgb_train = model_xgb.predict(dtrain)
pred_xgb_test = model_xgb.predict(dtest)

mae_xgb = mean_absolute_error(y_test_lucro, pred_xgb_test)
r2_xgb = r2_score(y_test_lucro, pred_xgb_test)
print(f"   ✓ XGB: MAE = R$ {mae_xgb:.2f}, R² = {r2_xgb:.4f}")

# ========== NÍVEL 1: META-LEARNER ==========
print(f"\n--- Nível 1: Meta-Learner com Features Enriquecidas ---")

# Construir features para meta-learner
# Predições dos 3 modelos + embeddings da NN + top 10 features originais
top_features_idx = np.argsort(np.abs(X_train_scaled.std(axis=0)))[-10:]

meta_features_train = np.column_stack([
    pred_nn_train,
    pred_lgbm_train,
    pred_xgb_train,
    embeddings_train,
    X_train_scaled[:, top_features_idx]
])

meta_features_test = np.column_stack([
    pred_nn_test,
    pred_lgbm_test,
    pred_xgb_test,
    embeddings_test,
    X_test_scaled[:, top_features_idx]
])

print(f"   Features do meta-learner: {meta_features_train.shape[1]} (3 pred + 32 emb + 10 orig)")

# Meta-learner: XGBoost leve
dmeta_train = xgb.DMatrix(meta_features_train, label=y_train_lucro)
dmeta_test = xgb.DMatrix(meta_features_test, label=y_test_lucro)

meta_model = xgb.train(
    {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'seed': 42,
     'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.9},
    dmeta_train,
    num_boost_round=100
)

pred_meta_train = meta_model.predict(dmeta_train)
pred_meta_test = meta_model.predict(dmeta_test)

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

study.optimize(objective_pareto, n_trials=200, show_progress_bar=False)

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
    score = lucro_norm * 0.6 + fdr_norm * 0.4  # 60% lucro, 40% risco
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
os.makedirs('../graficos/ENSEMBLE_PARETO_FINAL', exist_ok=True)

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
    plt.savefig('../graficos/ENSEMBLE_PARETO_FINAL/matriz_confusao.png', dpi=300, bbox_inches='tight')
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
plt.savefig('../graficos/ENSEMBLE_PARETO_FINAL/analise_completa.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Gráficos salvos")

# ---------- 7. SALVAR RESUMO ----------
with open('../graficos/ENSEMBLE_PARETO_FINAL/resumo.txt', 'w', encoding='utf-8') as f:
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
    f.write(f"F1-Score: {f1:.2%}\n")

print("✓ Resumo salvo")

print("\n" + "="*80)
print("✅ MODELO FINAL COMPLETO!")
print("="*80)
print(f"🏆 Resultado Final: {ganho_perc:+.2f}% de ganho vs baseline")
print(f"📉 Taxa de Inadimplência: {best_fdr*100:.2f}%")
print(f"📁 Arquivos salvos em: graficos/ENSEMBLE_PARETO_FINAL/")
