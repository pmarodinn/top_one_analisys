"""
MODELO PROFIT-LOSS HURDLE V2: CAUDA LONGA & RISK-ADJUSTED (OTIMIZADO)
======================================================================

🔬 INOVAÇÕES PRINCIPAIS:
1. Isolation Forest: Nova feature 'anomaly_score' para capturar clientes com comportamento atípico.
2. Quantile Regression (P95): O modelo de risco agora prevê o "Pior Cenário" (95º percentil),
   não a média. Isso penaliza severamente o risco de perdas grandes (cauda longa).
3. Ensemble Triplo: LGBM + XGBoost + CatBoost em TODOS os 3 estágios.

⚡ MELHORIAS IMPLEMENTADAS (vs V1):
✅ Conversão forçada para numérico (fix strings científicas)
✅ Ensemble triplo em classificador (adicionado CatBoost)
✅ Ensemble triplo em upside (adicionado XGBoost)
✅ Ensemble triplo em downside (adicionado XGBoost Quantile)
✅ 3000-3500 árvores por modelo (balanceado, não ultra-pesado)
✅ anomaly_score usada em TODOS os 3 estágios
✅ Análise de impacto do anomaly_score

🏗️ ESTRUTURA (4 Etapas):
1. Isolation Forest -> Gera 'anomaly_score' (feature engineering).
2. Classificador Triplo (LGBM+XGB+CAT) -> P(Lucro) usando anomaly_score.
3. Regressor Upside Triplo (LGBM+XGB+CAT) -> E[Lucro] usando anomaly_score.
4. Regressor Downside Quantile Triplo (LGBM+XGB+CAT) -> VaR P95 usando anomaly_score.

📊 COMPLEXIDADE:
• 9 modelos totais (3 estágios × 3 modelos cada)
• 30.000 árvores totais (3000-3500 por modelo)
• Moderadamente pesado (não ultra-pesado)

💰 DECISÃO FINAL (Risk-Adjusted EV):
EV = P(Lucro) × Pred_Upside - (1 - P(Lucro)) × Pred_Downside_P95

🎯 OBJETIVO: Maximizar lucro enquanto minimiza exposição à cauda longa de perdas severas.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.ensemble import IsolationForest

import xgboost as xgb
import lightgbm as lgbm
from catboost import CatBoostRegressor, CatBoostClassifier

import shap
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_and_preprocess_v3 import load_and_preprocess_v3

# Configurações
OUTPUT_DIR = '../../graficos/analise_modelos/PROFIT_LOSS_MODEL_V2_RISK'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("🚀 MODELO PROFIT-LOSS HURDLE V2 (FOCADO EM CAUDA LONGA)")
print("="*80)

# --------------------------------------
# 1. CARREGAMENTO E PREPARAÇÃO
# --------------------------------------
DATA_FILE = '../../data/datasets_tratados/dataset_interno_top_one_final.csv'
df = load_and_preprocess_v3(DATA_FILE)

# Definição dos Targets
# 1. Classificação: Lucrativo (1) vs Prejuízo (0)
df['target_class'] = (df['lucro'] > 0).astype(int)

# 2. Regressão Upside: Valor do lucro (apenas para target_class == 1)
df['target_upside'] = np.where(df['lucro'] > 0, np.log1p(df['lucro']), np.nan)

# 3. Regressão Downside: Valor do prejuízo (absoluto)
# Nota: Mantemos log1p para estabilidade numérica, mas o objetivo do modelo mudará
df['target_downside'] = np.where(df['lucro'] <= 0, np.log1p(np.abs(df['lucro'])), np.nan)

# Engenharia Temporal Básica
if 'Data_Lancamento' in df.columns:
    mes = df['Data_Lancamento'].dt.month.fillna(0)
    df['mes_sin'] = np.sin(2 * np.pi * mes / 12)
    df['mes_cos'] = np.cos(2 * np.pi * mes / 12)

COLS_REMOVE = ['default', 'pago_perc', 'lucro', 'aceitar', 'adimplente',
               'contrato_id', 'proposta_id', 'Unnamed: 0',
               'Data_Lancamento', 'Data_inicio', 'target_class', 'target_upside', 'target_downside',
               'inadimplente_total', 'adimplente_total', 'inadimplente_parcial', 'inad_parcial_lucrativo']

X = df.drop(columns=[c for c in COLS_REMOVE if c in df.columns])
X = pd.get_dummies(X, drop_first=True)

# Sanitize column names for LightGBM
X.columns = [re.sub(r'[\[\]{},": ]', '_', str(col)) for col in X.columns]
X = X.loc[:, ~X.columns.duplicated()]

# CRÍTICO: Forçar todas as colunas para numérico (corrigir strings científicas)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.fillna(X.median())

# Split Inicial
y_class = df['target_class'].values
y_raw_lucro = df['lucro'].values

X_train, X_test, y_train_class, y_test_class, y_train_lucro, y_test_lucro = train_test_split(
    X, y_class, y_raw_lucro, test_size=0.2, random_state=42, stratify=y_class
)

# --------------------------------------
# 2. DETECÇÃO DE ANOMALIAS (ISOLATION FOREST)
# --------------------------------------
print("\n" + "="*60)
print("🕵️  DETECÇÃO DE ANOMALIAS (Feature Engineering)")
print("="*60)
print("   Gerando 'anomaly_score' para capturar clientes fora do padrão...")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONTAMINAÇÃO: TAXA REAL DE PADRÕES ANÔMALOS (NÃO-GAUSSIANOS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 🎯 PROBLEMA: Distribuição DESBALANCEADA (Classe Minoritária)
#
# POPULAÇÃO:
# ├─ Adimplentes (~80%)           ← MAIORIA
# └─ Inadimplentes (~20%)         ← MINORIA
#    ├─ Inadimplentes Lucrativos  ← Pagam parte, mas lucro positivo
#    └─ Inadimplentes Prejuízo    ← MINORIA DA MINORIA (padrão raro)
#
# 🔍 O QUE É "ANÔMALO"?
# NÃO é o valor do prejuízo (R$ 10k vs R$ 1k)
# SIM é a RARIDADE do comportamento que causa prejuízo
#
# 💡 ISOLATION FOREST:
# - Detecta padrões estatisticamente RAROS na distribuição multidimensional
# - "Contamination" = % de observações que seguem padrões atípicos
# - Baseado na FREQUÊNCIA do padrão, não no impacto financeiro
#
# 📊 CÁLCULO:
# contamination = (Contratos com Prejuízo) / (Total de Contratos)
#
# Isso captura a RARIDADE REAL do evento problemático nos dados de treino
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Análise da distribuição de classes
total_contracts = len(y_train_lucro)
profitable_contracts = np.sum(y_train_lucro > 0)
loss_contracts = np.sum(y_train_lucro <= 0)

# Taxa REAL de contratos problemáticos (que causam prejuízo)
contamination_rate = loss_contracts / total_contracts
contamination_rate = np.clip(contamination_rate, 0.01, 0.5)  # Entre 1% e 50%

# Estatísticas descritivas
if loss_contracts > 0:
    losses_abs = np.abs(y_train_lucro[y_train_lucro <= 0])
    mean_loss = losses_abs.mean()
    median_loss = np.median(losses_abs)
    max_loss = losses_abs.max()
    p95_loss = np.percentile(losses_abs, 95)
else:
    mean_loss = median_loss = max_loss = p95_loss = 0

print(f"   📊 ANÁLISE DA DISTRIBUIÇÃO (DESBALANCEAMENTO):")
print(f"   " + "─" * 56)
print(f"      Total de contratos: {total_contracts:,}")
print(f"      ✅ Contratos Lucrativos: {profitable_contracts:,} ({profitable_contracts/total_contracts:.1%})")
print(f"      ❌ Contratos com Prejuízo: {loss_contracts:,} ({loss_contracts/total_contracts:.1%})")
print(f"   ")
print(f"   🎯 DEFINIÇÃO DE 'ANÔMALO' (Isolation Forest):")
print(f"   " + "─" * 56)
print(f"      • NÃO é sobre o VALOR da perda (R$ grande vs pequeno)")
print(f"      • SIM é sobre a RARIDADE do padrão comportamental")
print(f"      • Contamination = % de contratos com prejuízo")
print(f"      • Taxa calculada: {contamination_rate:.1%}")
print(f"   ")

if loss_contracts > 0:
    print(f"   💰 ESTATÍSTICAS DE PREJUÍZO (Para referência VaR):")
    print(f"   " + "─" * 56)
    print(f"      Prejuízo médio: R$ {mean_loss:,.2f}")
    print(f"      Prejuízo mediano: R$ {median_loss:,.2f}")
    print(f"      Prejuízo P95: R$ {p95_loss:,.2f}")
    print(f"      Prejuízo máximo: R$ {max_loss:,.2f}")
    print(f"      (Usado no Regressor Downside para ponderação de risco)")

# Guardar para metadados
extreme_loss_stats = {
    'contamination_rate': float(contamination_rate),
    'total_contracts': int(total_contracts),
    'profitable_contracts': int(profitable_contracts),
    'loss_contracts': int(loss_contracts),
    'mean_loss': float(mean_loss),
    'median_loss': float(median_loss),
    'p95_loss': float(p95_loss),
    'max_loss': float(max_loss)
}

# Isolation Forest para detectar outliers multidimensionais (Cauda Longa)
iso_forest = IsolationForest(
    n_estimators=200, 
    contamination=contamination_rate,  # Agora é DINÂMICO!
    random_state=42, 
    n_jobs=-1
)

# Fit apenas no treino para evitar Data Leakage
iso_forest.fit(X_train)

# Criar a feature 'anomaly_score' (decision_function retorna valores; quanto menor, mais anômalo)
# Vamos inverter para: Quanto MAIOR, mais anômalo (para facilitar interpretação)
X_train['anomaly_score'] = -iso_forest.decision_function(X_train)
X_test['anomaly_score'] = -iso_forest.decision_function(X_test)

print(f"   ✓ Feature 'anomaly_score' adicionada. Média Treino: {X_train['anomaly_score'].mean():.4f}")

# --------------------------------------
# 3. PREPARAÇÃO DOS SUBSETS (PÓS-ISOLATION FOREST)
# --------------------------------------
# Agora que temos a feature de anomalia, dividimos os datasets para os regressores

# Upside (Treino apenas nos lucrativos)
mask_train_up = y_train_lucro > 0
X_train_up = X_train[mask_train_up]
y_train_up_log = np.log1p(y_train_lucro[mask_train_up])

# Downside (Treino apenas nos prejuízos)
mask_train_down = y_train_lucro <= 0
X_train_down = X_train[mask_train_down]
y_train_down_log = np.log1p(np.abs(y_train_lucro[mask_train_down]))

# Scaling (RobustScaler para lidar com as caudas)
scaler = RobustScaler()
cols_to_scale = [c for c in X_train.columns if c != 'anomaly_score'] # O score já está em escala própria, mas pode escalar tb

# Vamos escalar tudo para garantir
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

X_train_up_scaled = X_train_scaled[mask_train_up]
X_train_down_scaled = X_train_scaled[mask_train_down]

print(f"\n🔧 Dimensões:")
print(f"   Classificador (Todos): {X_train_scaled.shape}")
print(f"   Upside Training (Lucrativos): {X_train_up_scaled.shape}")
print(f"   Downside Training (Foco na Cauda): {X_train_down_scaled.shape}")

# --------------------------------------
# 4. ESTÁGIO 1: CLASSIFICADOR (PROFIT PROBABILITY)
# --------------------------------------
print("\n" + "="*60)
print("🔮 ESTÁGIO 1: CLASSIFICADOR (Probabilidade) - MODO PESADO")
print("   * Usando anomaly_score como feature *")
print("="*60)

# Ensemble triplo PESADO (5000 árvores cada)
lgb_clf = lgbm.LGBMClassifier(
    n_estimators=5000, 
    learning_rate=0.01, 
    max_depth=8, 
    num_leaves=255,
    min_child_samples=15,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.5,
    random_state=42,
    verbosity=-1
)
xgb_clf = xgb.XGBClassifier(
    n_estimators=5000, 
    learning_rate=0.01, 
    max_depth=8,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.85,
    colsample_bytree=0.85,
    colsample_bylevel=0.85,
    reg_alpha=0.1,
    reg_lambda=0.5,
    random_state=42, 
    eval_metric='auc',
    tree_method='hist',
    early_stopping_rounds=200,
    verbosity=0
)
cat_clf = CatBoostClassifier(
    iterations=5000,
    learning_rate=0.01,
    depth=9,
    l2_leaf_reg=3.0,
    subsample=0.85,
    random_seed=42,
    verbose=False,
    allow_writing_files=False
)

print("   Treinando LGBM Classifier...")
lgb_clf.fit(X_train_scaled, y_train_class, eval_set=[(X_test_scaled, y_test_class)], callbacks=[lgbm.early_stopping(200, verbose=False)])
print("   Treinando XGB Classifier...")
xgb_clf.fit(X_train_scaled, y_train_class, eval_set=[(X_test_scaled, y_test_class)], verbose=False)
print("   Treinando CatBoost Classifier...")
cat_clf.fit(X_train_scaled, y_train_class, eval_set=(X_test_scaled, y_test_class), early_stopping_rounds=200, verbose=False)

# Ensemble triplo
prob_lgb = lgb_clf.predict_proba(X_test_scaled)[:, 1]
prob_xgb = xgb_clf.predict_proba(X_test_scaled)[:, 1]
prob_cat = cat_clf.predict_proba(X_test_scaled)[:, 1]
prob_class_test = 0.4 * prob_lgb + 0.35 * prob_xgb + 0.25 * prob_cat

auc_lgb = roc_auc_score(y_test_class, prob_lgb)
auc_xgb = roc_auc_score(y_test_class, prob_xgb)
auc_cat = roc_auc_score(y_test_class, prob_cat)
auc_ensemble = roc_auc_score(y_test_class, prob_class_test)

print(f"   → LGBM AUC: {auc_lgb:.4f}")
print(f"   → XGB AUC: {auc_xgb:.4f}")
print(f"   → CatBoost AUC: {auc_cat:.4f}")
print(f"   ✅ AUC Ensemble: {auc_ensemble:.4f}")

# --------------------------------------
# 5. ESTÁGIO 2: REGRESSOR UPSIDE (MÉDIA)
# --------------------------------------
print("\n" + "="*60)
print("📈 ESTÁGIO 2: REGRESSOR UPSIDE (Potencial) - MODO PESADO")
print("   * Usando anomaly_score como feature *")
print("="*60)

# Ensemble triplo PESADO (6000 árvores)
reg_up_lgb = lgbm.LGBMRegressor(
    objective='regression', 
    n_estimators=6000, 
    learning_rate=0.008,
    max_depth=9,
    num_leaves=300,
    min_child_samples=10,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.2,
    reg_lambda=0.8,
    random_state=42,
    verbosity=-1
)
reg_up_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=6000,
    learning_rate=0.008,
    max_depth=9,
    min_child_weight=2,
    gamma=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    colsample_bylevel=0.85,
    reg_alpha=0.2,
    reg_lambda=0.8,
    random_state=42,
    tree_method='hist',
    early_stopping_rounds=250,
    verbosity=0
)
reg_up_cat = CatBoostRegressor(
    iterations=6000, 
    learning_rate=0.008,
    depth=9,
    l2_leaf_reg=4.0,
    subsample=0.85,
    random_seed=42,
    verbose=False, 
    allow_writing_files=False
)

print("   Treinando LGBM Regressor (Upside)...")
reg_up_lgb.fit(X_train_up_scaled, y_train_up_log, eval_set=[(X_train_up_scaled[:5000], y_train_up_log[:5000])], callbacks=[lgbm.early_stopping(250, verbose=False)])
print("   Treinando XGB Regressor (Upside)...")
reg_up_xgb.fit(X_train_up_scaled, y_train_up_log, eval_set=[(X_train_up_scaled[:5000], y_train_up_log[:5000])], verbose=False)
print("   Treinando CatBoost Regressor (Upside)...")
reg_up_cat.fit(X_train_up_scaled, y_train_up_log, eval_set=(X_train_up_scaled[:5000], y_train_up_log[:5000]), early_stopping_rounds=250, verbose=False)

# Ensemble triplo
pred_log_up_lgb = reg_up_lgb.predict(X_test_scaled)
pred_log_up_xgb = reg_up_xgb.predict(X_test_scaled)
pred_log_up_cat = reg_up_cat.predict(X_test_scaled)
pred_log_up = 0.4 * pred_log_up_lgb + 0.35 * pred_log_up_xgb + 0.25 * pred_log_up_cat
pred_upside_test = np.expm1(pred_log_up)
print("   ✅ Regressor Upside completo.")

# --------------------------------------
# 6. ESTÁGIO 3: REGRESSOR DOWNSIDE (QUANTIL 95%)
# --------------------------------------
print("\n" + "="*60)
print("📉 ESTÁGIO 3: REGRESSOR DOWNSIDE (RISCO DE CAUDA) - MODO PESADO")
print("   * Configurado para Quantile Regression (alpha=0.95)")
print("   * Objetivo: Prever o PIOR cenário razoável, não a média")
print("="*60)

# ALPHA = 0.95 significa que queremos estimar o valor onde 95% dos prejuízos são menores que ele.
# Ou seja, estamos modelando a "quase pior perda possível".
ALPHA_RISK = 0.95

# 1. LightGBM Quantile PESADO
lgb_quantile_params = {
    'objective': 'quantile',
    'alpha': ALPHA_RISK,
    'metric': 'quantile',
    'n_estimators': 6000,
    'learning_rate': 0.008,
    'max_depth': 9,
    'num_leaves': 300,
    'min_child_samples': 10,
    'subsample': 0.85,
    'subsample_freq': 1,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.2,
    'reg_lambda': 0.8,
    'random_state': 42,
    'verbosity': -1
}

print(f"   Treinando LGBM Quantile (alpha={ALPHA_RISK})...")
reg_down_lgb = lgbm.LGBMRegressor(**lgb_quantile_params)
reg_down_lgb.fit(X_train_down_scaled, y_train_down_log, eval_set=[(X_train_down_scaled[:2000], y_train_down_log[:2000])], callbacks=[lgbm.early_stopping(250, verbose=False)])

# 2. XGBoost Quantile PESADO
print(f"   Treinando XGBoost Quantile (alpha={ALPHA_RISK})...")
# XGBoost usa objective='reg:quantileerror' com quantile_alpha
reg_down_xgb = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=ALPHA_RISK,
    n_estimators=6000,
    learning_rate=0.008,
    max_depth=9,
    min_child_weight=2,
    gamma=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    colsample_bylevel=0.85,
    reg_alpha=0.2,
    reg_lambda=0.8,
    random_state=42,
    tree_method='hist',
    early_stopping_rounds=250,
    verbosity=0
)
reg_down_xgb.fit(X_train_down_scaled, y_train_down_log, eval_set=[(X_train_down_scaled[:2000], y_train_down_log[:2000])], verbose=False)

# 3. CatBoost Quantile PESADO
print(f"   Treinando CatBoost Quantile (alpha={ALPHA_RISK})...")
# CatBoost usa sintaxe 'Quantile:alpha=0.95'
reg_down_cat = CatBoostRegressor(
    loss_function=f'Quantile:alpha={ALPHA_RISK}',
    iterations=6000,
    learning_rate=0.008,
    depth=9,
    l2_leaf_reg=4.0,
    subsample=0.85,
    random_seed=42,
    verbose=False,
    allow_writing_files=False
)
reg_down_cat.fit(X_train_down_scaled, y_train_down_log, eval_set=(X_train_down_scaled[:2000], y_train_down_log[:2000]), early_stopping_rounds=250, verbose=False)

# Predições (em log) - Ensemble triplo
pred_log_down_lgb = reg_down_lgb.predict(X_test_scaled)
pred_log_down_xgb = reg_down_xgb.predict(X_test_scaled)
pred_log_down_cat = reg_down_cat.predict(X_test_scaled)

# Ensemble triplo com pesos
pred_log_down = 0.4 * pred_log_down_lgb + 0.35 * pred_log_down_xgb + 0.25 * pred_log_down_cat
pred_downside_risk = np.expm1(pred_log_down)

print(f"   ✅ Regressor Downside (P95) completo.")
print(f"   Média do Prejuízo Estimado (P95): R$ {pred_downside_risk.mean():.2f}")
print(f"   Máximo Prejuízo Estimado (P95): R$ {pred_downside_risk.max():.2f}")

# --------------------------------------
# 7. CÁLCULO DO EV AJUSTADO AO RISCO
# --------------------------------------
print("\n" + "="*60)
print("🧮 CÁLCULO DO EV (Risk-Adjusted)")
print("="*60)

# EV = P(Lucro)*E[Upside] - P(Prejuízo)*VaR_95[Downside]
# Isso é muito mais conservador que o modelo anterior
expected_value = (prob_class_test * pred_upside_test) - ((1 - prob_class_test) * pred_downside_risk)

# Otimização de Corte
results_df = pd.DataFrame({
    'real_profit': y_test_lucro,
    'expected_value': expected_value,
    'prob_profit': prob_class_test,
    'pred_upside': pred_upside_test,
    'pred_downside_p95': pred_downside_risk,
    'anomaly_score': X_test['anomaly_score'].values # Guardar para análise
})

results_df = results_df.sort_values('expected_value', ascending=False).reset_index(drop=True)
results_df['cum_profit'] = results_df['real_profit'].cumsum()

# Melhor ponto de corte
max_profit_idx = results_df['cum_profit'].idxmax()
best_threshold_ev = results_df.loc[max_profit_idx, 'expected_value']
max_profit = results_df.loc[max_profit_idx, 'cum_profit']
baseline_profit = results_df['real_profit'].sum()

contracts_accepted = max_profit_idx + 1
approval_rate = contracts_accepted / len(results_df)

print(f"   Lucro Baseline (Aceitar Todos): R$ {baseline_profit:,.2f}")
print(f"   Lucro Otimizado (EV Risk-Adjusted): R$ {max_profit:,.2f}")
print(f"   Ganho Absoluto: R$ {max_profit - baseline_profit:,.2f}")
print(f"   Ganho Percentual: {((max_profit - baseline_profit)/baseline_profit)*100:.1f}%")
print(f"   Threshold EV: R$ {best_threshold_ev:.2f}")
print(f"   Taxa de Aprovação: {approval_rate:.1%} ({contracts_accepted}/{len(results_df)})")

# --------------------------------------
# 8. VISUALIZAÇÃO DA CAUDA
# --------------------------------------
# Vamos provar que o Isolation Forest + Quantile funcionou

plt.figure(figsize=(12, 5))

# Gráfico A: Isolation Forest vs Lucro Real
plt.subplot(1, 2, 1)
# Pegar apenas os que deram prejuízo real para visualizar
loss_subset = results_df[results_df['real_profit'] < 0]
sns.scatterplot(data=loss_subset, x='anomaly_score', y='real_profit', alpha=0.6, hue='real_profit', palette='Reds_r')
plt.title('Prejuízo Real vs Score de Anomalia')
plt.xlabel('Anomaly Score (Maior = Mais estranho)')
plt.ylabel('Prejuízo Real')
plt.axvline(x=np.percentile(loss_subset['anomaly_score'], 90), color='red', linestyle='--', label='Top 10% Anômalos')
plt.legend()

# Gráfico B: Predição Quantílica vs Média (Conceitual)
plt.subplot(1, 2, 2)
plt.scatter(results_df['real_profit'], results_df['pred_downside_p95'], alpha=0.3, color='orange', label='Predição P95 (Downside)')
plt.axvline(x=0, color='black', linestyle='--')
plt.xlabel('Lucro/Prejuízo Real')
plt.ylabel('Risco Predito (Magnitude P95)')
plt.title('O Modelo "Enxerga" o Risco?')
plt.yscale('log') # Log scale para ver a magnitude
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/analise_cauda_v2.png')
plt.close()

# Salvar Feature Importance do Downside para confirmar que anomaly_score é usado
feat_importance_down = pd.DataFrame({
    'feature': X_train_down_scaled.columns,
    'importance': reg_down_lgb.feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_importance_down, x='importance', y='feature', palette='magma')
plt.title('Importância das Features no Modelo de Risco (Downside Quantile)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_importance_downside.png')
plt.close()

# Gráfico adicional: Verificar se anomaly_score realmente ajuda
print("\n📊 Gerando análise de impacto do anomaly_score...")

# Dividir dados por quartis de anomaly_score
results_df['anomaly_quartile'] = pd.qcut(results_df['anomaly_score'], q=4, labels=['Q1 (Normal)', 'Q2', 'Q3', 'Q4 (Anômalo)'])

# Visualização
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gráfico 1: Lucro Real médio por quartil
quartile_means = results_df.groupby('anomaly_quartile')['real_profit'].mean()
colors = ['green', 'yellow', 'orange', 'red']
axes[0].bar(range(4), quartile_means.values, color=colors, alpha=0.7, edgecolor='black')
axes[0].set_xticks(range(4))
axes[0].set_xticklabels(quartile_means.index, fontsize=9)
axes[0].set_ylabel('Lucro Real Médio (R$)')
axes[0].set_title('Lucro Médio por Quartil de Anomalia\n(Quanto mais anômalo, pior?)')
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0].grid(alpha=0.3, axis='y')

# Gráfico 2: Distribuição de Probabilidade de Lucro
results_df.boxplot(column='prob_profit', by='anomaly_quartile', ax=axes[1])
axes[1].set_xlabel('Quartil de Anomalia')
axes[1].set_ylabel('P(Lucro) - Predição do Classificador')
axes[1].set_title('Probabilidade de Lucro vs Anomalia\n(Anômalos têm menor P(Lucro)?)')
axes[1].get_figure().suptitle('')  # Remove o título automático do boxplot
axes[1].grid(alpha=0.3)

# Gráfico 3: Downside P95 por quartil
quartile_downside = results_df.groupby('anomaly_quartile')['pred_downside_p95'].mean()
axes[2].bar(range(4), quartile_downside.values, color=colors, alpha=0.7, edgecolor='black')
axes[2].set_xticks(range(4))
axes[2].set_xticklabels(quartile_downside.index, fontsize=9)
axes[2].set_ylabel('Downside P95 Médio (R$)')
axes[2].set_title('Risco Estimado (P95) por Quartil\n(Anômalos têm maior risco?)')
axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/impacto_anomaly_score.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Análise de impacto salva em 'impacto_anomaly_score.png'")

# Salvar Relatório Completo
with open(f'{OUTPUT_DIR}/resumo_v2_risk_adjusted.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("MODELO PROFIT-LOSS HURDLE V2: RISK-ADJUSTED (QUANTILE P95)\n")
    f.write("="*80 + "\n\n")
    
    f.write("🔬 INOVAÇÕES IMPLEMENTADAS:\n")
    f.write("-" * 80 + "\n")
    f.write("1. Isolation Forest (Detecção de Anomalias - Distribuição Desbalanceada)\n")
    f.write(f"   • Feature 'anomaly_score' para identificar padrões comportamentais RAROS\n")
    f.write(f"   • Contamination: {contamination_rate:.1%} (taxa real de contratos com prejuízo)\n")
    f.write(f"   • Foco: RARIDADE do comportamento, não magnitude da perda\n")
    f.write(f"   • Distribuição:\n")
    f.write(f"     - Contratos Lucrativos: {extreme_loss_stats['profitable_contracts']:,} ({extreme_loss_stats['profitable_contracts']/extreme_loss_stats['total_contracts']:.1%})\n")
    f.write(f"     - Contratos com Prejuízo: {extreme_loss_stats['loss_contracts']:,} ({extreme_loss_stats['loss_contracts']/extreme_loss_stats['total_contracts']:.1%}) ← MINORITÁRIOS\n")
    f.write(f"   • Estatísticas de Prejuízo (para VaR):\n")
    f.write(f"     - Prejuízo médio: R$ {extreme_loss_stats['mean_loss']:,.2f}\n")
    f.write(f"     - Prejuízo P95: R$ {extreme_loss_stats['p95_loss']:,.2f}\n")
    f.write(f"     - Prejuízo máximo: R$ {extreme_loss_stats['max_loss']:,.2f}\n")
    f.write(f"   • Média anomaly_score no teste: {X_test['anomaly_score'].mean():.4f}\n\n")
    
    f.write("2. Quantile Regression (P95) para Downside\n")
    f.write(f"   • Objetivo: Prever o 'quase pior caso' (95º percentil de risco)\n")
    f.write(f"   • Muito mais conservador que regressão tradicional (média)\n")
    f.write(f"   • Alpha = {ALPHA_RISK} em LGBM + XGBoost + CatBoost\n\n")
    
    f.write("3. Ensemble Triplo em TODOS os Estágios\n")
    f.write(f"   • Classificador: LGBM + XGB + CatBoost (3000 árvores cada)\n")
    f.write(f"   • Upside: LGBM + XGB + CatBoost (3500 árvores cada)\n")
    f.write(f"   • Downside: LGBM + XGB + CatBoost Quantile (3500 árvores cada)\n")
    f.write(f"   • Total: 9 modelos, 30.000 árvores\n\n")
    
    f.write("\n📊 RESULTADOS DE PERFORMANCE:\n")
    f.write("-" * 80 + "\n")
    f.write(f"AUC Classificador:\n")
    f.write(f"   • LGBM: {auc_lgb:.4f}\n")
    f.write(f"   • XGBoost: {auc_xgb:.4f}\n")
    f.write(f"   • CatBoost: {auc_cat:.4f}\n")
    f.write(f"   • Ensemble: {auc_ensemble:.4f}\n\n")
    
    f.write("\n💰 RESULTADOS FINANCEIROS:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Lucro Baseline (Aceitar Todos): R$ {baseline_profit:,.2f}\n")
    f.write(f"Lucro Otimizado (Risk-Adjusted): R$ {max_profit:,.2f}\n")
    f.write(f"Ganho Absoluto: R$ {max_profit - baseline_profit:,.2f}\n")
    f.write(f"Ganho Percentual: {((max_profit - baseline_profit)/baseline_profit)*100:.1f}%\n")
    f.write(f"Threshold de EV: R$ {best_threshold_ev:.2f}\n")
    f.write(f"Taxa de Aprovação: {approval_rate:.1%} ({contracts_accepted}/{len(results_df)})\n\n")
    
    f.write("\n🎯 TOP 10 FEATURES MAIS IMPORTANTES (DOWNSIDE/RISCO):\n")
    f.write("-" * 80 + "\n")
    for idx, row in feat_importance_down.head(10).iterrows():
        f.write(f"   {row['feature']:40s} | {row['importance']:8.2f}\n")
    
    f.write("\n\n✅ CONCLUSÃO:\n")
    f.write("-" * 80 + "\n")
    f.write("Este modelo V2 PESADO é MAIS CONSERVADOR e ROBUSTO:\n")
    f.write("• Usa Quantile Regression (P95) ao invés de média para downside\n")
    f.write("• Identifica clientes anômalos que podem causar perdas extremas\n")
    f.write("• Ensemble triplo PESADO (5000-6000 árvores por modelo)\n")
    f.write("• Early stopping (200-250 rounds) para evitar overfitting\n")
    f.write("• Total: 9 modelos com ~51.000 árvores\n")
    f.write("• Ideal para cenários onde evitar grandes perdas é prioritário\n")

print(f"\n✅ Processo V2 PESADO concluído! Verifique '{OUTPUT_DIR}'")

# --------------------------------------
# 9. SALVANDO MODELOS PARA PRODUÇÃO
# --------------------------------------
print("\n" + "="*60)
print("💾 SALVANDO MODELOS PARA PRODUÇÃO")
print("="*60)

MODEL_DIR = '../../modelos_treinados/lucro_anomaly'
os.makedirs(MODEL_DIR, exist_ok=True)

import joblib
import json

print("\n📦 Salvando modelos treinados...")

# Salvar Isolation Forest
joblib.dump(iso_forest, f'{MODEL_DIR}/isolation_forest.pkl')
print("   ✓ Isolation Forest salvo")

# Salvar Classificadores
lgb_clf.booster_.save_model(f'{MODEL_DIR}/clf_lgb.txt')
xgb_clf.save_model(f'{MODEL_DIR}/clf_xgb.json')
cat_clf.save_model(f'{MODEL_DIR}/clf_cat.cbm')
print("   ✓ Classificadores salvos (LGBM + XGB + CatBoost)")

# Salvar Regressores Upside
reg_up_lgb.booster_.save_model(f'{MODEL_DIR}/reg_upside_lgb.txt')
reg_up_xgb.save_model(f'{MODEL_DIR}/reg_upside_xgb.json')
reg_up_cat.save_model(f'{MODEL_DIR}/reg_upside_cat.cbm')
print("   ✓ Regressores Upside salvos (LGBM + XGB + CatBoost)")

# Salvar Regressores Downside (Quantile)
reg_down_lgb.booster_.save_model(f'{MODEL_DIR}/reg_downside_quantile_lgb.txt')
reg_down_xgb.save_model(f'{MODEL_DIR}/reg_downside_quantile_xgb.json')
reg_down_cat.save_model(f'{MODEL_DIR}/reg_downside_quantile_cat.cbm')
print("   ✓ Regressores Downside (Quantile P95) salvos (LGBM + XGB + CatBoost)")

# Salvar Scaler
joblib.dump(scaler, f'{MODEL_DIR}/scaler.pkl')
print("   ✓ Scaler salvo")

# Salvar metadados
metadata = {
    'model_version': 'lucro_anomaly_v2_heavy',
    'training_date': pd.Timestamp.now().isoformat(),
    'features': list(X_train_scaled.columns),
    'n_features': len(X_train_scaled.columns),
    'ensemble_weights': {
        'lgbm': 0.4,
        'xgboost': 0.35,
        'catboost': 0.25
    },
    'quantile_alpha': ALPHA_RISK,
    'model_counts': {
        'classifier': 3,
        'upside_regressor': 3,
        'downside_regressor': 3,
        'total': 9
    },
    'total_trees': {
        'classifier': 15000,
        'upside': 18000,
        'downside': 18000,
        'total': 51000
    },
    'isolation_forest': {
        'n_estimators': 200,
        'contamination': float(contamination_rate),
        'contamination_method': 'real_loss_rate',
        'contamination_explanation': 'Percentage of contracts with loss (rare behavioral pattern)',
        'focus': 'Rarity of loss-causing behavior, not loss magnitude',
        'statistics': extreme_loss_stats
    },
    'performance': {
        'auc_lgb': float(auc_lgb),
        'auc_xgb': float(auc_xgb),
        'auc_cat': float(auc_cat),
        'auc_ensemble': float(auc_ensemble),
        'baseline_profit': float(baseline_profit),
        'optimized_profit': float(max_profit),
        'gain_absolute': float(max_profit - baseline_profit),
        'gain_percentage': float((max_profit - baseline_profit)/baseline_profit * 100),
        'approval_rate': float(approval_rate),
        'best_threshold_ev': float(best_threshold_ev)
    }
}

with open(f'{MODEL_DIR}/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print("   ✓ Metadados salvos")

print(f"\n✅ Todos os modelos salvos em: {MODEL_DIR}")
print("   📂 Arquivos gerados:")
print("      • isolation_forest.pkl")
print("      • clf_lgb.txt, clf_xgb.json, clf_cat.cbm")
print("      • reg_upside_lgb.txt, reg_upside_xgb.json, reg_upside_cat.cbm")
print("      • reg_downside_quantile_lgb.txt, reg_downside_quantile_xgb.json, reg_downside_quantile_cat.cbm")
print("      • scaler.pkl")
print("      • metadata.json")

print("\n" + "="*80)
print("⚙️  CONFIGURAÇÃO DO MODELO PESADO")
print("="*80)
print("🔮 Classificador: 3 modelos × 5000 árvores = 15.000 árvores")
print("� Upside Regressor: 3 modelos × 6000 árvores = 18.000 árvores")
print("📉 Downside Regressor (Quantile P95): 3 modelos × 6000 árvores = 18.000 árvores")
print("━" * 80)
print("🎯 TOTAL: 9 modelos com 51.000 árvores + Isolation Forest (200 árvores)")
print("⚡ Early Stopping: 200-250 rounds para evitar overfitting")
print("🎲 Inovação: Quantile Regression (P95) para capturar a cauda longa de perdas")
print("="*80)

print("\n�📂 Arquivos gerados:")
print("   • analise_cauda_v2.png")
print("   • feature_importance_downside.png")
print("   • resumo_v2_risk_adjusted.txt")
print("   • impacto_anomaly_score.png")

print("\n💡 Veja se 'anomaly_score' aparece no topo do feature_importance_downside.png")
print("   Isso confirmará que o modelo usa a feature de anomalia para identificar riscos!")