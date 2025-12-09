"""
MODELO PROFIT-LOSS HURDLE: ABORDAGEM FOCADA EM VALOR
====================================================

Mudança de Paradigma:
- Não classificamos mais "Adimplente vs Inadimplente".
- Classificamos "Gera Lucro" vs "Gera Prejuízo".

Estrutura em 3 Estágios para lidar com Cauda Longa:
1. Classificador (Gatekeeper): Probabilidade de o contrato ser lucrativo (Lucro > 0).
2. Regressor Upside: Estima o lucro esperado (dado que é lucrativo).
3. Regressor Downside (Risco): Estima o prejuízo esperado (dado que dá prejuízo).
   * Crítico para capturar a "cauda longa" de perdas severas.

Decisão Final baseada em Valor Esperado (EV):
EV = P(Lucro) * Pred_Upside - (1 - P(Lucro)) * Pred_Downside
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    precision_recall_curve
)

import xgboost as xgb
import lightgbm as lgbm
from catboost import CatBoostClassifier, CatBoostRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

import optuna
from optuna.samplers import NSGAIISampler

import shap

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelos'))
from load_and_preprocess_v3 import load_and_preprocess_v3

# Configurações
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
OUTPUT_DIR = '../../graficos/analise_modelos/PROFIT_LOSS_MODEL'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("🚀 MODELO PROFIT-LOSS HURDLE (FOCADO EM VALOR)")
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
# Usamos log1p para suavizar a distribuição
df['target_upside'] = np.where(df['lucro'] > 0, np.log1p(df['lucro']), np.nan)

# 3. Regressão Downside: Valor do prejuízo (absoluto) (apenas para target_class == 0)
# Usamos log1p do valor absoluto para lidar com a cauda longa de perdas
df['target_downside'] = np.where(df['lucro'] <= 0, np.log1p(np.abs(df['lucro'])), np.nan)

print(f"\n📊 Distribuição dos Dados:")
print(f"   Total Contratos: {len(df)}")
print(f"   Lucrativos (Classe 1): {df['target_class'].sum()} ({df['target_class'].mean():.1%})")
print(f"   Prejuízo (Classe 0): {(df['target_class'] == 0).sum()} ({(1-df['target_class'].mean()):.1%})")

# --------------------------------------
# ENGENHARIA TEMPORAL AVANÇADA
# --------------------------------------
print("\n" + "="*80)
print("📅 ENGENHARIA DE FEATURES TEMPORAIS")
print("="*80)

if 'Data_Lancamento' in df.columns:
    print("Criando features temporais complexas...")
    
    # Extrair componentes básicos
    df['ano'] = df['Data_Lancamento'].dt.year
    df['mes'] = df['Data_Lancamento'].dt.month
    df['dia'] = df['Data_Lancamento'].dt.day
    df['trimestre'] = df['Data_Lancamento'].dt.quarter
    df['dia_semana'] = df['Data_Lancamento'].dt.dayofweek
    df['dia_ano'] = df['Data_Lancamento'].dt.dayofyear
    df['semana_ano'] = df['Data_Lancamento'].dt.isocalendar().week
    
    # Sazonalidade Cíclica (sin/cos para capturar periodicidade)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['trimestre_sin'] = np.sin(2 * np.pi * df['trimestre'] / 4)
    df['trimestre_cos'] = np.cos(2 * np.pi * df['trimestre'] / 4)
    df['dia_ano_sin'] = np.sin(2 * np.pi * df['dia_ano'] / 365)
    df['dia_ano_cos'] = np.cos(2 * np.pi * df['dia_ano'] / 365)
    
    # Períodos Críticos Específicos (baseado na sua observação)
    # Fim de ano: Novembro, Dezembro, Janeiro
    df['periodo_fim_ano'] = df['mes'].isin([11, 12, 1]).astype(int)
    
    # Carnaval (aproximação: Fevereiro-Março)
    df['periodo_carnaval'] = df['mes'].isin([2, 3]).astype(int)
    
    # Período crítico completo (fim de ano + carnaval)
    df['periodo_critico'] = (df['periodo_fim_ano'] | df['periodo_carnaval']).astype(int)
    
    # Meio do ano (Junho-Julho - pode ter padrões diferentes)
    df['periodo_meio_ano'] = df['mes'].isin([6, 7]).astype(int)
    
    # Início de ano fiscal/escolar (Janeiro-Março)
    df['inicio_ano'] = df['mes'].isin([1, 2, 3]).astype(int)
    
    # Interação Ano-Mês (para capturar mudanças de comportamento entre anos)
    # Exemplo: Dezembro-2023 vs Dezembro-2024 podem ter padrões diferentes
    df['ano_mes'] = df['ano'].astype(str) + '_' + df['mes'].astype(str).str.zfill(2)
    
    # Tendência Temporal (índice sequencial normalizado)
    df = df.sort_values('Data_Lancamento').reset_index(drop=True)
    df['tendencia_temporal'] = np.arange(len(df)) / len(df)
    
    # Análise de padrões históricos por período
    print("\n📊 Análise de Inadimplência por Período:")
    
    # Por mês
    print("\n🗓️  Taxa de Prejuízo por Mês:")
    taxa_prejuizo_mes = df.groupby('mes').agg({
        'target_class': ['mean', 'count'],
        'lucro': 'mean'
    }).round(4)
    taxa_prejuizo_mes.columns = ['Taxa_Lucrativo', 'N_Casos', 'Lucro_Medio']
    taxa_prejuizo_mes['Taxa_Prejuizo'] = 1 - taxa_prejuizo_mes['Taxa_Lucrativo']
    
    meses_nome = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                  7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    
    for mes_num in sorted(taxa_prejuizo_mes.index):
        row = taxa_prejuizo_mes.loc[mes_num]
        print(f"   {meses_nome[mes_num]:3s}: Prejuízo={row['Taxa_Prejuizo']:.1%} | "
              f"Lucro Médio=R${row['Lucro_Medio']:>8,.0f} | N={row['N_Casos']:.0f}")
    
    # Por ano-mês (para ver evolução temporal)
    print("\n📈 Taxa de Prejuízo por Ano-Mês (Períodos Críticos):")
    anos_unicos = sorted(df['ano'].unique())
    meses_criticos = [11, 12, 1, 2, 3]  # Nov a Mar
    
    for ano in anos_unicos:
        print(f"\n   Ano {ano}:")
        for mes_num in meses_criticos:
            mask = (df['ano'] == ano) & (df['mes'] == mes_num)
            if mask.sum() > 0:
                taxa_prej = 1 - df.loc[mask, 'target_class'].mean()
                lucro_medio = df.loc[mask, 'lucro'].mean()
                n_casos = mask.sum()
                print(f"      {meses_nome[mes_num]}: Prejuízo={taxa_prej:.1%} | "
                      f"Lucro Médio=R${lucro_medio:>8,.0f} | N={n_casos}")
    
    # Comparação de períodos críticos vs normais
    print("\n⚠️  Comparação Períodos Críticos vs Normais:")
    for periodo_nome, periodo_col in [
        ('Fim de Ano (Nov-Jan)', 'periodo_fim_ano'),
        ('Carnaval (Fev-Mar)', 'periodo_carnaval'),
        ('Período Crítico Completo', 'periodo_critico')
    ]:
        mask_periodo = df[periodo_col] == 1
        mask_normal = df[periodo_col] == 0
        
        taxa_prej_periodo = 1 - df.loc[mask_periodo, 'target_class'].mean()
        taxa_prej_normal = 1 - df.loc[mask_normal, 'target_class'].mean()
        lucro_periodo = df.loc[mask_periodo, 'lucro'].mean()
        lucro_normal = df.loc[mask_normal, 'lucro'].mean()
        
        print(f"\n   {periodo_nome}:")
        print(f"      Prejuízo: {taxa_prej_periodo:.1%} (Período) vs {taxa_prej_normal:.1%} (Normal)")
        print(f"      Lucro Médio: R${lucro_periodo:,.0f} vs R${lucro_normal:,.0f}")
        print(f"      Diferença: {((taxa_prej_periodo - taxa_prej_normal) / taxa_prej_normal * 100):+.1f}% em prejuízo")
    
    # Features de agregação temporal (rolling statistics)
    # Ordenar por data para calcular rolling
    df_sorted = df.sort_values('Data_Lancamento').copy()
    
    # Calcular taxa de prejuízo dos últimos N contratos (aproximação de contexto temporal)
    # Isso ajuda a capturar "momentum" de inadimplência
    window_sizes = [30, 90, 180]  # Últimos 30, 90, 180 contratos
    
    for window in window_sizes:
        df_sorted[f'prejuizo_rolling_{window}'] = (
            df_sorted['target_class']
            .rolling(window=window, min_periods=1)
            .apply(lambda x: 1 - x.mean())
        )
        df_sorted[f'lucro_rolling_{window}'] = (
            df_sorted['lucro']
            .rolling(window=window, min_periods=1)
            .mean()
        )
    
    # Retornar ao índice original
    df = df_sorted.sort_index()
    
    print("\n✅ Features temporais criadas com sucesso!")
    print(f"   • Componentes básicos: ano, mês, trimestre, dia_semana, etc.")
    print(f"   • Sazonalidade cíclica: sin/cos para mês, trimestre, dia_ano")
    print(f"   • Períodos específicos: fim_ano, carnaval, período_crítico")
    print(f"   • Interações: ano_mes (para capturar mudanças entre anos)")
    print(f"   • Tendência temporal e rolling statistics")
else:
    print("⚠️  Coluna 'Data_Lancamento' não encontrada. Pulando engenharia temporal.")

COLS_REMOVE = ['default', 'pago_perc', 'lucro', 'aceitar', 'adimplente',
               'contrato_id', 'proposta_id', 'Unnamed: 0',
               'Data_Lancamento', 'Data_inicio', 'target_class', 'target_upside', 'target_downside',
               'inadimplente_total', 'adimplente_total', 'inadimplente_parcial', 'inad_parcial_lucrativo',
               'lucro_faixa', 'ano_mes']  # ano_mes será tratado com get_dummies

X = df.drop(columns=[c for c in COLS_REMOVE if c in df.columns])
X = pd.get_dummies(X, drop_first=True)

# Sanitize column names for LightGBM (remove special JSON characters and spaces)
X.columns = [re.sub(r'[\[\]{},": ]', '_', str(col)) for col in X.columns]

# Handle duplicate column names after sanitization
X = X.loc[:, ~X.columns.duplicated()]

# Forçar todas as colunas para numérico (corrigir strings científicas e outros problemas)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.fillna(X.median())

# --------------------------------------
# ANOMALY DETECTION - CLIENTES NÃO-LUCRATIVOS
# --------------------------------------
print("\n" + "="*80)
print("🔍 DETECTANDO ANOMALIAS - Clientes com Características Não-Lucrativas")
print("="*80)
print("Objetivo: Identificar padrões atípicos que levam a perdas severas (cauda longa)")

# ESTRATÉGIA MELHORADA: Criar dataset balanceado focado em perdas extremas
# Isolation Forest funciona melhor com oversampling de casos raros

# 1. Identificar perdas severas (bottom 10% de lucro = piores casos)
threshold_perda_severa = df['lucro'].quantile(0.10)
mask_perda_severa = df['lucro'] <= threshold_perda_severa
mask_lucrativo = df['lucro'] > 0
mask_prejuizo_leve = (df['lucro'] <= 0) & (~mask_perda_severa)

print(f"\n📊 Distribuição de Lucro:")
print(f"   Total de casos: {len(df):,}")
print(f"   Lucrativos (lucro > 0): {mask_lucrativo.sum():,} ({mask_lucrativo.mean():.1%})")
print(f"   Prejuízos Leves (0 a P10): {mask_prejuizo_leve.sum():,} ({mask_prejuizo_leve.mean():.1%})")
print(f"   Perdas Severas (≤ P10): {mask_perda_severa.sum():,} ({mask_perda_severa.mean():.1%})")
print(f"   Threshold P10: R$ {threshold_perda_severa:,.2f}")

# 2. SELEÇÃO DE FEATURES PARA ANOMALY DETECTION
# Problema: 10,794 features é DEMAIS para Isolation Forest (curse of dimensionality)
# Solução: Usar apenas features mais relevantes para lucro/prejuízo

print(f"\n🔍 Selecionando features mais relevantes para Anomaly Detection...")
print(f"   Total de features: {X.shape[1]:,}")

# Calcular correlação de cada feature com lucro
from sklearn.feature_selection import mutual_info_regression
correlacoes_lucro = []
for col in X.columns:
    if X[col].std() > 0:  # Apenas features com variância
        corr = np.corrcoef(X[col], df['lucro'])[0, 1]
        correlacoes_lucro.append((col, abs(corr)))

# Ordenar por correlação absoluta e pegar top features
correlacoes_lucro = sorted(correlacoes_lucro, key=lambda x: x[1], reverse=True)
top_n_features = 100  # Usar apenas 100 melhores features
top_features = [feat for feat, corr in correlacoes_lucro[:top_n_features]]

print(f"   Selecionadas: {len(top_features)} features mais correlacionadas com lucro")
print(f"   Top 5 features: {[f[:30] for f in top_features[:5]]}")

# Criar X reduzido
X_reduced = X[top_features].copy()

# 3. ESTRATÉGIA: Criar dataset BALANCEADO para treinar Isolation Forest
# - Amostra de lucrativos (representar normalidade)
# - Todos os prejuízos leves
# - SOBREAMOSTRAR perdas severas (fazer elas "gritarem")

n_sample_lucrativos = 3000
n_replicas_severas = 5  # ↑ 3 → 5 (mais peso para perdas severas)

X_sample_lucrativos = X_reduced[mask_lucrativo].sample(n=n_sample_lucrativos, random_state=42)
X_prejuizo_leve = X_reduced[mask_prejuizo_leve].copy()
X_perda_severa = X_reduced[mask_perda_severa].copy()

# Replicar perdas severas para dar mais peso
X_perda_severa_replicas = pd.concat([X_perda_severa] * n_replicas_severas, ignore_index=True)

# Combinar tudo
X_balanced = pd.concat([
    X_sample_lucrativos,
    X_prejuizo_leve,
    X_perda_severa_replicas
], ignore_index=True)

print(f"\n🎯 Treinando Anomaly Detector com dataset balanceado...")
print(f"   Features usadas: {X_balanced.shape[1]}")
print(f"   Lucrativos (amostra): {len(X_sample_lucrativos):,}")
print(f"   Prejuízos Leves: {len(X_prejuizo_leve):,}")
print(f"   Perdas Severas (replicadas {n_replicas_severas}x): {len(X_perda_severa_replicas):,}")
print(f"   Total para treino: {len(X_balanced):,}")
print(f"   Proporção Perdas Severas: {len(X_perda_severa_replicas)/len(X_balanced):.1%}")

# 4. Usar Isolation Forest otimizado
anomaly_detector = IsolationForest(
    n_estimators=500,      # Muitas árvores para estabilidade
    max_samples=min(256, len(X_balanced)),  # ↓ 512→256 (mais foco em outliers)
    contamination=0.20,    # ↑ 0.15→0.20 (mais permissivo)
    max_features=min(50, X_balanced.shape[1]),  # Usar até 50 features por árvore
    bootstrap=False,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

# Treinar no dataset balanceado
anomaly_detector.fit(X_balanced)

# 5. Calcular anomaly score para TODOS os dados originais (usando features reduzidas)
# Score negativo = mais anômalo (maior risco de perda severa)
anomaly_scores = anomaly_detector.decision_function(X_reduced)

# 6. Normalizar scores para [0, 1], onde 1 = maior risco de anomalia (perda severa)
# Usar percentis para normalização mais robusta
p1 = np.percentile(anomaly_scores, 1)
p99 = np.percentile(anomaly_scores, 99)
anomaly_scores_norm = np.clip((anomaly_scores - p1) / (p99 - p1), 0, 1)
anomaly_scores_norm = 1 - anomaly_scores_norm  # Inverter: 1 = alta anomalia (alto risco)

# 6. Adicionar como nova feature
X['anomaly_score'] = anomaly_scores_norm

# 7. Validação detalhada: verificar se está funcionando
print(f"\n🔬 Validação do Anomaly Detector:")
print(f"   Range de scores originais: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
print(f"   Range normalizado: [{anomaly_scores_norm.min():.4f}, {anomaly_scores_norm.max():.4f}]")

# Criar grupos por lucro (usar as mesmas máscaras)

grupos = {
    'Lucrativos (Lucro > 0)': mask_lucrativo,
    'Prejuízos Leves (0 a P10)': mask_prejuizo_leve,
    'Perdas Severas (≤ P10)': mask_perda_severa
}

print(f"\n✅ Anomaly Score por Grupo:")
for nome, mask in grupos.items():
    if mask.sum() > 0:
        scores = anomaly_scores_norm[mask]
        lucro_medio = df.loc[mask, 'lucro'].mean()
        print(f"   {nome:28s}:")
        print(f"      • Média Score: {scores.mean():.4f} | Mediana: {np.median(scores):.4f}")
        print(f"      • Min: {scores.min():.4f} | Max: {scores.max():.4f}")
        print(f"      • N casos: {mask.sum():,} | Lucro Médio: R$ {lucro_medio:,.0f}")

# Calcular separação entre grupos (KS statistic ou diferença de médias)
score_lucrativo = anomaly_scores_norm[mask_lucrativo].mean()
score_severo = anomaly_scores_norm[mask_perda_severa].mean()
separacao = abs(score_severo - score_lucrativo)

print(f"\n📊 Métrica de Separação:")
print(f"   Diferença Score (Severo - Lucrativo): {score_severo - score_lucrativo:+.4f}")
print(f"   Separação Absoluta: {separacao:.4f}")
if separacao < 0.1:
    print(f"   ⚠️  ALERTA: Separação baixa! Anomaly detector pode não estar efetivo.")
elif separacao < 0.2:
    print(f"   ⚠️  Separação moderada. Considere ajustar parâmetros.")
else:
    print(f"   ✅ Boa separação! Detector está identificando padrões distintos.")

# Análise de anomalias por faixa de lucro
print("\n📈 Anomaly Score por Faixa de Lucro:")
lucro_bins = [-np.inf, df['lucro'].quantile(0.1), 0, df['lucro'].quantile(0.9), np.inf]
lucro_labels = ['Perdas Severas (P10)', 'Perdas Leves', 'Lucros Leves', 'Lucros Altos (P90)']
df['lucro_faixa'] = pd.cut(df['lucro'], bins=lucro_bins, labels=lucro_labels)

for faixa in lucro_labels:
    mask_faixa = df['lucro_faixa'] == faixa
    score_medio = anomaly_scores_norm[mask_faixa].mean()
    n_casos = mask_faixa.sum()
    print(f"   {faixa:25s}: {score_medio:.4f} ({n_casos:,} casos)")

# Gráfico de diagnóstico rápido
print("\n📊 Gerando gráfico de diagnóstico do Anomaly Detector...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Distribuição de scores por grupo
ax1 = axes[0]
for nome, mask in grupos.items():
    if mask.sum() > 0:
        ax1.hist(anomaly_scores_norm[mask], bins=50, alpha=0.5, label=nome)
ax1.set_xlabel('Anomaly Score')
ax1.set_ylabel('Frequência')
ax1.set_title('Distribuição do Anomaly Score por Grupo')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Boxplot por grupo
ax2 = axes[1]
data_boxplot = [anomaly_scores_norm[mask] for mask in grupos.values() if mask.sum() > 0]
labels_boxplot = [nome.split('(')[0].strip() for nome, mask in grupos.items() if mask.sum() > 0]
bp = ax2.boxplot(data_boxplot, labels=labels_boxplot, patch_artist=True)
colors = ['green', 'orange', 'red']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax2.set_ylabel('Anomaly Score')
ax2.set_title('Distribuição do Anomaly Score por Grupo')
ax2.grid(True, alpha=0.3, axis='y')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/diagnostico_anomaly_detector.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Salvo em: {OUTPUT_DIR}/diagnostico_anomaly_detector.png")

# Split
y_class = df['target_class'].values
y_raw_lucro = df['lucro'].values

X_train, X_test, y_train_class, y_test_class, y_train_lucro, y_test_lucro = train_test_split(
    X, y_class, y_raw_lucro, test_size=0.2, random_state=42, stratify=y_class
)

# Preparação para Regressores (Upside e Downside)
# Upside (Treino apenas nos lucrativos)
mask_train_up = y_train_lucro > 0
X_train_up = X_train[mask_train_up]
y_train_up_log = np.log1p(y_train_lucro[mask_train_up])

# Downside (Treino apenas nos prejuízos)
mask_train_down = y_train_lucro <= 0
X_train_down = X_train[mask_train_down]
y_train_down_log = np.log1p(np.abs(y_train_lucro[mask_train_down]))

print(f"\n🔧 Dimensões de Treino:")
print(f"   Classificador (Todos): {X_train.shape}")
print(f"   Regressor Upside (Lucrativos): {X_train_up.shape}")
print(f"   Regressor Downside (Prejuízos): {X_train_down.shape} <- Foco na Cauda Longa")

# Scaling
scaler = RobustScaler() # RobustScaler é melhor para outliers
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Subsets scaled para regressores
X_train_up_scaled = X_train_scaled[mask_train_up]
X_train_down_scaled = X_train_scaled[mask_train_down]

# --------------------------------------
# 2. MODELAGEM - ESTÁGIO 1: CLASSIFICADOR (PROFIT PROBABILITY)
# --------------------------------------
print("\n" + "="*60)
print("🔮 ESTÁGIO 1: CLASSIFICADOR DE LUCRATIVIDADE")
print("="*60)

# Usaremos um Ensemble simples de LGBM + XGB para o classificador
lgb_clf_params = {
    'n_estimators': 5000,
    'learning_rate': 0.01,
    'max_depth': 8,
    'num_leaves': 255,
    'min_child_samples': 15,
    'subsample': 0.85,
    'subsample_freq': 1,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': -1
}

xgb_clf_params = {
    'n_estimators': 5000,
    'learning_rate': 0.01,
    'max_depth': 8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'colsample_bylevel': 0.85,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': 0
}

print("   Treinando LGBM Classifier (Modo Pesado)...")
clf_lgb = lgbm.LGBMClassifier(**lgb_clf_params, objective='binary', metric='auc')
clf_lgb.fit(X_train, y_train_class, eval_set=[(X_test, y_test_class)], callbacks=[lgbm.early_stopping(200, verbose=False)])

print("   Treinando XGB Classifier (Modo Pesado)...")
clf_xgb = xgb.XGBClassifier(**xgb_clf_params, objective='binary:logistic', eval_metric='auc', tree_method='hist', early_stopping_rounds=200)
clf_xgb.fit(X_train, y_train_class, eval_set=[(X_test, y_test_class)], verbose=False)

print("   Treinando CatBoost Classifier (Adicionado ao Ensemble)...")
clf_cat = CatBoostClassifier(
    iterations=5000,
    learning_rate=0.01,
    depth=9,
    l2_leaf_reg=3.0,
    subsample=0.85,
    random_seed=42,
    verbose=False,
    allow_writing_files=False
)
clf_cat.fit(X_train, y_train_class, eval_set=(X_test, y_test_class), early_stopping_rounds=200, verbose=False)

# Probabilidades (Ensemble com 3 modelos)
prob_lgb = clf_lgb.predict_proba(X_test)[:, 1]
prob_xgb = clf_xgb.predict_proba(X_test)[:, 1]
prob_cat = clf_cat.predict_proba(X_test)[:, 1]
prob_class_test = 0.4 * prob_lgb + 0.35 * prob_xgb + 0.25 * prob_cat

auc_lgb = roc_auc_score(y_test_class, prob_lgb)
auc_xgb = roc_auc_score(y_test_class, prob_xgb)
auc_cat = roc_auc_score(y_test_class, prob_cat)
auc_score = roc_auc_score(y_test_class, prob_class_test)

print(f"   → LGBM AUC: {auc_lgb:.4f}")
print(f"   → XGB AUC: {auc_xgb:.4f}")
print(f"   → CAT AUC: {auc_cat:.4f}")
print(f"   ✅ AUC (Ensemble 3 modelos): {auc_score:.4f}")

# Probabilidades para o conjunto de treino (para uso futuro se necessário, aqui simplificado)
prob_class_train = 0.5 * clf_lgb.predict_proba(X_train)[:, 1] + 0.5 * clf_xgb.predict_proba(X_train)[:, 1]

# --------------------------------------
# 3. MODELAGEM - ESTÁGIO 2: REGRESSOR UPSIDE (POTENCIAL)
# --------------------------------------
print("\n" + "="*60)
print("📈 ESTÁGIO 2: REGRESSOR UPSIDE (Potencial de Lucro)")
print("="*60)

lgb_reg_params = {
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
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': -1
}

xgb_reg_params = {
    'n_estimators': 6000,
    'learning_rate': 0.008,
    'max_depth': 9,
    'min_child_weight': 2,
    'gamma': 0.05,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'colsample_bylevel': 0.85,
    'reg_alpha': 0.2,
    'reg_lambda': 0.8,
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': 0
}

print("   Treinando LGBM Regressor (Upside - Modo Pesado)...")
reg_up_lgb = lgbm.LGBMRegressor(**lgb_reg_params, objective='regression', metric='rmse')
reg_up_lgb.fit(X_train_up, y_train_up_log, eval_set=[(X_train_up[:5000], y_train_up_log[:5000])], callbacks=[lgbm.early_stopping(250, verbose=False)])

print("   Treinando XGB Regressor (Upside - Adicionado)...")
reg_up_xgb = xgb.XGBRegressor(**xgb_reg_params, objective='reg:squarederror', tree_method='hist', early_stopping_rounds=250)
reg_up_xgb.fit(X_train_up, y_train_up_log, eval_set=[(X_train_up[:5000], y_train_up_log[:5000])], verbose=False)

print("   Treinando CatBoost Regressor (Upside - Modo Pesado)...")
reg_up_cat = CatBoostRegressor(
    iterations=6000, 
    learning_rate=0.008, 
    depth=9, 
    l2_leaf_reg=4.0,
    subsample=0.85,
    verbose=False, 
    allow_writing_files=False
)
reg_up_cat.fit(X_train_up, y_train_up_log, eval_set=(X_train_up[:5000], y_train_up_log[:5000]), early_stopping_rounds=250, verbose=False)

# Predições (em log)
pred_log_up_lgb = reg_up_lgb.predict(X_test)
pred_log_up_xgb = reg_up_xgb.predict(X_test)
pred_log_up_cat = reg_up_cat.predict(X_test)

# Ensemble triplo e Inversão do Log (expm1)
pred_log_up = 0.4 * pred_log_up_lgb + 0.35 * pred_log_up_xgb + 0.25 * pred_log_up_cat
pred_upside_test = np.expm1(pred_log_up) # Valor estimado do lucro SE for lucrativo

# --------------------------------------
# 4. MODELAGEM - ESTÁGIO 3: REGRESSOR DOWNSIDE (RISCO)
# --------------------------------------
print("\n" + "="*60)
print("📉 ESTÁGIO 3: REGRESSOR DOWNSIDE (Risco de Perda)")
print("   * Foco na Cauda Longa *")
print("="*60)

# Para o downside, queremos ser sensíveis a grandes perdas.
# Usaremos Quantile Regression ou Huber Loss se possível, mas log-transform já ajuda muito.
print("   Treinando LGBM Regressor (Downside - Modo Pesado)...")
reg_down_lgb = lgbm.LGBMRegressor(**lgb_reg_params, objective='regression', metric='rmse')
reg_down_lgb.fit(X_train_down, y_train_down_log, eval_set=[(X_train_down[:2000], y_train_down_log[:2000])], callbacks=[lgbm.early_stopping(250, verbose=False)])

print("   Treinando XGB Regressor (Downside - Modo Pesado)...")
reg_down_xgb = xgb.XGBRegressor(**xgb_reg_params, objective='reg:squarederror', tree_method='hist', early_stopping_rounds=250)
reg_down_xgb.fit(X_train_down, y_train_down_log, eval_set=[(X_train_down[:2000], y_train_down_log[:2000])], verbose=False)

print("   Treinando CatBoost Regressor (Downside - Adicionado)...")
reg_down_cat = CatBoostRegressor(
    iterations=6000, 
    learning_rate=0.008, 
    depth=9, 
    l2_leaf_reg=4.0,
    subsample=0.85,
    verbose=False, 
    allow_writing_files=False
)
reg_down_cat.fit(X_train_down, y_train_down_log, eval_set=(X_train_down[:2000], y_train_down_log[:2000]), early_stopping_rounds=250, verbose=False)

# Predições (em log)
pred_log_down_lgb = reg_down_lgb.predict(X_test)
pred_log_down_xgb = reg_down_xgb.predict(X_test)
pred_log_down_cat = reg_down_cat.predict(X_test)

# Ensemble triplo e Inversão
pred_log_down = 0.4 * pred_log_down_lgb + 0.35 * pred_log_down_xgb + 0.25 * pred_log_down_cat
pred_downside_test = np.expm1(pred_log_down) # Valor estimado do prejuízo (positivo) SE der prejuízo

# --------------------------------------
# 5. CÁLCULO DO VALOR ESPERADO (EV)
# --------------------------------------
print("\n" + "="*60)
print("🧮 CÁLCULO DO VALOR ESPERADO (EV)")
print("="*60)

# EV = P(Lucro) * E[Upside] - (1 - P(Lucro)) * E[Downside]
# P(Lucro) = prob_class_test
# P(Prejuízo) = 1 - prob_class_test

expected_value = (prob_class_test * pred_upside_test) - ((1 - prob_class_test) * pred_downside_test)

# Avaliação da Qualidade do EV
# Vamos ver a correlação do EV com o Lucro Real
correlation = np.corrcoef(expected_value, y_test_lucro)[0, 1]
mae_ev = mean_absolute_error(y_test_lucro, expected_value)

print(f"   Correlação EV vs Lucro Real: {correlation:.4f}")
print(f"   MAE do EV: R$ {mae_ev:.2f}")

# --------------------------------------
# 6. OTIMIZAÇÃO DE CORTE (PARETO SIMPLIFICADO)
# --------------------------------------
print("\n" + "="*60)
print("⚖️  OTIMIZAÇÃO DE DECISÃO")
print("="*60)

# Ordenar por EV decrescente
results_df = pd.DataFrame({
    'real_profit': y_test_lucro,
    'expected_value': expected_value,
    'prob_profit': prob_class_test,
    'pred_upside': pred_upside_test,
    'pred_downside': pred_downside_test
})

results_df = results_df.sort_values('expected_value', ascending=False).reset_index(drop=True)

# Calcular curvas cumulativas
results_df['cum_profit'] = results_df['real_profit'].cumsum()
results_df['cum_contracts'] = np.arange(1, len(results_df) + 1)
results_df['roi'] = results_df['cum_profit'] / results_df['cum_contracts'] # Lucro médio por contrato

# Encontrar ponto ótimo (Máximo Lucro Total)
max_profit_idx = results_df['cum_profit'].idxmax()
best_threshold_ev = results_df.loc[max_profit_idx, 'expected_value']
max_profit = results_df.loc[max_profit_idx, 'cum_profit']
contracts_accepted = max_profit_idx + 1

# Baseline (Aceitar tudo)
baseline_profit = results_df['real_profit'].sum()

print(f"   Cenário Baseline (Todos): R$ {baseline_profit:,.2f}")
print(f"   Cenário Otimizado (EV > {best_threshold_ev:.2f}): R$ {max_profit:,.2f}")
print(f"   Contratos Aceitos: {contracts_accepted} de {len(results_df)} ({contracts_accepted/len(results_df):.1%})")
print(f"   Ganho Financeiro: R$ {max_profit - baseline_profit:,.2f} ({((max_profit - baseline_profit)/baseline_profit)*100:.1f}%)")

# --------------------------------------
# 7. VISUALIZAÇÃO E RELATÓRIO
# --------------------------------------

# Gráfico 1: Curva de Lucro Acumulado
plt.figure(figsize=(10, 6))
plt.plot(results_df['cum_contracts'], results_df['cum_profit'], label='Modelo Profit-Loss', color='purple', linewidth=2)
plt.axhline(y=baseline_profit, color='gray', linestyle='--', label='Baseline (Aceitar Todos)')
plt.axvline(x=contracts_accepted, color='green', linestyle=':', label='Ponto de Corte Ótimo')
plt.title('Curva de Lucro Acumulado (Ordenado por EV)')
plt.xlabel('Número de Contratos Aceitos')
plt.ylabel('Lucro Acumulado (R$)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{OUTPUT_DIR}/curva_lucro_acumulado.png')
plt.close()

# Gráfico 2: Dispersão EV vs Real (Hexbin para densidade)
plt.figure(figsize=(10, 6))
plt.hexbin(results_df['expected_value'], results_df['real_profit'], gridsize=50, cmap='inferno', mincnt=1, bins='log')
plt.colorbar(label='log10(N)')
plt.xlabel('Valor Esperado (Predito)')
plt.ylabel('Lucro Real')
plt.title('Dispersão: Valor Esperado vs Realidade')
plt.grid(True, alpha=0.3)
plt.savefig(f'{OUTPUT_DIR}/dispersao_ev_real.png')
plt.close()

# Gráfico 3: Análise da Cauda (Downside)
# Vamos ver se o modelo Downside consegue identificar as grandes perdas
losses_df = results_df[results_df['real_profit'] < 0].copy()
plt.figure(figsize=(10, 6))
sns.scatterplot(data=losses_df, x='pred_downside', y='real_profit', alpha=0.5)
plt.xlabel('Prejuízo Predito (Magnitude)')
plt.ylabel('Prejuízo Real (Negativo)')
plt.title('Capacidade de Identificar Grandes Perdas')
plt.gca().invert_yaxis() # Inverter eixo Y para mostrar perdas maiores para baixo
plt.grid(True, alpha=0.3)
plt.savefig(f'{OUTPUT_DIR}/analise_cauda_perdas.png')
plt.close()

# Gráfico 4: Análise do Anomaly Score
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 4.1: Distribuição do Anomaly Score por Classe
ax1 = axes[0, 0]
anomaly_test = X_test['anomaly_score'].values
lucro_positivo = y_test_lucro > 0
ax1.hist([anomaly_test[lucro_positivo], anomaly_test[~lucro_positivo]], 
         bins=50, label=['Lucrativo', 'Prejuízo'], alpha=0.7, color=['green', 'red'])
ax1.set_xlabel('Anomaly Score')
ax1.set_ylabel('Frequência')
ax1.set_title('Distribuição do Anomaly Score por Classe')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 4.2: Anomaly Score vs Lucro Real
ax2 = axes[0, 1]
scatter = ax2.scatter(anomaly_test, y_test_lucro, c=y_test_lucro, 
                     cmap='RdYlGn', alpha=0.5, s=20)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('Anomaly Score')
ax2.set_ylabel('Lucro Real (R$)')
ax2.set_title('Anomaly Score vs Lucro Real')
plt.colorbar(scatter, ax=ax2, label='Lucro (R$)')
ax2.grid(True, alpha=0.3)

# 4.3: Anomaly Score por Faixa de Lucro
ax3 = axes[1, 0]
# Recriar as faixas para o conjunto de teste
lucro_test_bins = [-np.inf, np.percentile(y_test_lucro, 10), 0, 
                   np.percentile(y_test_lucro, 90), np.inf]
lucro_test_labels = ['Perdas\nSeveras', 'Perdas\nLeves', 'Lucros\nLeves', 'Lucros\nAltos']
lucro_test_faixa = pd.cut(y_test_lucro, bins=lucro_test_bins, labels=lucro_test_labels)

anomaly_by_faixa = [anomaly_test[lucro_test_faixa == label].tolist() for label in lucro_test_labels]
bp = ax3.boxplot(anomaly_by_faixa, labels=lucro_test_labels, patch_artist=True)
colors = ['darkred', 'salmon', 'lightgreen', 'darkgreen']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax3.set_ylabel('Anomaly Score')
ax3.set_title('Anomaly Score por Faixa de Lucro')
ax3.grid(True, alpha=0.3, axis='y')

# 4.4: Top Anomalias - Perdas Reais
ax4 = axes[1, 1]
# Pegar top 20% mais anômalos e ver suas perdas reais
threshold_anomaly = np.percentile(anomaly_test, 80)
high_anomaly_mask = anomaly_test >= threshold_anomaly
lucro_alto_anomaly = y_test_lucro[high_anomaly_mask]
lucro_baixo_anomaly = y_test_lucro[~high_anomaly_mask]

data_to_plot = [lucro_baixo_anomaly, lucro_alto_anomaly]
bp2 = ax4.boxplot(data_to_plot, labels=['Anomaly < P80', 'Anomaly ≥ P80'], patch_artist=True)
bp2['boxes'][0].set_facecolor('lightblue')
bp2['boxes'][1].set_facecolor('orange')
for box in bp2['boxes']:
    box.set_alpha(0.6)
ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax4.set_ylabel('Lucro Real (R$)')
ax4.set_title('Distribuição de Lucro: Baixa vs Alta Anomalia')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/analise_anomaly_score.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n📊 Análise do Anomaly Score:")
print(f"   Correlação Anomaly Score vs Lucro: {np.corrcoef(anomaly_test, y_test_lucro)[0,1]:.4f}")
print(f"   Lucro Médio (Baixa Anomalia < P80): R$ {lucro_baixo_anomaly.mean():,.2f}")
print(f"   Lucro Médio (Alta Anomalia ≥ P80): R$ {lucro_alto_anomaly.mean():,.2f}")
print(f"   % Prejuízo (Baixa Anomalia): {(lucro_baixo_anomaly < 0).mean():.1%}")
print(f"   % Prejuízo (Alta Anomalia): {(lucro_alto_anomaly < 0).mean():.1%}")

# Gráfico 5: Análise Temporal Detalhada
if 'Data_Lancamento' in df.columns:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Preparar dados temporais do conjunto de teste
    df_test = df.loc[X_test.index].copy()
    df_test['lucro_real'] = y_test_lucro
    df_test['ev_predito'] = expected_value
    df_test['prejuizo'] = (y_test_lucro < 0).astype(int)
    
    # 5.1: Taxa de Prejuízo por Mês
    ax1 = axes[0, 0]
    prejuizo_mes = df_test.groupby('mes')['prejuizo'].agg(['mean', 'count'])
    meses_ordem = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    prejuizo_mes = prejuizo_mes.reindex(meses_ordem, fill_value=0)
    
    colors = ['red' if m in [11, 12, 1, 2, 3] else 'lightblue' for m in prejuizo_mes.index]
    bars = ax1.bar(prejuizo_mes.index, prejuizo_mes['mean'], color=colors, alpha=0.7)
    ax1.set_xlabel('Mês')
    ax1.set_ylabel('Taxa de Prejuízo')
    ax1.set_title('Taxa de Prejuízo por Mês (Teste)\n(Vermelho = Período Crítico)')
    ax1.set_xticks(meses_ordem)
    ax1.set_xticklabels(['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                        'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=prejuizo_mes['mean'].mean(), color='black', linestyle='--', 
                linewidth=1, label='Média Geral')
    ax1.legend()
    
    # 5.2: Evolução Temporal (Ano-Mês)
    ax2 = axes[0, 1]
    if 'ano_mes' in df_test.columns:
        # Agrupar por ano_mes e calcular métricas
        temporal_agg = df_test.groupby('ano_mes').agg({
            'prejuizo': 'mean',
            'lucro_real': 'mean'
        }).reset_index()
        
        # Ordenar cronologicamente
        temporal_agg = temporal_agg.sort_values('ano_mes')
        
        # Limitar visualização para não ficar muito poluído
        if len(temporal_agg) > 36:  # Mais de 3 anos
            temporal_agg = temporal_agg.tail(36)  # Últimos 3 anos
        
        x_pos = np.arange(len(temporal_agg))
        ax2.plot(x_pos, temporal_agg['prejuizo'], marker='o', linewidth=2, 
                markersize=6, color='red', label='Taxa Prejuízo')
        ax2.set_xlabel('Período (Ano-Mês)')
        ax2.set_ylabel('Taxa de Prejuízo', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_xticks(x_pos[::3])  # Mostrar a cada 3 meses
        ax2.set_xticklabels(temporal_agg['ano_mes'].iloc[::3], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Segundo eixo Y para lucro médio
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x_pos, temporal_agg['lucro_real'], marker='s', linewidth=2, 
                     markersize=6, color='green', alpha=0.6, label='Lucro Médio')
        ax2_twin.set_ylabel('Lucro Médio (R$)', color='green')
        ax2_twin.tick_params(axis='y', labelcolor='green')
        ax2_twin.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        ax2.set_title('Evolução Temporal: Prejuízo e Lucro por Período')
        
        # Combinar legendas
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 5.3: Comparação Períodos Críticos
    ax3 = axes[1, 0]
    periodos = ['Normal', 'Fim de Ano\n(Nov-Jan)', 'Carnaval\n(Fev-Mar)']
    
    mask_normal = df_test['periodo_critico'] == 0
    mask_fim_ano = df_test['periodo_fim_ano'] == 1
    mask_carnaval = df_test['periodo_carnaval'] == 1
    
    taxa_prejuizo_periodos = [
        df_test.loc[mask_normal, 'prejuizo'].mean(),
        df_test.loc[mask_fim_ano, 'prejuizo'].mean(),
        df_test.loc[mask_carnaval, 'prejuizo'].mean()
    ]
    
    lucro_medio_periodos = [
        df_test.loc[mask_normal, 'lucro_real'].mean(),
        df_test.loc[mask_fim_ano, 'lucro_real'].mean(),
        df_test.loc[mask_carnaval, 'lucro_real'].mean()
    ]
    
    x_pos_periodos = np.arange(len(periodos))
    bars1 = ax3.bar(x_pos_periodos - 0.2, taxa_prejuizo_periodos, 0.4, 
                   label='Taxa Prejuízo', color='red', alpha=0.7)
    
    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x_pos_periodos + 0.2, lucro_medio_periodos, 0.4, 
                        label='Lucro Médio', color='green', alpha=0.7)
    
    ax3.set_xlabel('Período')
    ax3.set_ylabel('Taxa de Prejuízo', color='red')
    ax3_twin.set_ylabel('Lucro Médio (R$)', color='green')
    ax3.set_title('Comparação: Períodos Críticos vs Normal')
    ax3.set_xticks(x_pos_periodos)
    ax3.set_xticklabels(periodos)
    ax3.tick_params(axis='y', labelcolor='red')
    ax3_twin.tick_params(axis='y', labelcolor='green')
    ax3_twin.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # Combinar legendas
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 5.4: Heatmap Ano vs Mês
    ax4 = axes[1, 1]
    if 'ano' in df_test.columns and 'mes' in df_test.columns:
        # Criar pivot table
        heatmap_data = df_test.pivot_table(
            values='prejuizo', 
            index='ano', 
            columns='mes', 
            aggfunc='mean'
        )
        
        # Garantir que todos os meses estejam presentes
        for mes in range(1, 13):
            if mes not in heatmap_data.columns:
                heatmap_data[mes] = np.nan
        
        heatmap_data = heatmap_data[sorted(heatmap_data.columns)]
        
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                   center=heatmap_data.values[~np.isnan(heatmap_data.values)].mean(),
                   cbar_kws={'label': 'Taxa de Prejuízo'}, ax=ax4)
        ax4.set_xlabel('Mês')
        ax4.set_ylabel('Ano')
        ax4.set_title('Heatmap: Taxa de Prejuízo (Ano vs Mês)')
        ax4.set_xticklabels(['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                           'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/analise_temporal_detalhada.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n📅 Análise Temporal Salva!")

# Salvar Resumo
with open(f'{OUTPUT_DIR}/resumo_profit_loss.txt', 'w') as f:
    f.write("MODELO PROFIT-LOSS HURDLE COM ANOMALY DETECTION\n")
    f.write("===============================================\n\n")
    f.write(f"AUC Classificador: {auc_score:.4f}\n")
    f.write(f"Correlação EV vs Real: {correlation:.4f}\n\n")
    f.write("ANOMALY DETECTION (Cauda Longa de Perdas):\n")
    f.write(f"Correlação Anomaly Score vs Lucro: {np.corrcoef(anomaly_test, y_test_lucro)[0,1]:.4f}\n")
    f.write(f"Lucro Médio (Baixa Anomalia): R$ {lucro_baixo_anomaly.mean():,.2f}\n")
    f.write(f"Lucro Médio (Alta Anomalia): R$ {lucro_alto_anomaly.mean():,.2f}\n")
    f.write(f"% Prejuízo (Baixa Anomalia): {(lucro_baixo_anomaly < 0).mean():.1%}\n")
    f.write(f"% Prejuízo (Alta Anomalia): {(lucro_alto_anomaly < 0).mean():.1%}\n\n")
    f.write("RESULTADOS FINANCEIROS:\n")
    f.write(f"Lucro Baseline: R$ {baseline_profit:,.2f}\n")
    f.write(f"Lucro Otimizado: R$ {max_profit:,.2f}\n")
    f.write(f"Ganho Líquido: R$ {max_profit - baseline_profit:,.2f}\n")
    f.write(f"Taxa de Aprovação: {contracts_accepted/len(results_df):.1%}\n")
    f.write(f"Threshold de EV: R$ {best_threshold_ev:.2f}\n")

print(f"\n✅ Processo concluído! Gráficos e resumo salvos em {OUTPUT_DIR}")

# --------------------------------------
# 7.5 SALVAR MODELOS PARA PRODUÇÃO
# --------------------------------------
print("\n" + "="*60)
print("💾 SALVANDO MODELOS PARA PRODUÇÃO")
print("="*60)

MODEL_DIR = '../../modelos_treinados/PROFIT_LOSS_HURDLE'
os.makedirs(MODEL_DIR, exist_ok=True)

import joblib
import json

print("\n📦 Salvando modelos treinados...")

# 1. Classificadores (Ensemble)
print("   Salvando classificadores...")
clf_lgb.booster_.save_model(f'{MODEL_DIR}/clf_lgb.txt')
clf_xgb.save_model(f'{MODEL_DIR}/clf_xgb.json')
clf_cat.save_model(f'{MODEL_DIR}/clf_cat.cbm')

# 2. Regressores Upside (Ensemble)
print("   Salvando regressores Upside...")
reg_up_lgb.booster_.save_model(f'{MODEL_DIR}/reg_upside_lgb.txt')
reg_up_xgb.save_model(f'{MODEL_DIR}/reg_upside_xgb.json')
reg_up_cat.save_model(f'{MODEL_DIR}/reg_upside_cat.cbm')

# 3. Regressores Downside (Ensemble)
print("   Salvando regressores Downside...")
reg_down_lgb.booster_.save_model(f'{MODEL_DIR}/reg_downside_lgb.txt')
reg_down_xgb.save_model(f'{MODEL_DIR}/reg_downside_xgb.json')
reg_down_cat.save_model(f'{MODEL_DIR}/reg_downside_cat.cbm')

# 4. Scaler
print("   Salvando scaler...")
joblib.dump(scaler, f'{MODEL_DIR}/scaler.pkl')

# 4.5. Anomaly Detector e suas dependências
print("   Salvando anomaly detector e features...")
joblib.dump(anomaly_detector, f'{MODEL_DIR}/anomaly_detector.pkl')
joblib.dump(top_features, f'{MODEL_DIR}/anomaly_detector_features.pkl')  # Features usadas

# 5. Metadados e Configuração
print("   Salvando metadados...")
metadata = {
    'model_name': 'Profit-Loss Hurdle Model with Anomaly Detection',
    'trained_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_features': len(X.columns),
    'feature_names': X.columns.tolist(),
    'metrics': {
        'auc_lgb': float(auc_lgb),
        'auc_xgb': float(auc_xgb),
        'auc_cat': float(auc_cat),
        'auc_ensemble': float(auc_score),
        'correlation_ev_real': float(correlation),
        'mae_ev': float(mae_ev),
        'anomaly_score_correlation': float(np.corrcoef(anomaly_test, y_test_lucro)[0,1])
    },
    'anomaly_detection': {
        'method': 'Isolation Forest with feature selection and balanced dataset',
        'contamination': 0.20,
        'n_estimators': 500,
        'n_features_selected': len(top_features),
        'max_features_per_tree': min(50, len(top_features)),
        'training_strategy': 'oversampled_severe_losses_5x',
        'replicas_severe': 5,
        'avg_score_profitable': float(anomaly_scores_norm[mask_lucrativo].mean()),
        'avg_score_light_loss': float(anomaly_scores_norm[mask_prejuizo_leve].mean()),
        'avg_score_severe_loss': float(anomaly_scores_norm[mask_perda_severa].mean()),
        'separation_score': float(abs(anomaly_scores_norm[mask_perda_severa].mean() - anomaly_scores_norm[mask_lucrativo].mean()))
    },
    'business_metrics': {
        'baseline_profit': float(baseline_profit),
        'optimized_profit': float(max_profit),
        'gain': float(max_profit - baseline_profit),
        'threshold_ev': float(best_threshold_ev),
        'approval_rate': float(contracts_accepted/len(results_df)),
        'profit_low_anomaly': float(lucro_baixo_anomaly.mean()),
        'profit_high_anomaly': float(lucro_alto_anomaly.mean()),
        'loss_rate_low_anomaly': float((lucro_baixo_anomaly < 0).mean()),
        'loss_rate_high_anomaly': float((lucro_alto_anomaly < 0).mean())
    },
    'ensemble_weights': {
        'classifier': {'lgb': 0.4, 'xgb': 0.35, 'cat': 0.25},
        'regressor_upside': {'lgb': 0.4, 'xgb': 0.35, 'cat': 0.25},
        'regressor_downside': {'lgb': 0.4, 'xgb': 0.35, 'cat': 0.25}
    },
    'training_samples': {
        'total': int(len(X_train)),
        'upside': int(len(X_train_up)),
        'downside': int(len(X_train_down)),
        'anomaly_training': int(len(X_balanced)),  # Dataset balanceado
        'severe_losses': int(mask_perda_severa.sum())
    }
}

with open(f'{MODEL_DIR}/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\n✅ Modelos salvos com sucesso em: {MODEL_DIR}")
print(f"   • 9 modelos (3 classificadores + 3 reg_upside + 3 reg_downside)")
print(f"   • 1 scaler")
print(f"   • 1 anomaly detector (Isolation Forest)")
print(f"   • 1 lista de features do anomaly detector")
print(f"   • 1 arquivo de metadados")
print(f"   Total: 13 arquivos para produção")

# --------------------------------------
# 8. INTERPRETABILIDADE E ANÁLISE DE DECISÕES
# --------------------------------------
print("\n" + "="*60)
print("🔍 INTERPRETABILIDADE: ANÁLISE DE DECISÕES")
print("="*60)

# 8.1 Importância de Features (Feature Importance Global)
print("\n📊 Calculando importância global de features...")

# Feature importance do classificador (LGBM)
feat_importance_clf = pd.DataFrame({
    'feature': X.columns,
    'importance_clf': clf_lgb.feature_importances_
}).sort_values('importance_clf', ascending=False).head(20)

# Feature importance dos regressores
feat_importance_up = pd.DataFrame({
    'feature': X.columns,
    'importance_upside': reg_up_lgb.feature_importances_
}).sort_values('importance_upside', ascending=False).head(20)

feat_importance_down = pd.DataFrame({
    'feature': X.columns,
    'importance_downside': reg_down_lgb.feature_importances_
}).sort_values('importance_downside', ascending=False).head(20)

# Visualização
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

axes[0].barh(feat_importance_clf['feature'][:15], feat_importance_clf['importance_clf'][:15], color='purple')
axes[0].set_xlabel('Importância')
axes[0].set_title('Top 15 Features - Classificador (Lucro vs Prejuízo)')
axes[0].invert_yaxis()

axes[1].barh(feat_importance_up['feature'][:15], feat_importance_up['importance_upside'][:15], color='green')
axes[1].set_xlabel('Importância')
axes[1].set_title('Top 15 Features - Regressor Upside (Potencial)')
axes[1].invert_yaxis()

axes[2].barh(feat_importance_down['feature'][:15], feat_importance_down['importance_downside'][:15], color='red')
axes[2].set_xlabel('Importância')
axes[2].set_title('Top 15 Features - Regressor Downside (Risco)')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_importance_global.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Importância global salva.")

# 8.2 SHAP Values (Explicação Local)
print("\n🎯 Calculando SHAP values para interpretação local...")

try:
    # Sample para SHAP (mais rápido e reduzido para evitar problemas de memória)
    sample_size = min(300, len(X_test))  # Reduzido de 500 para 300
    X_test_sample = X_test.sample(sample_size, random_state=42)

    # Converter para array numpy e garantir que é numérico
    X_test_sample_clean = X_test_sample.copy()
    for col in X_test_sample_clean.columns:
        X_test_sample_clean[col] = pd.to_numeric(X_test_sample_clean[col], errors='coerce')
    
    # Preencher NaNs com 0 (se houver após a conversão)
    X_test_sample_clean = X_test_sample_clean.fillna(0)
    
    # Remover infinitos
    X_test_sample_clean = X_test_sample_clean.replace([np.inf, -np.inf], 0)
    
    # Converter para array numpy float64
    X_test_sample_array = X_test_sample_clean.astype(np.float64).values
    
    print(f"   Amostra para SHAP: {X_test_sample_array.shape}")
    
    # Tentar com CatBoost primeiro (geralmente mais estável para SHAP)
    print("   Criando explainer (tentando CatBoost)...")
    try:
        # CatBoost é mais robusto para SHAP
        explainer_clf = shap.TreeExplainer(clf_cat)
        print("   ✓ Usando CatBoost para SHAP")
    except Exception as e1:
        print(f"   ⚠️  CatBoost falhou, tentando XGBoost: {str(e1)[:100]}")
        try:
            # Fallback para XGBoost
            explainer_clf = shap.TreeExplainer(clf_xgb)
            print("   ✓ Usando XGBoost para SHAP")
        except Exception as e2:
            print(f"   ⚠️  XGBoost falhou, tentando LGBM: {str(e2)[:100]}")
            # Último fallback - criar novo modelo LGBM simplificado
            from lightgbm import LGBMClassifier
            clf_lgb_simple = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbosity=-1
            )
            clf_lgb_simple.fit(X_train, y_train_class)
            explainer_clf = shap.TreeExplainer(clf_lgb_simple)
            print("   ✓ Usando LGBM simplificado para SHAP")
    
    print("   Calculando SHAP values...")
    shap_values_clf = explainer_clf.shap_values(X_test_sample_array)

    # Se retornar lista (binary), pegar classe positiva
    if isinstance(shap_values_clf, list):
        shap_values_clf = shap_values_clf[1]
    
    print("   Gerando gráfico SHAP...")
    
    # SHAP Summary Plot - usar DataFrame para garantir nomes corretos
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_clf, 
        X_test_sample_clean,  # Passar DataFrame ao invés de array
        max_display=15, 
        show=False,
        plot_type='dot'
    )
    plt.title('SHAP Summary - Classificador de Lucratividade', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/shap_summary_classificador.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ SHAP Summary (Classificador) salvo com sucesso!")
    
except Exception as e:
    import traceback
    print(f"   ⚠️  Erro ao gerar SHAP: {str(e)}")
    print(f"   Stack trace completo:")
    print(traceback.format_exc())
    print("   → Feature importance global ainda disponível para interpretação.")

# 8.3 Correlação Features vs Classe (Lucro vs Prejuízo)
print("\n📈 Analisando correlação Features × Classes...")

try:
    # Limpar dados antes de calcular correlação
    X_train_clean = X_train.copy()
    X_train_clean = X_train_clean.replace([np.inf, -np.inf], np.nan)
    X_train_clean = X_train_clean.fillna(X_train_clean.median())
    
    print(f"   Calculando correlações para {len(X_train_clean.columns)} features...")
    print(f"   (Método otimizado - cálculo direto ao invés de matriz completa)")
    
    # OTIMIZAÇÃO: Calcular correlação feature-por-feature ao invés de matriz completa
    # Isso é MUITO mais rápido: O(n) ao invés de O(n²)
    correlations = {}
    n_features = len(X_train_clean.columns)
    
    # Progresso a cada 10%
    checkpoint = n_features // 10
    
    for i, col in enumerate(X_train_clean.columns):
        if checkpoint > 0 and i % checkpoint == 0:
            progress = (i / n_features) * 100
            print(f"   Progresso: {progress:.0f}% ({i}/{n_features} features)")
        
        # Calcular correlação apenas desta feature com o target
        feature_values = X_train_clean[col].values
        
        # Verificar se tem variância (evitar divisão por zero)
        if np.std(feature_values) > 1e-10:
            corr = np.corrcoef(feature_values, y_train_class)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr
    
    print(f"   ✓ Correlações calculadas para {len(correlations)} features válidas")
    
    # Converter para Series e ordenar
    correlations = pd.Series(correlations).sort_values(ascending=False)

    # Top positivas (indicam lucro)
    top_positive = correlations.head(15)
    # Top negativas (indicam prejuízo)
    top_negative = correlations.tail(15)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].barh(range(len(top_positive)), top_positive.values, color='green', alpha=0.7)
    axes[0].set_yticks(range(len(top_positive)))
    axes[0].set_yticklabels(top_positive.index, fontsize=9)
    axes[0].set_xlabel('Correlação com Lucratividade')
    axes[0].set_title('Top 15 Features Positivas\n(Indicam LUCRO)')
    axes[0].invert_yaxis()
    axes[0].grid(alpha=0.3, axis='x')

    axes[1].barh(range(len(top_negative)), top_negative.values, color='red', alpha=0.7)
    axes[1].set_yticks(range(len(top_negative)))
    axes[1].set_yticklabels(top_negative.index, fontsize=9)
    axes[1].set_xlabel('Correlação com Lucratividade')
    axes[1].set_title('Top 15 Features Negativas\n(Indicam PREJUÍZO)')
    axes[1].invert_yaxis()
    axes[1].grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/correlacao_features_classe.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Correlação Features × Classe salva com sucesso!")
    
except Exception as e:
    import traceback
    print(f"   ⚠️  Erro ao calcular correlações: {str(e)}")
    print(f"   Stack trace: {traceback.format_exc()[:300]}")
    print("   → Pulando análise de correlação.")

# 8.4 Exemplos de Decisões Individuais
print("\n📋 Gerando exemplos de decisões individuais...")

try:
    # Selecionar exemplos representativos
    # 1. Melhor decisão (alto EV, alto lucro real)
    # 2. Pior decisão (baixo EV negativo, grande prejuízo)
    # 3. Falso Positivo (EV alto mas prejuízo real)
    # 4. Falso Negativo (EV baixo mas lucro real)

    results_df_test = results_df.copy()
    results_df_test['idx_original'] = X_test.index

    # Casos
    best_case_idx = results_df_test['expected_value'].idxmax()
    worst_case_idx = results_df_test['expected_value'].idxmin()

    # Falso Positivo: EV > threshold mas real_profit < 0
    fp_candidates = results_df_test[(results_df_test['expected_value'] > best_threshold_ev) & (results_df_test['real_profit'] < 0)]
    false_positive_idx = fp_candidates['real_profit'].idxmin() if len(fp_candidates) > 0 else None

    # Falso Negativo: EV < threshold mas real_profit > 0
    fn_candidates = results_df_test[(results_df_test['expected_value'] < best_threshold_ev) & (results_df_test['real_profit'] > 0)]
    false_negative_idx = fn_candidates['real_profit'].idxmax() if len(fn_candidates) > 0 else None

    # Gerar relatório de exemplos
    with open(f'{OUTPUT_DIR}/exemplos_decisoes.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EXEMPLOS DE DECISÕES INDIVIDUAIS - MODELO PROFIT-LOSS HURDLE\n")
        f.write("="*80 + "\n\n")
        
        def write_case(f, idx, title):
            if idx is None:
                f.write(f"\n{title}: NÃO ENCONTRADO\n")
                return
                
            row = results_df_test.loc[idx]
            orig_idx = int(row['idx_original'])
            
            f.write(f"\n{title}\n")
            f.write("-" * 80 + "\n")
            f.write(f"📍 Índice Original: {orig_idx}\n")
            f.write(f"💰 Lucro Real: R$ {row['real_profit']:.2f}\n")
            f.write(f"🎯 Valor Esperado (EV): R$ {row['expected_value']:.2f}\n")
            f.write(f"📊 P(Lucrativo): {row['prob_profit']:.2%}\n")
            f.write(f"📈 Upside Predito: R$ {row['pred_upside']:.2f}\n")
            f.write(f"📉 Downside Predito: R$ {row['pred_downside']:.2f}\n")
            
            decision = "✅ ACEITAR" if row['expected_value'] > best_threshold_ev else "❌ REJEITAR"
            f.write(f"🚦 Decisão: {decision} (Threshold: R$ {best_threshold_ev:.2f})\n")
            
            # Top 10 features deste contrato
            contract_features = X_test.loc[orig_idx]
            f.write(f"\n🔍 Top 10 Features deste Contrato:\n")
            
            # Ordenar por valor absoluto (features mais "ativas")
            top_features = contract_features.abs().sort_values(ascending=False).head(10)
            for feat_name in top_features.index:
                feat_val = contract_features[feat_name]
                f.write(f"   • {feat_name}: {feat_val:.4f}\n")
            
            f.write("\n")
        
        write_case(f, best_case_idx, "1️⃣ MELHOR CASO (Alto EV, Alto Lucro Real)")
        write_case(f, worst_case_idx, "2️⃣ PIOR CASO (Baixo EV, Grande Prejuízo)")
        write_case(f, false_positive_idx, "3️⃣ FALSO POSITIVO (Alto EV, mas Prejuízo Real)")
        write_case(f, false_negative_idx, "4️⃣ FALSO NEGATIVO (Baixo EV, mas Lucro Real)")

    print("   ✓ Exemplos de decisões salvos em exemplos_decisoes.txt")
    
except Exception as e:
    import traceback
    print(f"   ⚠️  Erro ao gerar exemplos de decisões: {str(e)}")
    print(f"   Stack trace: {traceback.format_exc()[:300]}")
    print("   → Pulando exemplos de decisões.")

# 8.5 Análise de Ativação Dimensional
print("\n🧠 Análise de Ativação Dimensional...")

try:
    # Para cada grupo (Aceitos vs Rejeitados), ver quais dimensões estão mais ativas
    accepted_mask = results_df_test['expected_value'] > best_threshold_ev
    accepted_indices = results_df_test[accepted_mask]['idx_original'].values
    rejected_indices = results_df_test[~accepted_mask]['idx_original'].values

    X_accepted = X_test.loc[accepted_indices]
    X_rejected = X_test.loc[rejected_indices]

    # Média de ativação por grupo
    mean_accepted = X_accepted.mean()
    mean_rejected = X_rejected.mean()

    # Diferença entre grupos (features que diferenciam aceitos de rejeitados)
    diff_activation = (mean_accepted - mean_rejected).abs().sort_values(ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(diff_activation)), diff_activation.values, color='teal', alpha=0.7)
    plt.yticks(range(len(diff_activation)), diff_activation.index, fontsize=9)
    plt.xlabel('Diferença Absoluta na Ativação Média')
    plt.title('Top 20 Dimensões que Diferenciam\nContratos ACEITOS vs REJEITADOS', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/ativacao_dimensional.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Análise de ativação dimensional salva com sucesso!")
    
except Exception as e:
    import traceback
    print(f"   ⚠️  Erro ao gerar análise de ativação dimensional: {str(e)}")
    print(f"   Stack trace: {traceback.format_exc()[:300]}")
    print("   → Pulando análise dimensional.")

print("\n" + "="*60)
print("✅ INTERPRETABILIDADE CONCLUÍDA!")
print("="*60)
print(f"📂 Todos os artefatos salvos em: {OUTPUT_DIR}")
print("   • feature_importance_global.png")
print("   • shap_summary_classificador.png")
print("   • correlacao_features_classe.png")
print("   • exemplos_decisoes.txt")
print("   • ativacao_dimensional.png")
