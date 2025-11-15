"""
MODELO DE OTIMIZAÇÃO DE LUCRO
===============================
Ao invés de classificação binária (adimplente/inadimplente),
este modelo PREVÊ O LUCRO ESPERADO de cada contrato.

Estratégia:
1. Calcular o lucro mínimo necessário (VaR - Value at Risk)
2. Prever o valor esperado de pagamento (% de parcelas pagas)
3. Calcular lucro esperado = (valor_pago_previsto) - (custo_operacional)
4. Aceitar apenas contratos com lucro esperado > threshold

Isso permite otimizar diretamente o lucro ao invés de apenas classificar.
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 1. CARREGA E PREPARA ----------
def load_and_preprocess_v3(filepath):
    df = pd.read_csv(filepath, sep=';', decimal=',')
    cols_comma = ['valor_inicial_da_prestacao','salario_perc','lucro','IPCA',
                  'Score_MC','idhm_2010','idhm_renda_2010','idhm_longevidade_2010','idhm_educacao_2010',
                  'populacao','area','densidade_pop','preco_combustivel','valor_cestabasica','preco_cb_perc']
    for c in cols_comma:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
            df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ['Data_Lancamento','Data_inicio']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    df['pago_perc'] = pd.to_numeric(df['pago_perc'], errors='coerce')
    df = df.dropna(subset=['pago_perc']).copy()
    df['default'] = (df['pago_perc'] < 1).astype(int)

    df['data_mes']   = df['Data_Lancamento'].dt.month
    df['data_ano']   = df['Data_Lancamento'].dt.year
    df['data_dia']   = df['Data_Lancamento'].dt.dayofweek
    df['categoria_limpa'] = df['categorias'].astype(str).str.strip("[]'").str.split(',').str[0].str.strip("'")
    df['divida_sobre_renda'] = (df['valor_inicial_da_prestacao'] / (df['renda_cliente'] + 1e-6)).clip(upper=100)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

# ---------- 2. ENGENHARIA DE FEATURES FINANCEIRAS ----------
def calcular_metricas_financeiras(df):
    """
    Calcula métricas financeiras detalhadas para cada contrato
    """
    df = df.copy()
    
    # Valor total financiado
    df['valor_total_financiado'] = df['valor_inicial_da_prestacao'] * df['plano_financiamento']
    
    # Custo operacional estimado (assumindo uma margem)
    # Se não tiver informação, assumir ~20% do valor como custo
    df['custo_operacional'] = df['valor_total_financiado'] * 0.20
    
    # Receita esperada = valor_total_financiado + margem de lucro esperada
    # A margem já está embutida no valor_inicial_da_prestacao
    
    # Valor mínimo de parcelas para Break-Even
    # Break-even = custo_operacional / valor_inicial_da_prestacao
    df['parcelas_break_even'] = (df['custo_operacional'] / (df['valor_inicial_da_prestacao'] + 1e-6)).clip(lower=0, upper=df['plano_financiamento'])
    
    # Percentual mínimo para break-even
    df['perc_break_even'] = (df['parcelas_break_even'] / (df['plano_financiamento'] + 1e-6)).clip(upper=1.0)
    
    # VaR: Valor em Risco (máxima perda possível)
    df['var_max_loss'] = -df['custo_operacional']
    
    # Lucro potencial máximo (se pagar 100%)
    df['lucro_potencial_max'] = df['lucro'].where(df['lucro'] > 0, 0)
    
    return df

print("="*70)
print("🎯 MODELO DE OTIMIZAÇÃO DIRETA DE LUCRO")
print("="*70)

# ---------- 3. PREPARAÇÃO DOS DADOS ----------
DATA_FILE = '../data/dataset_interno_top_one_atualizado.csv'
df = load_and_preprocess_v3(DATA_FILE)
df = calcular_metricas_financeiras(df)

print(f"\n📊 Dados carregados: {len(df)} linhas")
print(f"Distribuição da variável alvo 'pago_perc':")
print(df['pago_perc'].describe())

# Variável alvo: PERCENTUAL PAGO (regressão) ao invés de binário
y_pago_perc = df['pago_perc'].values
y_lucro_real = df['lucro'].values

# Features
COLS_REMOVE = ['default','pago_perc','lucro','aceitar',
               'contrato_id','proposta_id','Unnamed: 0',
               'Data_Lancamento','Data_inicio',
               'valor_total_financiado','custo_operacional','parcelas_break_even',
               'perc_break_even','var_max_loss','lucro_potencial_max']

X = df.drop(columns=[c for c in COLS_REMOVE if c in df.columns])

# Manter as métricas financeiras calculadas para usar na predição de lucro
metricas_financeiras = df[['valor_inicial_da_prestacao', 'plano_financiamento', 
                            'valor_total_financiado', 'custo_operacional',
                            'parcelas_break_even', 'perc_break_even']].copy()

print(f"\n🔧 Features utilizadas: {X.shape[1]}")
print(f"Features: {X.columns.tolist()}")

# Identificar features numéricas e categóricas
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

print(f"\nDetectadas {len(numeric_features)} features numéricas.")
print(f"Detectadas {len(categorical_features)} features categóricas.")

X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.median())

# Split
X_train, X_test, y_train_perc, y_test_perc, y_train_lucro, y_test_lucro, metrics_train, metrics_test = train_test_split(
    X, y_pago_perc, y_lucro_real, metricas_financeiras,
    test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Split: {len(X_train)} treino, {len(X_test)} teste")

# ---------- 4. MODELO DE REGRESSÃO: PREVER % PAGO ----------
print("\n" + "="*70)
print("🤖 TREINANDO MODELO DE REGRESSÃO (Previsão de % Pago)")
print("="*70)

dtrain = xgb.DMatrix(X_train_scaled, label=y_train_perc)
dtest = xgb.DMatrix(X_test_scaled, label=y_test_perc)

params_regression = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

model_regression = xgb.train(
    params_regression, 
    dtrain, 
    num_boost_round=500, 
    evals=[(dtest, 'test')],
    early_stopping_rounds=50,
    verbose_eval=False
)

print("✅ Modelo de regressão treinado!")

# ---------- 5. PREVISÕES E CÁLCULO DE LUCRO ESPERADO ----------
y_pred_perc_train = model_regression.predict(xgb.DMatrix(X_train_scaled))
y_pred_perc_test = model_regression.predict(dtest)

# Clipar previsões para [0, 1]
y_pred_perc_train = np.clip(y_pred_perc_train, 0, 1)
y_pred_perc_test = np.clip(y_pred_perc_test, 0, 1)

# Calcular lucro esperado para cada contrato
def calcular_lucro_esperado(y_pred_perc, metrics_df):
    """
    Calcula o lucro esperado baseado no % previsto de pagamento
    
    Lucro Esperado = (Valor Pago Previsto) - (Custo Operacional)
    Valor Pago Previsto = valor_inicial_prestacao * plano_financiamento * perc_previsto
    """
    valor_pago_previsto = (
        metrics_df['valor_inicial_da_prestacao'].values * 
        metrics_df['plano_financiamento'].values * 
        y_pred_perc
    )
    lucro_esperado = valor_pago_previsto - metrics_df['custo_operacional'].values
    return lucro_esperado

lucro_esperado_train = calcular_lucro_esperado(y_pred_perc_train, metrics_train)
lucro_esperado_test = calcular_lucro_esperado(y_pred_perc_test, metrics_test)

# ---------- 6. MÉTRICAS DO MODELO DE REGRESSÃO ----------
print("\n" + "="*70)
print("📊 MÉTRICAS DO MODELO DE REGRESSÃO (% Pago)")
print("="*70)
print(f"MAE (Teste):  {mean_absolute_error(y_test_perc, y_pred_perc_test):.4f}")
print(f"RMSE (Teste): {np.sqrt(mean_squared_error(y_test_perc, y_pred_perc_test)):.4f}")
print(f"R² (Teste):   {r2_score(y_test_perc, y_pred_perc_test):.4f}")

# ---------- 7. OTIMIZAÇÃO DO THRESHOLD DE LUCRO ----------
print("\n" + "="*70)
print("💰 OTIMIZAÇÃO DO THRESHOLD DE LUCRO ESPERADO")
print("="*70)

def otimizar_threshold_lucro(lucro_esperado, lucro_real):
    """
    Encontra o threshold ótimo de lucro esperado para maximizar lucro real
    """
    thresholds = np.percentile(lucro_esperado, range(0, 101, 5))
    
    melhor_threshold = 0
    melhor_lucro_total = -np.inf
    melhor_aceitos = 0
    
    resultados = []
    
    for threshold in thresholds:
        # Aceitar apenas contratos com lucro esperado > threshold
        aceitar = lucro_esperado >= threshold
        
        if aceitar.sum() == 0:
            continue
            
        # Lucro real dos contratos aceitos
        lucro_total = lucro_real[aceitar].sum()
        num_aceitos = aceitar.sum()
        
        resultados.append({
            'threshold': threshold,
            'lucro_total': lucro_total,
            'num_aceitos': num_aceitos,
            'lucro_medio': lucro_total / num_aceitos if num_aceitos > 0 else 0
        })
        
        if lucro_total > melhor_lucro_total:
            melhor_lucro_total = lucro_total
            melhor_threshold = threshold
            melhor_aceitos = num_aceitos
    
    return melhor_threshold, melhor_lucro_total, melhor_aceitos, pd.DataFrame(resultados)

threshold_opt, lucro_opt, num_aceitos_opt, df_resultados = otimizar_threshold_lucro(
    lucro_esperado_test, y_test_lucro
)

print(f"🎯 Threshold Ótimo de Lucro Esperado: R$ {threshold_opt:,.2f}")
print(f"💰 Lucro Total Otimizado: R$ {lucro_opt:,.2f}")
print(f"📊 Contratos Aceitos: {num_aceitos_opt:,} de {len(y_test_lucro):,} ({num_aceitos_opt/len(y_test_lucro)*100:.2f}%)")

# Comparação com cenários
lucro_total_real = y_test_lucro.sum()
lucro_maximo_teorico = y_test_lucro[y_test_lucro > 0].sum()
num_lucrativos = (y_test_lucro > 0).sum()

print("\n" + "="*70)
print("📈 COMPARAÇÃO DE CENÁRIOS")
print("="*70)
print(f"{'Cenário':<40} {'Lucro':>15} {'Eficiência':>12} {'Contratos':>12}")
print("="*70)
print(f"{'1. Aceitar TODOS':<40} R$ {lucro_total_real:>12,.2f} {lucro_total_real/lucro_maximo_teorico:>11.2%} {len(y_test_lucro):>11,}")
print(f"{'2. Máximo Teórico (só lucrativos)':<40} R$ {lucro_maximo_teorico:>12,.2f} {100.0:>11.2%} {num_lucrativos:>11,}")
print(f"{'3. MODELO OTIMIZADO (threshold lucro)':<40} R$ {lucro_opt:>12,.2f} {lucro_opt/lucro_maximo_teorico:>11.2%} {num_aceitos_opt:>11,}")
print("="*70)

ganho_vs_aceitar_todos = lucro_opt - lucro_total_real
print(f"\n💎 Ganho vs Aceitar Todos: R$ {ganho_vs_aceitar_todos:,.2f} ({(lucro_opt/lucro_total_real-1)*100:+.2f}%)")
print(f"🎯 Distância do Máximo Teórico: R$ {lucro_maximo_teorico - lucro_opt:,.2f}")

# ---------- 8. ANÁLISE DETALHADA ----------
aceitar_otimizado = lucro_esperado_test >= threshold_opt

print("\n" + "="*70)
print("🔍 ANÁLISE DETALHADA DOS CONTRATOS ACEITOS")
print("="*70)

df_analise = pd.DataFrame({
    'lucro_real': y_test_lucro,
    'lucro_esperado': lucro_esperado_test,
    'perc_pago_real': y_test_perc,
    'perc_pago_previsto': y_pred_perc_test,
    'aceito': aceitar_otimizado
})

print("\nContratos ACEITOS:")
print(df_analise[df_analise['aceito']].describe())

print("\nContratos REJEITADOS:")
print(df_analise[~df_analise['aceito']].describe())

# Matriz de confusão (lucrativo vs prejuízo)
y_test_lucrativo = (y_test_lucro > 0).astype(int)
y_pred_lucrativo = aceitar_otimizado.astype(int)

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test_lucrativo, y_pred_lucrativo)
print("\n📊 Matriz de Confusão (Lucrativo vs Prejuízo):")
print(pd.DataFrame(
    cm,
    index=['Real: Prejuízo', 'Real: Lucro'],
    columns=['Pred: Rejeitar', 'Pred: Aceitar']
))

# ---------- 9. SALVAR MODELO ----------
os.makedirs('../modelos', exist_ok=True)
model_regression.save_model('../modelos/xgb_lucro_otimizado.json')
print("\n✅ Modelo salvo: ../modelos/xgb_lucro_otimizado.json")

# ---------- 10. GRÁFICOS ----------
print("\n--- Gerando Gráficos ---")
os.makedirs('../graficos/LUCRO_OTIMIZADO', exist_ok=True)

# 1. Lucro Real vs Lucro Esperado
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Scatter: Lucro Real vs Esperado
ax1 = axes[0, 0]
ax1.scatter(lucro_esperado_test, y_test_lucro, alpha=0.5, s=10)
ax1.axhline(0, color='red', linestyle='--', linewidth=1, label='Break-even Real')
ax1.axvline(threshold_opt, color='green', linestyle='--', linewidth=2, label=f'Threshold Ótimo: R$ {threshold_opt:,.0f}')
ax1.set_xlabel('Lucro Esperado (Previsto)', fontsize=12)
ax1.set_ylabel('Lucro Real', fontsize=12)
ax1.set_title('Lucro Real vs Esperado', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Otimização de Threshold
ax2 = axes[0, 1]
ax2.plot(df_resultados['threshold'], df_resultados['lucro_total'], linewidth=2, marker='o')
ax2.axvline(threshold_opt, color='red', linestyle='--', linewidth=2, label='Threshold Ótimo')
ax2.set_xlabel('Threshold de Lucro Esperado', fontsize=12)
ax2.set_ylabel('Lucro Total Real', fontsize=12)
ax2.set_title('Otimização de Threshold', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# % Pago: Real vs Previsto
ax3 = axes[1, 0]
ax3.scatter(y_pred_perc_test, y_test_perc, alpha=0.5, s=10)
ax3.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfeito')
ax3.set_xlabel('% Pago Previsto', fontsize=12)
ax3.set_ylabel('% Pago Real', fontsize=12)
ax3.set_title(f'% Pago: Real vs Previsto (R²={r2_score(y_test_perc, y_pred_perc_test):.3f})', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# Distribuição de Lucro: Aceitos vs Rejeitados
ax4 = axes[1, 1]
ax4.hist(y_test_lucro[aceitar_otimizado], bins=50, alpha=0.7, label='Aceitos', color='green')
ax4.hist(y_test_lucro[~aceitar_otimizado], bins=50, alpha=0.7, label='Rejeitados', color='red')
ax4.axvline(0, color='black', linestyle='--', linewidth=2, label='Break-even')
ax4.set_xlabel('Lucro Real', fontsize=12)
ax4.set_ylabel('Frequência', fontsize=12)
ax4.set_title('Distribuição de Lucro: Aceitos vs Rejeitados', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../graficos/LUCRO_OTIMIZADO/analise_completa.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráficos salvos em graficos/LUCRO_OTIMIZADO/analise_completa.png")

# 2. Feature Importance
try:
    feature_importance = model_regression.get_score(importance_type='weight')
    feature_names = X.columns.tolist()
    
    # Mapear f0, f1, etc para nomes reais
    feature_importance_mapped = {}
    for feat_id, importance in feature_importance.items():
        feat_idx = int(feat_id[1:])  # Remove 'f' e converte para int
        if feat_idx < len(feature_names):
            feature_importance_mapped[feature_names[feat_idx]] = importance
    
    feature_importance_df = pd.DataFrame(
        list(feature_importance_mapped.items()),
        columns=['feature', 'importance']
    ).sort_values(by='importance', ascending=False)
    
    print("\n--- Top 10 Features Mais Importantes ---")
    print(feature_importance_df.head(10))
    
    plt.figure(figsize=(10, 8))
    top_n = min(20, len(feature_importance_df))
    top_features = feature_importance_df.head(top_n)
    plt.barh(range(top_n), top_features['importance'].values)
    plt.yticks(range(top_n), top_features['feature'].values)
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 20 Feature Importances - Modelo de Lucro', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('../graficos/LUCRO_OTIMIZADO/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Feature Importance Plot salvo em graficos/LUCRO_OTIMIZADO/feature_importance.png")
except Exception as e:
    print(f"⚠️ Erro ao gerar Feature Importance: {e}")

print("\n" + "="*70)
print("✅ ANÁLISE COMPLETA!")
print("="*70)
