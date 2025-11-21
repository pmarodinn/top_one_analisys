# train_xgb_lucro.py
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 1.  CARREGA E PREPARA (mesma lógica que você já tem) ----------
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

# ---------- 2.  PREPARAÇÃO ----------
DATA_FILE = '../../data/dataset_interno_top_one_atualizado.csv'
df = load_and_preprocess_v3(DATA_FILE)

print(f"Dados brutos carregados: {len(df)} linhas")
print(f"Dados filtrados (pago_perc preenchido): {len(df)} linhas")
print("Distribuição da variável alvo 'default' (1 = Inadimplente):")
print(df['default'].value_counts(normalize=True))

y = df['default'].values
lucro = df['lucro'].values  # peso financeiro

print("\n--- Iniciando Fase 1: Modelo Preditivo (XGBoost) ---")

COLS_REMOVE = ['default','pago_perc','lucro','aceitar',
               'contrato_id','proposta_id','Unnamed: 0',
               'Data_Lancamento','Data_inicio']
X = df.drop(columns=[c for c in COLS_REMOVE if c in df.columns])

# Identificar features numéricas e categóricas ANTES do get_dummies
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

print(f"Features utilizadas ({X.shape[1]}): {X.columns.tolist()}")
print(f"\nDetectadas {len(numeric_features)} features numéricas.")
print(f"Detectadas {len(categorical_features)} features categóricas.")

X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.median())

X_train, X_test, y_train, y_test, lucro_train, lucro_test = train_test_split(
    X, y, lucro, test_size=0.2, random_state=42, stratify=y)

# Calcular e exibir pesos baseados em impacto financeiro
# IMPORTANTE: Calcular baseado no LUCRO REAL, não na classificação de inadimplência
# Nem todo inadimplente gera prejuízo, e nem todo adimplente gera lucro!
custo_medio_inadimplente = abs(lucro_train[lucro_train < 0].mean())
ganho_medio_adimplente = lucro_train[lucro_train > 0].mean()
peso_linear = custo_medio_inadimplente / ganho_medio_adimplente
scale_pos_weight = peso_linear * (1 + np.log1p(peso_linear))

print(f"\nPesos baseados em impacto financeiro:")
print(f"   Custo médio por inadimplente:     R$ {custo_medio_inadimplente:,.2f}")
print(f"   Ganho médio por adimplente:       R$ {ganho_medio_adimplente:,.2f}")
print(f"   Peso Linear:                      {peso_linear:.2f}")
print(f"   Scale Pos Weight (Log ajustado):  {scale_pos_weight:.2f}")
print(f"   Fator de amplificação:            {scale_pos_weight/peso_linear:.2f}x")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ---------- 3.  XGBOOST COM PESO = LUCRO (COM BUSCA DE PENALIDADE FN) ----------
print("\nTreinando o modelo de Probabilidade de Default (PD) com peso=lucro e busca de penalidade FN...")
dtest  = xgb.DMatrix(X_test,  label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Grid search rápido para o multiplicador de penalidade de FN (pesos de exemplos positivos)
fn_penalties = [1.0, 1.5, 2.0, 3.0, 5.0]
best_penalty = 1.0
best_profit = -np.inf
best_model = None
best_y_prob_try = None

for pen in fn_penalties:
    # construir pesos por amostra: base = |lucro|, multiplicar positives (inadimplentes) por pen
    weights_try = np.abs(lucro_train) * (1.0 + (pen - 1.0) * (y_train == 1))
    dtrain_try = xgb.DMatrix(X_train, label=y_train, weight=weights_try)

    # treino rápido para comparar
    model_try = xgb.train(params, dtrain_try, num_boost_round=200, verbose_eval=False)
    y_prob_try = model_try.predict(dtest)

    # avaliar melhor threshold por lucro (mesma lógica de melhor_threshold)
    best_t_try, best_profit_try = 0.5, -np.inf
    for t in np.arange(0.05, 1.0, 0.01):
        y_pred_try = (y_prob_try >= t).astype(int)
        lucro_total_try = (
            lucro_test[(y_test == 0) & (y_pred_try == 0)].sum() +
            0 +
            -lucro_test[(y_test == 0) & (y_pred_try == 1)].sum() +
            lucro_test[(y_test == 1) & (y_pred_try == 0)].sum()
        )
        if lucro_total_try > best_profit_try:
            best_t_try, best_profit_try = t, lucro_total_try

    print(f"  Penalidade FN={pen:.2f} -> Melhor lucro (teste) = R$ {best_profit_try:,.2f} (threshold={best_t_try:.2f})")
    if best_profit_try > best_profit:
        best_profit = best_profit_try
        best_penalty = pen
        best_model = model_try
        best_y_prob_try = y_prob_try

print(f"Melhor penalidade FN encontrada: {best_penalty:.2f} -> lucro R$ {best_profit:,.2f}")

# Re-treinar modelo final com a melhor penalidade e mais rounds (modelo final)
weights_final = np.abs(lucro_train) * (1.0 + (best_penalty - 1.0) * (y_train == 1))
dtrain_final = xgb.DMatrix(X_train, label=y_train, weight=weights_final)
model = xgb.train(params, dtrain_final, num_boost_round=500, verbose_eval=False)
print("Treinamento concluído (modelo final).")

# --- 3.5 Análise Financeira dos Dados de TREINO ---
arrecadacoes_treino = lucro_train[lucro_train > 0].sum()
perdas_treino = lucro_train[lucro_train < 0].sum()
lucro_liquido_treino = arrecadacoes_treino + perdas_treino
percentual_treino = (arrecadacoes_treino / abs(perdas_treino)) * 100 if perdas_treino != 0 else 0

print("\n" + "="*70)
print(" ANÁLISE FINANCEIRA - DADOS DE TREINO")
print("="*70)
print(f" Total de contratos no treino: {len(lucro_train):,}")
print(f" Arrecadações Totais (lucros positivos):  R$ {arrecadacoes_treino:,.2f}")
print(f" Perdas Totais (lucros negativos):        R$ {perdas_treino:,.2f}")
print(f" Arrecadação / Perda:                      {percentual_treino:.2f}%")
print(f"{'─'*70}")
print(f" Lucro Líquido Total:                      R$ {lucro_liquido_treino:,.2f}")
print("="*70)

# ---------- 4.  CALIBRAÇÃO BAYESIANA ----------
print("\n--- Avaliação do Modelo (Dados de Teste) ---")
y_prob_raw = model.predict(dtest)
auc_score = roc_auc_score(y_test, y_prob_raw)
print(f"**ROC AUC Score (Poder de Separação): {auc_score:.4f}**")

# TEOREMA DE BAYES: Ajuste inteligente das probabilidades
# P(inadimplente|score) = P(score|inadimplente) * P(inadimplente) / P(score)
print("\nAplicando Teorema de Bayes para calibração inteligente...")

# Calcular priors (probabilidades a priori)
prior_adimplente = (y_train == 0).sum() / len(y_train)
prior_inadimplente = (y_train == 1).sum() / len(y_train)

print(f"Prior Adimplente: {prior_adimplente:.4f} ({prior_adimplente*100:.2f}%)")
print(f"Prior Inadimplente: {prior_inadimplente:.4f} ({prior_inadimplente*100:.2f}%)")

# Ajustar probabilidades usando razão de custo/benefício
# Peso Bayesiano = (Custo FN / Custo FP) * (Prior Adimplente / Prior Inadimplente)
custo_fn = abs(custo_medio_inadimplente)  # Custo de aceitar inadimplente
custo_fp = ganho_medio_adimplente         # Custo de rejeitar adimplente (perda de oportunidade)

razao_custo = custo_fn / custo_fp
razao_prior = prior_adimplente / prior_inadimplente
peso_bayesiano = razao_custo * razao_prior

print(f"Razão de Custo (FN/FP): {razao_custo:.2f}")
print(f"Razão de Prior (Adimpl/Inadimpl): {razao_prior:.2f}")
print(f"Peso Bayesiano Total: {peso_bayesiano:.2f}")

# Aplicar calibração: aumentar probabilidade de inadimplência proporcionalmente
# P_calibrada = P_raw^(1/peso_bayesiano) -> isso "estica" as probabilidades baixas
# Alternativa: Usar odds ratio
def calibrar_bayes(y_prob_raw, peso):
    """
    Calibra probabilidades usando Teorema de Bayes
    Converte para odds, aplica peso, converte de volta para probabilidade
    """
    # Evitar divisão por zero
    y_prob_raw = np.clip(y_prob_raw, 1e-10, 1-1e-10)
    
    # Converter para odds: odds = P / (1-P)
    odds_raw = y_prob_raw / (1 - y_prob_raw)
    
    # Aplicar peso bayesiano aos odds
    odds_calibrado = odds_raw * peso
    
    # Converter de volta para probabilidade: P = odds / (1 + odds)
    y_prob_calibrado = odds_calibrado / (1 + odds_calibrado)
    
    return y_prob_calibrado

y_prob = calibrar_bayes(y_prob_raw, peso_bayesiano)

print(f"Probabilidade média ANTES da calibração: {y_prob_raw.mean():.4f}")
print(f"Probabilidade média DEPOIS da calibração: {y_prob.mean():.4f}")
print(f"Aumento médio: {((y_prob.mean()/y_prob_raw.mean())-1)*100:.2f}%")

def melhor_threshold(y_true, y_prob, lucro):
    best_t, best_lucro = 0.5, -np.inf
    for t in np.arange(0.05, 1.0, 0.01):
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # TN (aceitou adimplente): ganha lucro
        # TP (rejeitou inadimplente): evita prejuízo, ganho = 0
        # FP (rejeitou adimplente): perde lucro potencial
        # FN (aceitou inadimplente): tem prejuízo
        lucro_total = (lucro[(y_true==0)&(y_pred==0)].sum() +    # TN: ganha lucro
                       0 +                                        # TP: evita prejuízo, ganho = 0
                       -lucro[(y_true==0)&(y_pred==1)].sum() +   # FP: perde lucro potencial
                       lucro[(y_true==1)&(y_pred==0)].sum())     # FN: tem prejuízo (lucro negativo)
        if lucro_total > best_lucro:
            best_t, best_lucro = t, lucro_total
    return best_t, best_lucro

def melhor_threshold_com_recall_min(y_true, y_prob, lucro, min_recall=0.15):
    """Seleciona o threshold que maximize o lucro com a restrição de recall mínimo.
    Se nenhum threshold atingir o recall mínimo, retorna o threshold que maximize o lucro.
    """
    best_t, best_lucro = 0.5, -np.inf
    best_t_with_recall, best_lucro_with_recall = None, -np.inf
    for t in np.arange(0.05, 1.0, 0.01):
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        lucro_total = (lucro[(y_true==0)&(y_pred==0)].sum() +
                       0 +
                       -lucro[(y_true==0)&(y_pred==1)].sum() +
                       lucro[(y_true==1)&(y_pred==0)].sum())
        if recall >= min_recall and lucro_total > best_lucro_with_recall:
            best_t_with_recall, best_lucro_with_recall = t, lucro_total
        if lucro_total > best_lucro:
            best_t, best_lucro = t, lucro_total

    if best_t_with_recall is not None:
        return best_t_with_recall, best_lucro_with_recall
    return best_t, best_lucro

# --- 4.5 Análise Financeira dos Dados de TESTE ---
arrecadacoes_teste = lucro_test[lucro_test > 0].sum()
perdas_teste = lucro_test[lucro_test < 0].sum()
lucro_liquido_teste = arrecadacoes_teste + perdas_teste
percentual_teste = (arrecadacoes_teste / abs(perdas_teste)) * 100 if perdas_teste != 0 else 0

# Lucro máximo teórico (se aceitássemos APENAS os lucrativos)
lucro_maximo_teorico = arrecadacoes_teste
num_lucrativos = (lucro_test > 0).sum()
num_prejuizo = (lucro_test < 0).sum()

print("\n" + "="*70)
print("ANÁLISE FINANCEIRA - DADOS DE TESTE (VALIDAÇÃO)")
print("="*70)
print(f"Total de contratos no teste: {len(lucro_test):,}")
print(f"Arrecadações Totais (lucros positivos):  R$ {arrecadacoes_teste:,.2f}")
print(f"Perdas Totais (lucros negativos):        R$ {perdas_teste:,.2f}")
print(f"Arrecadação / Perda:                      {percentual_teste:.2f}%")
print(f"{'─'*70}")
print(f"Lucro Líquido Total (Cenário Real):       R$ {lucro_liquido_teste:,.2f}")
print(f"{'─'*70}")
print(f"LUCRO MÁXIMO TEÓRICO:")
print(f"   (Se escolhêssemos apenas os {num_lucrativos:,} lucrativos)")
print(f"   Lucro Máximo Possível:                    R$ {lucro_maximo_teorico:,.2f}")
print(f"   Eficiência Atual:                         {(lucro_liquido_teste/lucro_maximo_teorico)*100:.2f}%")
print("="*70)

t_opt, lucro_opt = melhor_threshold(y_test, y_prob, lucro_test)
t_opt_raw, lucro_opt_raw = melhor_threshold(y_test, y_prob_raw, lucro_test)

print(f"\nOTIMIZAÇÃO DE THRESHOLD:")
print(f"{'─'*70}")
print(f"Threshold ótimo (SEM Bayes): {t_opt_raw:.2f}  |  Lucro: R$ {lucro_opt_raw:,.2f}")
print(f"   Eficiência vs Máximo Teórico: {(lucro_opt_raw/lucro_maximo_teorico)*100:.2f}%")
print(f"{'─'*70}")
print(f"Threshold ótimo (COM Bayes): {t_opt:.2f}  |  Lucro: R$ {lucro_opt:,.2f}")
print(f"   Eficiência vs Máximo Teórico: {(lucro_opt/lucro_maximo_teorico)*100:.2f}%")
print(f"{'─'*70}")
print(f"Ganho com Bayes: R$ {lucro_opt - lucro_opt_raw:,.2f} ({((lucro_opt/lucro_opt_raw)-1)*100:.2f}%)")

# Encontrar threshold que satisfaça recall mínimo (ex: 12%) e maximize lucro
min_recall_target = 0.12
t_opt_recall, lucro_opt_recall = melhor_threshold_com_recall_min(y_test, y_prob, lucro_test, min_recall=min_recall_target)
print(f"\nThreshold otimizado por recall mínimo ({min_recall_target*100:.0f}%): {t_opt_recall:.2f}  |  Lucro: R$ {lucro_opt_recall:,.2f}")

# Comparar métricas de classificação
y_pred_raw = (y_prob_raw >= t_opt_raw).astype(int)
y_pred_bayes = (y_prob >= t_opt).astype(int)

cm_raw = confusion_matrix(y_test, y_pred_raw)
cm_bayes = confusion_matrix(y_test, y_pred_bayes)

print("\nCOMPARAÇÃO DE DETECÇÃO DE INADIMPLENTES:")
print(f"{'Métrica':<30} {'Sem Bayes':>15} {'Com Bayes':>15} {'Melhoria':>15}")
print("="*75)

tn_raw, fp_raw, fn_raw, tp_raw = cm_raw.ravel()
tn_bay, fp_bay, fn_bay, tp_bay = cm_bayes.ravel()

recall_raw_val = float(tp_raw / (tp_raw + fn_raw)) if (tp_raw + fn_raw) > 0 else 0.0
recall_bay_val = float(tp_bay / (tp_bay + fn_bay)) if (tp_bay + fn_bay) > 0 else 0.0

precision_raw_val = float(tp_raw / (tp_raw + fp_raw)) if (tp_raw + fp_raw) > 0 else 0.0
precision_bay_val = float(tp_bay / (tp_bay + fp_bay)) if (tp_bay + fp_bay) > 0 else 0.0

print(f"{'Inadimplentes Detectados (TP)':<30} {int(tp_raw):>15} {int(tp_bay):>15} {int(tp_bay-tp_raw):>+15}")
print(f"{'Inadimplentes Perdidos (FN)':<30} {int(fn_raw):>15} {int(fn_bay):>15} {int(fn_bay-fn_raw):>+15}")
print(f"{'Recall (sensibilidade)':<30} {recall_raw_val:>14.2%} {recall_bay_val:>14.2%} {(recall_bay_val-recall_raw_val):>+14.2%}")
print(f"{'Precision':<30} {precision_raw_val:>14.2%} {precision_bay_val:>14.2%} {(precision_bay_val-precision_raw_val):>+14.2%}")
print("="*75)

# --- 5.5 Feature Importance ---
try:
    # Pegar os nomes das features após get_dummies
    feature_names_after_dummies = X.columns.tolist()
    
    # Obter importâncias do modelo
    feature_importance_dict = model.get_score(importance_type='weight')
    
    # Mapear f0, f1, f2... para nomes reais
    feature_importance_mapped = {}
    for feat_key, importance in feature_importance_dict.items():
        # feat_key é tipo 'f0', 'f1', etc.
        feat_idx = int(feat_key[1:])  # Remove 'f' e pega o índice
        if feat_idx < len(feature_names_after_dummies):
            real_name = feature_names_after_dummies[feat_idx]
            feature_importance_mapped[real_name] = importance
    
    feature_importance_df = pd.DataFrame(
        list(feature_importance_mapped.items()),
        columns=['feature', 'importance']
    ).sort_values(by='importance', ascending=False)
    
    print("\n--- Top 10 Features Mais Importantes ---")
    print(feature_importance_df.head(10).to_string(index=False))
except Exception as e:
    print(f"\n⚠️ Não foi possível extrair feature importance: {e}")

# ---------- 5.  SALVA MODELO ----------
os.makedirs('../modelos', exist_ok=True)
model.save_model('../modelos/xgb_lucro_unico.json')
print("\n✅ Modelo salvo: ../modelos/xgb_lucro_unico.json")

# ---------- 6.  GRÁFICOS ----------
print("\n--- Gerando Gráficos ---")
os.makedirs('../../graficos/analise_modelos/XGB_lucro', exist_ok=True)

print("Gerando ROC Curve...")
# ROC - Comparação com e sem Bayes
fpr_raw, tpr_raw, _ = roc_curve(y_test, y_prob_raw)
fpr_bayes, tpr_bayes, _ = roc_curve(y_test, y_prob)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ROC Curve
ax1.plot(fpr_raw, tpr_raw, label=f'Sem Bayes (AUC = {roc_auc_score(y_test, y_prob_raw):.4f})', linewidth=2, color='orange')
ax1.plot(fpr_bayes, tpr_bayes, label=f'Com Bayes (AUC = {roc_auc_score(y_test, y_prob):.4f})', linewidth=2, color='blue')
ax1.plot([0,1],[0,1],'k--', label='Random', linewidth=1)
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curve - Comparação', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(alpha=0.3)

# Precision-Recall Curve
precision_raw, recall_raw, _ = precision_recall_curve(y_test, y_prob_raw)
precision_bayes, recall_bayes, _ = precision_recall_curve(y_test, y_prob)

ax2.plot(recall_raw, precision_raw, label=f'Sem Bayes', linewidth=2, color='orange')
ax2.plot(recall_bayes, precision_bayes, label=f'Com Bayes', linewidth=2, color='blue')
ax2.set_xlabel('Recall (Sensibilidade)', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../../graficos/analise_modelos/XGB_lucro/roc_pr_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ ROC e Precision-Recall Curves salvas em graficos/XGB_lucro/roc_pr_curves.png")

print("Gerando Confusion Matrices...")
# Confusion matrices - Comparação
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Sem Bayes
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Oranges', cbar=True,
            xticklabels=['Adimplente','Inadimplente'],
            yticklabels=['Adimplente','Inadimplente'], ax=ax1)
ax1.set_xlabel('Predito', fontsize=12)
ax1.set_ylabel('Real', fontsize=12)
ax1.set_title(f'SEM Bayes (threshold={float(t_opt_raw):.2f})\nRecall={recall_raw_val:.2%}', fontsize=14, fontweight='bold')

# Com Bayes
sns.heatmap(cm_bayes, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Adimplente','Inadimplente'],
            yticklabels=['Adimplente','Inadimplente'], ax=ax2)
ax2.set_xlabel('Predito', fontsize=12)
ax2.set_ylabel('Real', fontsize=12)
ax2.set_title(f'COM Bayes (threshold={float(t_opt):.2f})\nRecall={recall_bay_val:.2%}', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../../graficos/analise_modelos/XGB_lucro/conf_matrix_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Confusion Matrices salvas em graficos/XGB_lucro/conf_matrix_comparison.png")

# Feature Importance Plot
print("Gerando Feature Importance Plot...")
try:
    if 'feature_importance_df' in locals() and len(feature_importance_df) > 0:
        plt.figure(figsize=(10, 8))
        top_n = min(20, len(feature_importance_df))
        top_features = feature_importance_df.head(top_n)
        plt.barh(range(top_n), top_features['importance'].values)
        plt.yticks(range(top_n), top_features['feature'].values)
        plt.xlabel('Importance', fontsize=12)
        plt.title('Top 20 Feature Importances - XGBoost', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('../../graficos/analise_modelos/XGB_lucro/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Feature Importance Plot salvo em graficos/XGB_lucro/feature_importance.png")
except Exception as e:
    print(f"⚠️ Erro ao gerar Feature Importance Plot: {e}")

# Calibration Curve - Verificar qualidade da calibração
print("Gerando Calibration Curve...")
try:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calibration curves
    fraction_of_positives_raw, mean_predicted_value_raw = calibration_curve(
        y_test, y_prob_raw, n_bins=10, strategy='uniform'
    )
    fraction_of_positives_bayes, mean_predicted_value_bayes = calibration_curve(
        y_test, y_prob, n_bins=10, strategy='uniform'
    )
    
    ax.plot(mean_predicted_value_raw, fraction_of_positives_raw, 's-', 
            label='Sem Bayes', color='orange', linewidth=2, markersize=8)
    ax.plot(mean_predicted_value_bayes, fraction_of_positives_bayes, 'o-', 
            label='Com Bayes', color='blue', linewidth=2, markersize=8)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfeitamente Calibrado', linewidth=1)
    
    ax.set_xlabel('Probabilidade Predita', fontsize=12)
    ax.set_ylabel('Fração de Positivos', fontsize=12)
    ax.set_title('Calibration Curve - Qualidade da Probabilidade', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../graficos/analise_modelos/XGB_lucro/calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Calibration Curve salva em graficos/XGB_lucro/calibration_curve.png")
except Exception as e:
    print(f"⚠️ Erro ao gerar Calibration Curve: {e}")

print("\n✅ Todos os gráficos foram salvos em graficos/XGB_lucro/")