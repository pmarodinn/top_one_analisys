"""
REDE NEURAL + AUDITOR LGBM (STACKING) - VERSÃO 6.4
=====================================================
Abordagem em Três Etapas:
1. Rede Neural (Filtro): A NN é treinada com features de alta
   cardinalidade convertidas por Target Encoding.
2. Otimização Bayesiana (Optuna): O Optuna otimiza o LGBM para LUCRO.
3. Auditor LightGBM (Especialista): O LGBM final é treinado.

NOVA OTIMIZAÇÃO v6.4: Treinamento mais longo da NN
- Épocas: 100→200 (dobrado)
- Early stopping patience: 15→30 (dobrado)  
- ReduceLR patience: 15→25 (demora mais para reduzir LR)
- Objetivo: Permitir NN encontrar mínimo global melhor
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import optuna
import shap
import warnings

# --- Configuração de Ambiente ---
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
optuna.logging.set_verbosity(optuna.logging.WARNING)

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Usando aceleração GPU")
else:
    print("Usando CPU")

# ----------------------
# CONFIGURÁVEIS (experimente esses valores)
# ----------------------
CONFIG = {
    # Rede Neural
    'nn_layers': [512, 256, 128, 64],
    'nn_dropout': [0.30, 0.30, 0.20, 0.20],
    'embedding_dim': 32,
    'use_batchnorm': True,
    'nn_lr': 1e-3,
    'nn_epochs': 200,  # 🔥 AUMENTADO: 100→200 para treinar mais
    'nn_batch_size': 256,
    'early_stop_patience_nn': 30,  # 🔥 AUMENTADO: 15→30 para dar mais chances

    # Optuna / LGBM
    'optuna_trials': 100,
    # Observação: aumentar optuna_trials melhora busca, mas demora mais
}


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
    df = df.dropna(subset=['pago_perc', 'lucro']).copy()
    
    df['default'] = (df['pago_perc'] < 1).astype(int)
    df['data_mes']   = df['Data_Lancamento'].dt.month
    df['data_ano']   = df['Data_Lancamento'].dt.year
    df['data_dia']   = df['Data_Lancamento'].dt.dayofweek
    df['categoria_limpa'] = df['categorias'].astype(str).str.strip("[]'").str.split(',').str[0].str.strip("'")
    df['divida_sobre_renda'] = (df['valor_inicial_da_prestacao'] / (df['renda_cliente'] + 1e-6)).clip(upper=100)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    for col in df.select_dtypes(include=['object', 'datetime64[ns]']).columns:
        if col not in ['Data_Lancamento', 'Data_inicio']:
            df[col] = df[col].astype(str)
            
    return df

print("="*70)
print("REDE NEURAL + AUDITOR LGBM OTIMIZADO (V6.4 - TREINAMENTO LONGO)")
print("="*70)

# ---------- 2. PREPARAÇÃO DOS DADOS ----------
DATA_FILE = '../../data/dataset_interno_top_one_atualizado.csv'
df = load_and_preprocess_v3(DATA_FILE)

print(f"\nDados carregados: {len(df)} linhas")

y_target = (df['lucro'] > 0).astype(int).values
y_lucro_real = df['lucro'].values
y_pago_perc = df['pago_perc'].values 

lucro_maximo_global = df[df['lucro'] > 0]['lucro'].sum()
print(f"Lucro Máximo Teórico Global: R$ {lucro_maximo_global:,.2f}")

COLS_REMOVE = ['default','pago_perc','lucro','aceitar',
               'contrato_id','proposta_id','Unnamed: 0',
               'Data_Lancamento','Data_inicio']
X_original = df.drop(columns=[c for c in COLS_REMOVE if c in df.columns])

numeric_features = X_original.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_original.select_dtypes(include=['object']).columns.tolist()

HI_CARD_COLS = ['Mercadoria', 'descricao_da_Profissao', 'Cidade_Loja', 'municipio', 'categorias']
LOW_CARD_COLS = [col for col in categorical_features if col not in HI_CARD_COLS]

print(f"\nFeatures Numéricas: {len(numeric_features)}")
print(f"Features Categóricas (Baixa Cardinalidade): {len(LOW_CARD_COLS)}")
print(f"Features Categóricas (Alta Cardinalidade/TE): {len(HI_CARD_COLS)}")

# --- 2a. SPLIT PRIMEIRO ---
(X_train_orig, X_test_orig, 
 y_train_target, y_test_target, 
 y_train_lucro, y_test_lucro, 
 y_train_pago_perc, y_test_pago_perc) = train_test_split(
    X_original, y_target, y_lucro_real, y_pago_perc,
    test_size=0.2, random_state=42, stratify=y_target
)

lucro_max_test = y_test_lucro[y_test_lucro > 0].sum()
lucro_aceitar_todos = y_test_lucro.sum()
print(f"\nSplit: {len(X_train_orig)} treino, {len(X_test_orig)} teste")
print(f"Lucro Máximo Teórico (Teste): R$ {lucro_max_test:.2f}")

# --- 2b. PIPELINE DE FEATURES PARA REDE NEURAL (NN) ---
print("\n[NN] Criando features (Numeric + Dummies + Target Encoding)...")

te_mappings = {}
global_mean = y_train_target.mean()

X_train_nn = X_train_orig[numeric_features].copy()
X_test_nn = X_test_orig[numeric_features].copy()

y_train_target_s = pd.Series(y_train_target, index=X_train_orig.index, name='lucrativo')

# 1. Target Encoding para colunas de Alta Cardinalidade
for col in HI_CARD_COLS:
    train_data_with_target = X_train_orig[[col]].join(y_train_target_s)
    mapping = train_data_with_target.groupby(col)['lucrativo'].mean()
    te_mappings[col] = mapping
    X_train_nn[f'{col}_te'] = X_train_orig[col].map(mapping).fillna(global_mean)
    X_test_nn[f'{col}_te'] = X_test_orig[col].map(mapping).fillna(global_mean)

# 2. Dummies para colunas de Baixa Cardinalidade
dummies_train = pd.get_dummies(X_train_orig[LOW_CARD_COLS], drop_first=True, dtype=int)
dummies_test = pd.get_dummies(X_test_orig[LOW_CARD_COLS], drop_first=True, dtype=int)
dummies_train, dummies_test = dummies_train.align(dummies_test, join='left', axis=1, fill_value=0)

# 3. Concatenar tudo para a NN
X_train_nn = pd.concat([X_train_nn, dummies_train], axis=1)
X_test_nn = pd.concat([X_test_nn, dummies_test], axis=1)
X_test_nn = X_test_nn[X_train_nn.columns] # Garantir ordem das colunas
print(f"[NN] Dimensão final das features: {X_train_nn.shape[1]}")


print("[NN] Preenchendo NaNs e aplicando StandardScaler...")

X_train_nn_median = X_train_nn.median()
X_train_nn = X_train_nn.fillna(X_train_nn_median)
X_test_nn = X_test_nn.fillna(X_train_nn_median) 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_nn)
X_test_scaled = scaler.transform(X_test_nn)

# --- 2c. PIPELINE DE FEATURES PARA LGBM ---
X_train_lgbm = X_train_orig.copy()
X_test_lgbm = X_test_orig.copy()
for col in categorical_features:
    X_train_lgbm[col] = X_train_lgbm[col].astype('category')
    X_test_lgbm[col] = X_test_lgbm[col].astype('category')

# --- 2d. Pesos de Classe ---
weights = compute_class_weight('balanced', classes=np.unique(y_train_target), y=y_train_target)
class_weight = {0: weights[0], 1: weights[1]}
print(f"Pesos de Classe: {class_weight}")


# ---------- 3. FUNÇÃO DE OTIMIZAÇÃO ----------
def otimizar_threshold(y_prob, y_lucro, lucro_max):
    thresholds = np.arange(0.0, 1.0, 0.01)
    melhor_threshold = 0.5
    melhor_lucro = -np.inf
    melhor_eficiencia = 0
    
    for threshold in thresholds:
        aceitar = y_prob >= threshold
        if aceitar.sum() == 0:
            continue
        lucro_total = y_lucro[aceitar].sum()
        if lucro_total > melhor_lucro:
            melhor_lucro = lucro_total
            melhor_threshold = threshold
            melhor_eficiencia = lucro_total / lucro_max if lucro_max > 0 else 0
            
    return melhor_threshold, melhor_lucro, melhor_eficiencia, pd.DataFrame() # DF Nulo


# ---------- 4. CONSTRUIR REDE NEURAL (Etapa 1) ----------
print("\n" + "="*70)
print("ETAPA 1: CONSTRUINDO REDE NEURAL")
print("="*70)

def criar_modelo(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    # Construir camadas a partir do CONFIG
    for i, units in enumerate(CONFIG.get('nn_layers', [512, 256, 128, 64])):
        x = layers.Dense(units, activation='relu')(x)
        if CONFIG.get('use_batchnorm', True):
            x = layers.BatchNormalization()(x)
        # escolher dropout correspondente (se fornecido)
        dr = CONFIG.get('nn_dropout', [0.3] * len(CONFIG.get('nn_layers', [])))
        dropout_rate = dr[min(i, len(dr)-1)] if len(dr) > 0 else 0.0
        if dropout_rate and dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

    embeddings = layers.Dense(CONFIG.get('embedding_dim', 32), activation='relu', name='embeddings')(x)
    outputs = layers.Dense(1, activation='sigmoid')(embeddings)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG.get('nn_lr', 1e-3)),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.AUC(name='auc')]
    )
    return model

model = criar_modelo(X_train_scaled.shape[1])
model.summary()

# ---------- 5. CALLBACKS (Etapa 1) ----------
class LucroEarlyStopping(keras.callbacks.Callback):
    def __init__(self, X_val, y_val_lucro, patience=15, verbose=1):
        super().__init__()
        self.X_val = X_val
        self.y_val_lucro = y_val_lucro
        self.patience = patience
        self.verbose = verbose
        self.best_lucro = -np.inf
        self.wait = 0
        self.best_weights = None
        self.lucro_max_val = self.y_val_lucro[self.y_val_lucro > 0].sum()

    def on_epoch_end(self, epoch, logs=None):
        y_prob = self.model.predict(self.X_val, verbose=0).flatten()
        
        # Lidar com NaNs na previsão (caso a rede ainda esteja instável)
        if np.isnan(y_prob).any():
            print(f"\n  Epoca {epoch+1}: Previsão da NN gerou NaN. Ignorando época.")
            lucro_total = -np.inf
        else:
            threshold, lucro_total, _, _ = otimizar_threshold(
                y_prob, self.y_val_lucro, self.lucro_max_val
            )

        if lucro_total > self.best_lucro:
            self.best_lucro = lucro_total
            self.wait = 0
            self.best_weights = self.model.get_weights()
            if self.verbose:
                print(f"\n  Epoca {epoch+1}: Novo melhor lucro: R$ {lucro_total:,.2f} (threshold: {threshold:.3f})")
        else:
            self.wait += 1
            if self.verbose:
                print(f"\n  Epoca {epoch+1}: Sem melhora de lucro ha {self.wait} epoca(s). (Melhor: R$ {self.best_lucro:,.2f})")
        
        if self.wait >= self.patience:
            self.model.stop_training = True
            if self.best_weights is None: # Se nunca melhorou
                 print(f"\n  Parada antecipada na Epoca {epoch+1}. Nenhum modelo bom foi encontrado.")
            else:
                if self.verbose:
                    print(f"\n  Parada antecipada na Epoca {epoch+1}. Restaurando melhor modelo com lucro de R$ {self.best_lucro:,.2f}")
                self.model.set_weights(self.best_weights)

early_stop_lucro = LucroEarlyStopping(
    X_val=X_test_scaled, y_val_lucro=y_test_lucro, patience=CONFIG.get('early_stop_patience_nn', 15), verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_auc', mode='max', factor=0.5, patience=25, min_lr=1e-6, verbose=1
    # 🔥 AUMENTADO: patience 15→25 para demorar mais para reduzir LR
)

# ---------- 6. TREINAMENTO (Etapa 1: Rede Neural) ----------
print("\n" + "="*70)
print("ETAPA 1: INICIANDO TREINAMENTO (REDE NEURAL)")
print("="*70)
history = model.fit(
    X_train_scaled, y_train_target,
    validation_data=(X_test_scaled, y_test_target),
    epochs=CONFIG.get('nn_epochs', 100), batch_size=CONFIG.get('nn_batch_size', 256),
    callbacks=[early_stop_lucro, reduce_lr],
    class_weight=class_weight,
    verbose=1
)
print("\nTreinamento da Rede Neural concluído!")

# ---------- 7. EXTRAÇÃO DE FEATURES (NN) ----------
print("\n" + "="*70)
print("ETAPA 1B: EXTRAINDO EMBEDDINGS E PREVISÕES DA NN")
print("="*70)

feature_extractor = Model(
    inputs=model.inputs,
    outputs=model.get_layer('embeddings').output,
)
nn_embeddings_train = feature_extractor.predict(X_train_scaled, verbose=0)
nn_embeddings_test = feature_extractor.predict(X_test_scaled, verbose=0)
print(f"Shape dos embeddings extraídos (treino): {nn_embeddings_train.shape}")

y_pred_prob_train_nn = model.predict(X_train_scaled, verbose=0).flatten()
y_pred_prob_test_nn = model.predict(X_test_scaled, verbose=0).flatten()

# ---------- 7b. ANÁLISE INTERMEDIÁRIA (NN-ONLY) ----------
print("\nAnalisando desempenho intermediário (NN-Only)...")
threshold_opt_nn, lucro_opt_nn, _, _ = otimizar_threshold(
    y_pred_prob_test_nn, y_test_lucro, lucro_max_test
)
aceitar_nn_only = y_pred_prob_test_nn >= threshold_opt_nn
cm_nn = confusion_matrix(y_test_target, aceitar_nn_only)
try:
    TN_nn, FP_nn, FN_nn, TP_nn = cm_nn.ravel()
    fdr_nn = FP_nn / (FP_nn + TP_nn) if (FP_nn + TP_nn) > 0 else 0
    rejeitados_nn = TN_nn + FN_nn
    ganho_vs_todos_nn = lucro_opt_nn - lucro_aceitar_todos
except ValueError:
    FP_nn, FN_nn, fdr_nn, rejeitados_nn, ganho_vs_todos_nn = 0, 0, 0, 0, 0
print(f"Lucro Otimizado (NN-Only): R$ {lucro_opt_nn:,.2f}")
print(f"Erro em Aceitos (NN-Only): {fdr_nn:.2%}")

# ---------- 8. PREPARAÇÃO DE DADOS (Etapa 2: Auditor) ----------
print("\nPreparando dados para o Auditor (LGBM + Optuna)...")

embedding_cols = [f'embed_{i}' for i in range(nn_embeddings_train.shape[1])]
df_embeddings_train = pd.DataFrame(nn_embeddings_train, columns=embedding_cols, index=X_train_lgbm.index)
df_embeddings_test = pd.DataFrame(nn_embeddings_test, columns=embedding_cols, index=X_test_lgbm.index)

# ⚠️ TESTE: Removendo nn_prob para forçar LGBM a aprender com features originais
print("⚠️  TESTE: nn_prob REMOVIDA - LGBM usará apenas embeddings + features originais")
X_train_stacked = pd.concat([X_train_lgbm, df_embeddings_train], axis=1)
X_test_stacked = pd.concat([X_test_lgbm, df_embeddings_test], axis=1)

# Versão ANTERIOR (com nn_prob):
# X_train_stacked = pd.concat([X_train_lgbm, df_embeddings_train, pd.Series(y_pred_prob_train_nn, name='nn_prob', index=X_train_lgbm.index)], axis=1)
# X_test_stacked = pd.concat([X_test_lgbm, df_embeddings_test, pd.Series(y_pred_prob_test_nn, name='nn_prob', index=X_test_lgbm.index)], axis=1)

X_train_stacked.columns = [str(col) for col in X_train_stacked.columns]
X_test_stacked.columns = [str(col) for col in X_test_stacked.columns]

categorical_features_lgbm_final = categorical_features
print(f"Features do Auditor: {X_train_stacked.shape[1]}")
print(f"Features Categóricas para LGBM: {len(categorical_features_lgbm_final)}")

lgb_train = lgb.Dataset(
    X_train_stacked, 
    label=y_train_target,
    categorical_feature=categorical_features_lgbm_final,
    free_raw_data=False
)
lgb_eval = lgb.Dataset(
    X_test_stacked, 
    label=y_test_target, 
    reference=lgb_train,
    categorical_feature=categorical_features_lgbm_final,
    free_raw_data=False
)

# ---------- 9. OTIMIZAÇÃO (Etapa 2: Optuna) ----------
print("\n" + "="*70)
print("ETAPA 2: OTIMIZAÇÃO DE HIPERPARÂMETROS (OPTUNA)")
print("="*70)

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc', 
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
        'scale_pos_weight': class_weight[0] / class_weight[1],
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 25), 
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }

    lgbm = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_eval],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    y_pred_prob_test_lgbm = lgbm.predict(X_test_stacked, num_iteration=lgbm.best_iteration)
    
    _, lucro_total, _, _ = otimizar_threshold(
        y_pred_prob_test_lgbm, y_test_lucro, lucro_max_test
    )
    
    return lucro_total

N_TRIALS = CONFIG.get('optuna_trials', 100)
print(f"Iniciando estudo do Optuna com {N_TRIALS} tentativas para maximizar o LUCRO...")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("Estudo do Optuna concluído.")
print(f"Melhor Lucro (Teste): R$ {study.best_value:,.2f}")
print("Melhores Hiperparâmetros encontrados:")
print(study.best_params)

# ---------- 10. TREINAMENTO (Etapa 3: Modelo Final) ----------
print("\n" + "="*70)
print("ETAPA 3: TREINANDO AUDITOR FINAL (LGBM Otimizado)")
print("="*70)

best_params = study.best_params
best_params.update({
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 3000,
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
    'scale_pos_weight': class_weight[0] / class_weight[1],
})

print("Treinando o modelo LGBM final com os melhores parâmetros...")
lgbm_final = lgb.train(
    best_params,
    lgb_train,
    valid_sets=[lgb_train, lgb_eval],
    callbacks=[
        lgb.early_stopping(100, verbose=True),
        lgb.log_evaluation(period=200)
    ]
)
print("Treinamento do Auditor Final concluído!")

# ---------- 11. PREVISÕES (Etapa 3: Saída Final) ----------
print("\n" + "="*70)
print("ETAPA 3: GERANDO PREVISÕES (SAÍDA FINAL)")
print("="*70)
y_pred_prob_train = lgbm_final.predict(X_train_stacked, num_iteration=lgbm_final.best_iteration)
y_pred_prob_test = lgbm_final.predict(X_test_stacked, num_iteration=lgbm_final.best_iteration)
print(f"Probabilidade média (Final - LGBM) (treino): {y_pred_prob_train.mean():.4f}")
print(f"Probabilidade média (Final - LGBM) (teste): {y_pred_prob_test.mean():.4f}")

# ---------- 12. OTIMIZAÇÃO DO THRESHOLD (Final) ----------
threshold_opt_lgbm, lucro_opt_lgbm, eficiencia_lgbm, _ = otimizar_threshold(
    y_pred_prob_test, y_test_lucro, lucro_max_test
)
print(f"\nThreshold Ótimo (Final - LGBM): {threshold_opt_lgbm:.4f}")
print(f"Lucro Otimizado (Final - LGBM): R$ {lucro_opt_lgbm:,.2f}")
print(f"Eficiência (Final - LGBM): {eficiencia_lgbm:.2%}")

# ---------- 13. ANÁLISE COMPARATIVA ----------
print("\n" + "="*70)
print("COMPARAÇÃO DE CENÁRIOS (Conjunto de Teste)")
print("="*70)
aceitar_lgbm = y_pred_prob_test >= threshold_opt_lgbm
lucro_lgbm = y_test_lucro[aceitar_lgbm].sum()
num_aceitos_lgbm = aceitar_lgbm.sum()
lucro_lgbm_ponderado = (y_pred_prob_test * y_test_lucro).sum()

print(f"{'Cenário':<45} {'Lucro':>15} {'Eficiência':>12} {'Contratos':>12}")
print("="*85)
print(f"{'1. Aceitar TODOS':<45} R$ {lucro_aceitar_todos:>12,.2f} {lucro_aceitar_todos/lucro_max_test:>11.2%} {len(y_test_lucro):>11,}")
print(f"{'2. Máximo Teórico (oracle)':<45} R$ {lucro_max_test:>12,.2f} {100.0:>11.2%} {(y_test_lucro > 0).sum():>11,}")
print(f"{'3. Modelo (NN+LGBM) com threshold':<45} R$ {lucro_lgbm:>12,.2f} {lucro_lgbm/lucro_max_test:>11.2%} {num_aceitos_lgbm:>11,}")
print(f"{'4. Modelo (NN+LGBM) ponderado':<45} R$ {lucro_lgbm_ponderado:>12,.2f} {lucro_lgbm_ponderado/lucro_max_test:>11.2%} {'N/A':>11}")
print("="*85)
ganho_vs_todos_lgbm = lucro_lgbm - lucro_aceitar_todos
ganho_perc_vs_todos = (lucro_lgbm / lucro_aceitar_todos - 1) * 100 if lucro_aceitar_todos > 0 else float('inf')
print(f"\nGanho vs Aceitar Todos: R$ {ganho_vs_todos_lgbm:,.2f} ({ganho_perc_vs_todos:+.2f}%)")
print(f"Distância do Máximo Teórico: R$ {lucro_max_test - lucro_lgbm:,.2f}")


# ---------- 14. ANÁLISE DETALHADA (Teste / Validação) ----------
print("\n" + "="*70)
print("ANÁLISE DETALHADA (SAÍDA FINAL - LGBM OTIMIZADO)")
print("="*70)
df_analise = pd.DataFrame({
    'lucro_real': y_test_lucro,
    'prob_aceitar': y_pred_prob_test,
    'aceito': aceitar_lgbm,
    'lucrativo': y_test_lucro > 0,
    'pago_perc': y_test_pago_perc
})

print("\n--- Análise de Lucro (Describe) ---")
print("\nContratos ACEITOS:")
print(df_analise[df_analise['aceito']]['lucro_real'].describe())
print("\nContratos REJEITADOS:")
print(df_analise[~df_analise['aceito']]['lucro_real'].describe())

cm_lgbm = confusion_matrix(df_analise['lucrativo'], df_analise['aceito'])
print("\nMatriz de Confusão (Lucrativo vs Aceito):")
print(pd.DataFrame(
    cm_lgbm,
    index=['Real: Prejuízo', 'Real: Lucro'],
    columns=['Pred: Rejeitar', 'Pred: Aceitar']
))

print("\n--- Métricas de Desempenho (Lucro) ---")
try:
    TN_lgbm, FP_lgbm, FN_lgbm, TP_lgbm = cm_lgbm.ravel()
    total_positivos_reais = TP_lgbm + FN_lgbm
    recall_lgbm = TP_lgbm / total_positivos_reais if total_positivos_reais > 0 else 0
    print(f"Acerto em Lucrativos (Recall): {recall_lgbm:.2%}")
    print(f"   (Dos {total_positivos_reais:,} contratos lucrativos, o modelo aceitou {TP_lgbm:,})")
    
    total_aceitos_pelo_modelo = TP_lgbm + FP_lgbm
    fdr_lgbm = FP_lgbm / total_aceitos_pelo_modelo if total_aceitos_pelo_modelo > 0 else 0
    print(f"Erro em Aceitos (False Discovery Rate): {fdr_lgbm:.2%}")
    print(f"   (Dos {total_aceitos_pelo_modelo:,} contratos aceitos, {FP_lgbm:,} eram prejuízo - Falsos Positivos)")
    
    rejeitados_lgbm = TN_lgbm + FN_lgbm
except ValueError:
    fdr_lgbm, rejeitados_lgbm, FP_lgbm, FN_lgbm = 0, 0, 0, 0

print("\n" + "="*70)
print("ANÁLISE DO AUDITOR: LGBM (Otimizado) vs. REDE NEURAL (Sozinha)")
print("="*70)
print("Comparação do desempenho financeiro da Rede Neural (Etapa 1) vs. o Auditor LGBM (Otimizado)")
print(f"\n{'Métrica':<35} {'Rede Neural (Etapa 1)':>25} {'Auditor LGBM (Final)':>25}")
print("-"*87)
print(f"{'Lucro Otimizado':<35} R$ {lucro_opt_nn:>24,.2f} R$ {lucro_opt_lgbm:>24,.2f}")
print(f"{'Ganho vs Aceitar Todos':<35} R$ {ganho_vs_todos_nn:>24,.2f} R$ {ganho_vs_todos_lgbm:>24,.2f}")
print(f"{'Erro em Aceitos (FDR)':<35} {fdr_nn:>24.2%} {fdr_lgbm:>24.2%}")
print(f"{'Falsos Positivos (Prejuízo Aceito)':<35} {FP_nn:>25,} {FP_lgbm:>25,}")
print(f"{'Falsos Negativos (Lucro Rejeitado)':<35} {FN_nn:>25,} {FN_lgbm:>25,}")
print(f"{'Total Rejeitados':<35} {rejeitados_nn:>25,} {rejeitados_lgbm:>25,}")

print("\n--- Análise de Pagamento (Total no Teste) ---")
inadimplentes_totais = (df_analise['pago_perc'] < 1.0).sum()
print(f"Total de Inadimplentes (<100% pago): {inadimplentes_totais:,}")

print("\n--- Análise de Pagamento (Contratos ACEITOS pelo LGBM) ---")
df_aceitos = df_analise[df_analise['aceito'] == True]
total_aceitos = len(df_aceitos)
if total_aceitos > 0:
    adimplentes_totais_aceitos = (df_aceitos['pago_perc'] >= 1.0).sum()
    inadimplentes_lucrativos_aceitos = ((df_aceitos['pago_perc'] < 1.0) & (df_analise['lucrativo'] == True)).sum()
    inadimplentes_prejuizo_aceitos = ((df_aceitos['pago_perc'] < 1.0) & (df_analise['lucrativo'] == False)).sum()
    print(f"Total de Aceitos: {total_aceitos:,}")
    print(f"  - 1. Adimplentes Totais (pagaram 100%): {adimplentes_totais_aceitos:,} ({adimplentes_totais_aceitos/total_aceitos:.2%})")
    print(f"  - 2. Inadimplentes Lucrativos ('Bom Risco'): {inadimplentes_lucrativos_aceitos:,} ({inadimplentes_lucrativos_aceitos/total_aceitos:.2%})")
    print(f"  - 3. Inadimplentes Prejuízo ('Mau Risco'): {inadimplentes_prejuizo_aceitos:,} ({inadimplentes_prejuizo_aceitos/total_aceitos:.2%})")
else:
    print("Nenhum contrato aceito.")

# ---------- 15. SALVAR MODELO ----------
model.save('../modelos/nn_filtro_etapa1.h5')
print("\nModelo NN (Etapa 1) salvo: ../modelos/nn_filtro_etapa1.h5")
lgbm_final.save_model('../modelos/lgbm_auditor_etapa2_otimizado.txt')
print("Modelo LGBM (Etapa 2 Otimizado) salvo: ../modelos/lgbm_auditor_etapa2_otimizado.txt")

# ---------- 15B. ANÁLISE DE INTERPRETABILIDADE (SHAP) ----------
print("\n" + "="*70)
print("ANÁLISE DE INTERPRETABILIDADE (SHAP VALUES)")
print("="*70)

print("Calculando SHAP values para o conjunto de teste...")
# Criar TreeExplainer para LightGBM
explainer = shap.TreeExplainer(lgbm_final)

# Calcular SHAP values (amostra de 1000 contratos para velocidade)
sample_size = min(1000, len(X_test_stacked))
X_test_sample = X_test_stacked.sample(n=sample_size, random_state=42)
shap_values = explainer.shap_values(X_test_sample)

# Para classificação binária, pegar apenas SHAP values da classe 1 (Lucrativo)
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
else:
    shap_values_class1 = shap_values

print(f"SHAP values calculados para {sample_size} contratos")

# Feature Importance Global (Mean Absolute SHAP)
shap_importance = pd.DataFrame({
    'feature': X_test_sample.columns,
    'importance': np.abs(shap_values_class1).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\n--- TOP 20 FEATURES MAIS IMPORTANTES (SHAP) ---")
print(shap_importance.head(20).to_string(index=False))

# Análise de Casos Específicos: Falsos Positivos e Falsos Negativos
print("\n--- ANÁLISE DE ERROS DO MODELO ---")

# Identificar FP e FN no sample
# Converter índices do DataFrame para posições no array
sample_positions = [list(X_test_stacked.index).index(idx) for idx in X_test_sample.index]
y_test_sample_target = y_test_target[sample_positions]
y_test_sample_lucro = y_test_lucro[sample_positions]
y_pred_sample = lgbm_final.predict(X_test_sample, num_iteration=lgbm_final.best_iteration)
aceitar_sample = y_pred_sample >= threshold_opt_lgbm

# Falsos Positivos: Modelo aceitou (pred=1), mas era prejuízo (real=0)
fp_mask = (aceitar_sample == True) & (y_test_sample_target == 0)
fn_mask = (aceitar_sample == False) & (y_test_sample_target == 1)

print(f"\nFalsos Positivos (Prejuízo aceito): {fp_mask.sum()} contratos")
print(f"Falsos Negativos (Lucro rejeitado): {fn_mask.sum()} contratos")

# Salvar gráficos SHAP
os.makedirs('../../graficos/analise_modelos/NN_LGBM_AUDITOR_OTIMIZADO/shap', exist_ok=True)

print("\n--- Gerando Gráficos SHAP ---")

# 1. Summary Plot (Importância Global + Distribuição)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_class1, X_test_sample, plot_type="dot", show=False, max_display=20)
plt.title('SHAP Summary Plot - Importância e Impacto das Features', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../../graficos/analise_modelos/NN_LGBM_AUDITOR_OTIMIZADO/shap/summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Summary Plot salvo")

# 2. Bar Plot (Importância Média)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_class1, X_test_sample, plot_type="bar", show=False, max_display=20)
plt.title('SHAP Feature Importance - Top 20 Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../../graficos/analise_modelos/NN_LGBM_AUDITOR_OTIMIZADO/shap/feature_importance_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Bar Plot salvo")

# 3. Waterfall Plot - Exemplo de Contrato Aceito Corretamente (TP)
tp_mask = (aceitar_sample == True) & (y_test_sample_target == 1)  # Aceito E Lucrativo
if tp_mask.sum() > 0:
    tp_idx = np.where(tp_mask)[0][0]  # Primeiro True Positive
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_class1[tp_idx], 
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_test_sample.iloc[tp_idx],
            feature_names=X_test_sample.columns.tolist()
        ),
        show=False,
        max_display=15
    )
    plt.title(f'Waterfall Plot - Contrato Aceito CORRETAMENTE (Lucro Real: R$ {y_test_sample_lucro[tp_idx]:.2f})', 
              fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('../../graficos/analise_modelos/NN_LGBM_AUDITOR_OTIMIZADO/shap/waterfall_true_positive.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Waterfall Plot (True Positive) salvo")

# 4. Waterfall Plot - Exemplo de Falso Positivo (Prejuízo aceito)
if fp_mask.sum() > 0:
    fp_idx = np.where(fp_mask)[0][0]  # Primeiro Falso Positivo
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_class1[fp_idx], 
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_test_sample.iloc[fp_idx],
            feature_names=X_test_sample.columns.tolist()
        ),
        show=False,
        max_display=15
    )
    plt.title(f'Waterfall Plot - FALSO POSITIVO (Prejuízo: R$ {y_test_sample_lucro[fp_idx]:.2f} - Modelo ERROU)', 
              fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('../../graficos/analise_modelos/NN_LGBM_AUDITOR_OTIMIZADO/shap/waterfall_false_positive.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Waterfall Plot (False Positive) salvo")

# 5. Waterfall Plot - Exemplo de Falso Negativo (Lucro rejeitado)
if fn_mask.sum() > 0:
    fn_idx = np.where(fn_mask)[0][0]  # Primeiro Falso Negativo
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_class1[fn_idx], 
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_test_sample.iloc[fn_idx],
            feature_names=X_test_sample.columns.tolist()
        ),
        show=False,
        max_display=15
    )
    plt.title(f'Waterfall Plot - FALSO NEGATIVO (Lucro: R$ {y_test_sample_lucro[fn_idx]:.2f} - Modelo PERDEU)', 
              fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('../../graficos/analise_modelos/NN_LGBM_AUDITOR_OTIMIZADO/shap/waterfall_false_negative.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Waterfall Plot (False Negative) salvo")

# 6. Dependence Plots - Top 3 Features
top_3_features = shap_importance.head(3)['feature'].tolist()
for i, feat in enumerate(top_3_features, 1):
    try:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feat, 
            shap_values_class1, 
            X_test_sample, 
            show=False,
            interaction_index='auto'
        )
        plt.title(f'Dependence Plot - {feat}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'../../graficos/analise_modelos/NN_LGBM_AUDITOR_OTIMIZADO/shap/dependence_plot_{i}_{feat[:30]}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Dependence Plot {i} ({feat}) salvo")
    except Exception as e:
        print(f"⚠️  Erro ao gerar Dependence Plot para {feat}: {e}")

# 7. Análise Comparativa: FP vs TP (Média de SHAP values)
if fp_mask.sum() > 0 and tp_mask.sum() > 0:
    shap_fp_mean = np.abs(shap_values_class1[fp_mask]).mean(axis=0)
    shap_tp_mean = np.abs(shap_values_class1[tp_mask]).mean(axis=0)
    
    comparison_df = pd.DataFrame({
        'feature': X_test_sample.columns,
        'FP_mean_shap': shap_fp_mean,
        'TP_mean_shap': shap_tp_mean,
        'diff': shap_fp_mean - shap_tp_mean
    }).sort_values('diff', key=abs, ascending=False).head(15)
    
    print("\n--- DIFERENÇA DE IMPORTÂNCIA: Falsos Positivos vs True Positives ---")
    print(comparison_df.to_string(index=False))
    
    # Gráfico de comparação
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(comparison_df))
    width = 0.35
    ax.bar(x - width/2, comparison_df['FP_mean_shap'], width, label='Falsos Positivos (Erros)', color='red', alpha=0.7)
    ax.bar(x + width/2, comparison_df['TP_mean_shap'], width, label='True Positives (Acertos)', color='green', alpha=0.7)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Mean |SHAP|', fontsize=12)
    ax.set_title('Comparação SHAP: Erros (FP) vs Acertos (TP)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['feature'], rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('../../graficos/analise_modelos/NN_LGBM_AUDITOR_OTIMIZADO/shap/comparison_fp_vs_tp.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Comparação FP vs TP salva")

# 8. ANÁLISE DE EMBEDDINGS: Correlação com Features Originais
print("\n--- Analisando Embeddings (Correlação com Features Originais) ---")

# Pegar os top 5 embeddings mais importantes
top_embeddings = [feat for feat in shap_importance.head(10)['feature'].tolist() if feat.startswith('embed_')][:5]

# Criar DataFrame com embeddings de treino
embed_cols = [f'embed_{i}' for i in range(nn_embeddings_train.shape[1])]
df_embeddings_full = pd.DataFrame(nn_embeddings_train, columns=embed_cols, index=X_train_nn.index)

# Calcular correlação de cada embedding com features originais
embedding_correlations = {}
for embed in top_embeddings:
    embed_idx = int(embed.split('_')[1])
    correlations = X_train_nn.corrwith(df_embeddings_full[embed])
    top_corr = correlations.abs().sort_values(ascending=False).head(10)
    embedding_correlations[embed] = top_corr
    
    print(f"\n{embed.upper()} - Top 10 Correlações com Features Originais:")
    for feat, corr in top_corr.items():
        print(f"  {feat:40s} {corr:+.4f}")

# Gráfico de correlação para top 3 embeddings
fig, axes = plt.subplots(len(top_embeddings[:3]), 1, figsize=(12, 4*len(top_embeddings[:3])))
if len(top_embeddings[:3]) == 1:
    axes = [axes]

for idx, embed in enumerate(top_embeddings[:3]):
    ax = axes[idx]
    top_corr = embedding_correlations[embed].head(15)
    colors = ['red' if x < 0 else 'green' for x in top_corr.values]
    ax.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_corr)))
    ax.set_yticklabels(top_corr.index, fontsize=10)
    ax.set_xlabel('Correlação', fontsize=11)
    ax.set_title(f'{embed.upper()} - Correlação com Features Originais (Top 15)', fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('../../graficos/analise_modelos/NN_LGBM_AUDITOR_OTIMIZADO/shap/embedding_correlations.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Análise de correlação dos embeddings salva")

# 9. VISUALIZAÇÃO t-SNE: Embeddings em 2D
print("\n--- Gerando Visualização t-SNE dos Embeddings ---")

from sklearn.manifold import TSNE

# Usar amostra menor para t-SNE (mais rápido)
tsne_sample_size = min(500, len(nn_embeddings_test))
tsne_indices = np.random.choice(len(nn_embeddings_test), tsne_sample_size, replace=False)
embeddings_for_tsne = nn_embeddings_test[tsne_indices]
y_lucro_for_tsne = y_test_lucro[tsne_indices]
y_target_for_tsne = y_test_target[tsne_indices]

print(f"Executando t-SNE em {tsne_sample_size} contratos (pode demorar ~30s)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings_for_tsne)

# Criar 3 visualizações diferentes
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1. Colorido por Lucro Real (contínuo)
ax1 = axes[0]
scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                       c=y_lucro_for_tsne, cmap='RdYlGn', alpha=0.6, s=50, 
                       vmin=-2000, vmax=2000)
ax1.set_xlabel('t-SNE Dimensão 1', fontsize=12)
ax1.set_ylabel('t-SNE Dimensão 2', fontsize=12)
ax1.set_title('Embeddings em 2D - Colorido por Lucro Real', fontsize=14, fontweight='bold')
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Lucro (R$)', fontsize=11)
ax1.grid(alpha=0.3)

# 2. Colorido por Classe (Lucrativo vs Prejuízo)
ax2 = axes[1]
colors_class = ['red' if x == 0 else 'green' for x in y_target_for_tsne]
ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
           c=colors_class, alpha=0.6, s=50)
ax2.set_xlabel('t-SNE Dimensão 1', fontsize=12)
ax2.set_ylabel('t-SNE Dimensão 2', fontsize=12)
ax2.set_title('Embeddings em 2D - Lucrativo (Verde) vs Prejuízo (Vermelho)', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)

# 3. Colorido por Magnitude do Lucro/Prejuízo
ax3 = axes[2]
magnitude_lucro = np.abs(y_lucro_for_tsne)
scatter3 = ax3.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                       c=magnitude_lucro, cmap='plasma', alpha=0.6, s=50)
ax3.set_xlabel('t-SNE Dimensão 1', fontsize=12)
ax3.set_ylabel('t-SNE Dimensão 2', fontsize=12)
ax3.set_title('Embeddings em 2D - Magnitude do Lucro/Prejuízo', fontsize=14, fontweight='bold')
cbar3 = plt.colorbar(scatter3, ax=ax3)
cbar3.set_label('|Lucro| (R$)', fontsize=11)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../../graficos/analise_modelos/NN_LGBM_AUDITOR_OTIMIZADO/shap/tsne_embeddings_2d.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Visualização t-SNE salva")

# Análise de clusters (identificar regiões de alto/baixo lucro)
from scipy.stats import gaussian_kde

# Densidade de lucro positivo vs negativo
lucrativos_mask_tsne = y_target_for_tsne == 1
prejuizo_mask_tsne = y_target_for_tsne == 0

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Densidade de contratos lucrativos
ax1 = axes[0]
if lucrativos_mask_tsne.sum() > 10:
    xy_lucrativos = embeddings_2d[lucrativos_mask_tsne].T
    try:
        kde_lucrativos = gaussian_kde(xy_lucrativos)
        x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
        y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density_lucrativos = np.reshape(kde_lucrativos(positions).T, xx.shape)
        ax1.contourf(xx, yy, density_lucrativos, cmap='Greens', alpha=0.6, levels=10)
        ax1.scatter(embeddings_2d[lucrativos_mask_tsne, 0], embeddings_2d[lucrativos_mask_tsne, 1], 
                   c='green', alpha=0.3, s=20, label='Lucrativos')
    except:
        ax1.scatter(embeddings_2d[lucrativos_mask_tsne, 0], embeddings_2d[lucrativos_mask_tsne, 1], 
                   c='green', alpha=0.5, s=30)
ax1.set_xlabel('t-SNE Dimensão 1', fontsize=12)
ax1.set_ylabel('t-SNE Dimensão 2', fontsize=12)
ax1.set_title('Densidade de Contratos LUCRATIVOS', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)

# Densidade de contratos prejuízo
ax2 = axes[1]
if prejuizo_mask_tsne.sum() > 10:
    xy_prejuizo = embeddings_2d[prejuizo_mask_tsne].T
    try:
        kde_prejuizo = gaussian_kde(xy_prejuizo)
        density_prejuizo = np.reshape(kde_prejuizo(positions).T, xx.shape)
        ax2.contourf(xx, yy, density_prejuizo, cmap='Reds', alpha=0.6, levels=10)
        ax2.scatter(embeddings_2d[prejuizo_mask_tsne, 0], embeddings_2d[prejuizo_mask_tsne, 1], 
                   c='red', alpha=0.3, s=20, label='Prejuízo')
    except:
        ax2.scatter(embeddings_2d[prejuizo_mask_tsne, 0], embeddings_2d[prejuizo_mask_tsne, 1], 
                   c='red', alpha=0.5, s=30)
ax2.set_xlabel('t-SNE Dimensão 1', fontsize=12)
ax2.set_ylabel('t-SNE Dimensão 2', fontsize=12)
ax2.set_title('Densidade de Contratos PREJUÍZO', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../../graficos/analise_modelos/NN_LGBM_AUDITOR_OTIMIZADO/shap/tsne_density_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Análise de densidade t-SNE salva")

# Salvar relatório de interpretabilidade
with open('../../graficos/analise_modelos/NN_LGBM_AUDITOR_OTIMIZADO/shap/interpretability_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("RELATÓRIO DE INTERPRETABILIDADE - SHAP VALUES\n")
    f.write("="*70 + "\n\n")
    f.write(f"Data: {pd.Timestamp.now()}\n")
    f.write(f"Amostra analisada: {sample_size} contratos\n\n")
    
    f.write("--- TOP 20 FEATURES MAIS IMPORTANTES ---\n")
    f.write(shap_importance.head(20).to_string(index=False))
    f.write("\n\n")
    
    f.write("--- ANÁLISE DE ERROS ---\n")
    f.write(f"Falsos Positivos: {fp_mask.sum()} ({fp_mask.sum()/len(X_test_sample)*100:.2f}%)\n")
    f.write(f"Falsos Negativos: {fn_mask.sum()} ({fn_mask.sum()/len(X_test_sample)*100:.2f}%)\n")
    f.write(f"True Positives: {tp_mask.sum()} ({tp_mask.sum()/len(X_test_sample)*100:.2f}%)\n\n")
    
    if fp_mask.sum() > 0 and tp_mask.sum() > 0:
        f.write("--- DIFERENÇA DE IMPORTÂNCIA: FP vs TP ---\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
    
    f.write("="*70 + "\n")
    f.write("ANÁLISE DOS EMBEDDINGS\n")
    f.write("="*70 + "\n\n")
    f.write("Os embeddings são representações de 32 dimensões aprendidas pela\n")
    f.write("Rede Neural. Eles capturam padrões complexos das 99 features originais.\n\n")
    
    for embed in top_embeddings[:3]:
        f.write(f"\n--- {embed.upper()} (Importância SHAP: {shap_importance[shap_importance['feature']==embed]['importance'].values[0]:.6f}) ---\n")
        f.write("Top 10 Correlações com Features Originais:\n")
        for feat, corr in embedding_correlations[embed].head(10).items():
            f.write(f"  {feat:40s} {corr:+.4f}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("INTERPRETAÇÃO t-SNE\n")
    f.write("="*70 + "\n\n")
    f.write(f"t-SNE projeta os 32 embeddings em 2D para visualização.\n")
    f.write(f"Amostra analisada: {tsne_sample_size} contratos\n\n")
    f.write("Observações:\n")
    f.write("- Clusters verdes = regiões de contratos lucrativos\n")
    f.write("- Clusters vermelhos = regiões de contratos com prejuízo\n")
    f.write("- Separação clara indica que a NN aprendeu bem os padrões\n")
    f.write("- Sobreposição indica casos ambíguos (difíceis de classificar)\n")

print("\n✓ Relatório de interpretabilidade salvo em: ../graficos/NN_LGBM_AUDITOR_OTIMIZADO/shap/interpretability_report.txt")
print(f"✓ Total de gráficos SHAP salvos: {6 + len(top_3_features) + 3}")
print("  - Summary plots (2)")
print("  - Waterfall plots (3: TP, FP, FN)")
print(f"  - Dependence plots ({len(top_3_features)})")
print("  - Comparação FP vs TP (1)")
print("  - Correlação de embeddings (1)")
print("  - t-SNE visualizações (2)")

# ---------- 16. GRÁFICOS (Tradicionais) ----------
print("\n--- Gerando Gráficos ---")
os.makedirs('../../graficos/analise_modelos/NN_LGBM_AUDITOR_OTIMIZADO', exist_ok=True)
# 1. Histórico de treinamento (da Rede Neural)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax1 = axes[0]
ax1.plot(history.history['loss'], label='Treino', linewidth=2)
ax1.plot(history.history['val_loss'], label='Validação', linewidth=2)
ax1.set_xlabel('Época', fontsize=12)
ax1.set_ylabel('Loss (Binary Crossentropy)', fontsize=12)
ax1.set_title('Histórico de Loss (Rede Neural)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)
ax2 = axes[1]
ax2.plot(history.history['auc'], label='Treino', linewidth=2)
ax2.plot(history.history['val_auc'], label='Validação', linewidth=2)
ax2.set_xlabel('Época', fontsize=12)
ax2.set_ylabel('AUC (Area Under Curve)', fontsize=12)
ax2.set_title('Evolução da AUC (Rede Neural)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../../graficos/analise_modelos/NN_LGBM_AUDITOR_OTIMIZADO/historico_treinamento_NN.png', dpi=300)
plt.close()
print("Histórico de treinamento da NN salvo")

# 2. Otimização de threshold (do LGBM Otimizado)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
thresholds_list = np.arange(0.0, 1.0, 0.01)
lucros_list = []
eficiencias_list = []
for t in thresholds_list:
    aceitar = y_pred_prob_test >= t
    lucro_total = y_test_lucro[aceitar].sum()
    lucros_list.append(lucro_total)
    eficiencias_list.append(lucro_total / lucro_max_test if lucro_max_test > 0 else 0)
df_thresholds = pd.DataFrame({'threshold': thresholds_list, 'lucro_total': lucros_list, 'eficiencia': eficiencias_list})
ax1 = axes[0, 0]
ax1.plot(df_thresholds['threshold'], df_thresholds['lucro_total'], linewidth=2)
ax1.axvline(threshold_opt_lgbm, color='red', linestyle='--', linewidth=2, label=f'Ótimo: {threshold_opt_lgbm:.3f}')
ax1.set_xlabel('Threshold de Probabilidade (LGBM)', fontsize=12)
ax1.set_ylabel('Lucro Total', fontsize=12)
ax1.set_title('Otimização de Threshold (LGBM)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)
ax2 = axes[0, 1]
ax2.plot(df_thresholds['threshold'], df_thresholds['eficiencia']*100, linewidth=2)
ax2.axvline(threshold_opt_lgbm, color='red', linestyle='--', linewidth=2, label=f'Ótimo: {threshold_opt_lgbm:.3f}')
ax2.set_xlabel('Threshold de Probabilidade (LGBM)', fontsize=12)
ax2.set_ylabel('Eficiência (%)', fontsize=12)
ax2.set_title('Eficiência vs Threshold (LGBM)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)
ax3 = axes[1, 0]
lucrativos_mask = df_analise['lucrativo'] == True
prejuizo_mask = df_analise['lucrativo'] == False
if lucrativos_mask.sum() > 0:
    ax3.hist(df_analise.loc[lucrativos_mask, 'prob_aceitar'], bins=50, alpha=0.7, label='Lucrativos', color='green')
if prejuizo_mask.sum() > 1:
    try:
        ax3.hist(df_analise.loc[prejuizo_mask, 'prob_aceitar'], bins=50, alpha=0.7, label='Prejuízo', color='red')
    except ValueError:
        n_bins = min(10, prejuizo_mask.sum())
        if n_bins > 1:
            ax3.hist(df_analise.loc[prejuizo_mask, 'prob_aceitar'], bins=n_bins, alpha=0.7, label='Prejuízo', color='red')
ax3.axvline(threshold_opt_lgbm, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold_opt_lgbm:.3f}')
ax3.set_xlabel('Probabilidade (de ser Lucrativo - LGBM)', fontsize=12)
ax3.set_ylabel('Frequência', fontsize=12)
ax3.set_title('Distribuição de Probabilidades (LGBM)', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)
ax4 = axes[1, 1]
scatter = ax4.scatter(df_analise['prob_aceitar'], df_analise['lucro_real'], 
                     c=df_analise['aceito'], cmap='RdYlGn', alpha=0.6, s=10)
ax4.axhline(0, color='black', linestyle='--', linewidth=1)
ax4.axvline(threshold_opt_lgbm, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold_opt_lgbm:.3f}')
ax4.set_xlabel('Probabilidade (de ser Lucrativo - LGBM)', fontsize=12)
ax4.set_ylabel('Lucro Real', fontsize=12)
ax4.set_title('Probabilidade (LGBM) vs Lucro Real', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax4, label='Aceito (1=Sim, 0=Não)')
ax4.legend()
ax4.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../../graficos/analise_modelos/NN_LGBM_AUDITOR_OTIMIZADO/analise_completa.png', dpi=300)
plt.close()
print("Análise completa (LGBM) salva")

# 3. Matriz de confusão visual
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_lgbm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Rejeitar', 'Aceitar'],
            yticklabels=['Prejuízo', 'Lucro'], ax=ax)
ax.set_xlabel('Decisão do Modelo (LGBM)', fontsize=12)
ax.set_ylabel('Realidade', fontsize=12)
ax.set_title('Matriz de Confusão - Lucro vs Decisão (LGBM)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../../graficos/analise_modelos/NN_LGBM_AUDITOR_OTIMIZADO/confusion_matrix.png', dpi=300)
plt.close()
print("Matriz de confusão (LGBM) salva")

print("\n" + "="*70)
print("ANÁLISE COMPLETA")
print("="*70)