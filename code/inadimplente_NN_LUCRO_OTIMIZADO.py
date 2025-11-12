"""
REDE NEURAL + AUDITOR LGBM (STACKING) - VERSÃO 6.2
=====================================================
Abordagem em Três Etapas:
1. Rede Neural (Filtro): A NN é treinada com features de alta
   cardinalidade convertidas por Target Encoding.
2. Otimização Bayesiana (Optuna): O Optuna otimiza o LGBM para LUCRO.
3. Auditor LightGBM (Especialista): O LGBM final é treinado.

CORREÇÃO: Corrigido bug de 'NaN' que impedia o treino da NN.
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

# ---------- 1. CARREGA E PREPARA ----------
def load_and_preprocess_v3(filepath):
    df = pd.read_csv(filepath, sep=';', decimal=',')
    cols_comma = ['valor_inicial_da_prestacao','salario_perc','lucro','IPCA',
                  'Score_MC','idhm_2010','idhm_renda_2010','idhm_longevidade_2010','idhm_educacao_2010']
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
print("REDE NEURAL + AUDITOR LGBM OTIMIZADO (V6.2 - CORRIGIDO)")
print("="*70)

# ---------- 2. PREPARAÇÃO DOS DADOS ----------
DATA_FILE = '../data/dataset_interno_top_one.csv'
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
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.2)(x)
    embeddings = layers.Dense(32, activation='relu', name='embeddings')(x) 
    outputs = layers.Dense(1, activation='sigmoid')(embeddings)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
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
    X_val=X_test_scaled, y_val_lucro=y_test_lucro, patience=15, verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_auc', mode='max', factor=0.5, patience=15, min_lr=1e-6, verbose=1
)

# ---------- 6. TREINAMENTO (Etapa 1: Rede Neural) ----------
print("\n" + "="*70)
print("ETAPA 1: INICIANDO TREINAMENTO (REDE NEURAL)")
print("="*70)
history = model.fit(
    X_train_scaled, y_train_target, 
    validation_data=(X_test_scaled, y_test_target),
    epochs=100, batch_size=256,
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

X_train_stacked = pd.concat([X_train_lgbm, df_embeddings_train, pd.Series(y_pred_prob_train_nn, name='nn_prob', index=X_train_lgbm.index)], axis=1)
X_test_stacked = pd.concat([X_test_lgbm, df_embeddings_test, pd.Series(y_pred_prob_test_nn, name='nn_prob', index=X_test_lgbm.index)], axis=1)

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

N_TRIALS = 100 
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

# ---------- 16. GRÁFICOS ----------
print("\n--- Gerando Gráficos ---")
os.makedirs('../graficos/NN_LGBM_AUDITOR_OTIMIZADO', exist_ok=True)
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
plt.savefig('../graficos/NN_LGBM_AUDITOR_OTIMIZADO/historico_treinamento_NN.png', dpi=300)
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
plt.savefig('../graficos/NN_LGBM_AUDITOR_OTIMIZADO/analise_completa.png', dpi=300)
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
plt.savefig('../graficos/NN_LGBM_AUDITOR_OTIMIZADO/confusion_matrix.png', dpi=300)
plt.close()
print("Matriz de confusão (LGBM) salva")

print("\n" + "="*70)
print("ANÁLISE COMPLETA")
print("="*70)