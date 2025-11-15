"""
REDE NEURAL DUPLA + AUDITOR LGBM (STACKING) - VERSÃO 7.1
=========================================================
🎯 ESTRATÉGIA APRIMORADA: Foco em Inadimplentes Lucrativos

1. Redes Neurais Especialistas:
   - NN A (Adimplência): Prevê Adimplente (100%) vs Inadimplente (<100%)
   - NN B (Lucro-Inadimplente): Prevê LUCRO apenas entre Inadimplentes
     (Identifica os "inadimplentes lucrativos" - o ouro escondido 💎)

2. Optuna: Otimização Bayesiana do LGBM para LUCRO.

3. Auditor LGBM (Fusor): Combina embeddings (64D) + probabilities + features
   para decisão final integrando as duas visões complementares.
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
print("REDE NEURAL DUPLA + AUDITOR LGBM OTIMIZADO (V7.0 - Dual Stacking)")
print("="*70)

# ---------- 2. PREPARAÇÃO DOS DADOS ----------
DATA_FILE = '../data/dataset_interno_top_one_atualizado.csv'
df = load_and_preprocess_v3(DATA_FILE)

print(f"\nDados carregados: {len(df)} linhas")

# --- DEFINIÇÃO DOS TARGETS ---
# Target LGBM (Lucro Final) - Usado pelo Auditor LGBM
y_target_lucro = (df['lucro'] > 0).astype(int).values
y_lucro_real = df['lucro'].values
y_pago_perc = df['pago_perc'].values 

# Target A (Adimplência) - Usado pela NN-A
# 1 = Adimplente (pagou 100%), 0 = Inadimplente (pagou <100%)
y_target_adimplencia = (df['pago_perc'] >= 1.0).astype(int).values

# Target B (Lucro em Inadimplentes) - Usado pela NN-B
# Apenas para contratos INADIMPLENTES, identifica os lucrativos
# Estratégia: Isolar os "inadimplentes lucrativos" (ouro escondido 💎)
inadimplentes_mask = (df['pago_perc'] < 1.0)
y_target_lucro_inadimplente = np.zeros(len(df), dtype=int)
# Marca como 1 os inadimplentes que geraram lucro
y_target_lucro_inadimplente[inadimplentes_mask & (df['lucro'] > 0)] = 1

lucro_maximo_global = df[df['lucro'] > 0]['lucro'].sum()
print(f"Lucro Máximo Teórico Global: R$ {lucro_maximo_global:,.2f}")

# 🔍 ANÁLISE DOS TARGETS
print("\n" + "="*70)
print("ANÁLISE DOS TARGETS (V7.1 - Inadimplentes Lucrativos)")
print("="*70)

# Distribuição Target A (Adimplência)
print(f"\nDistribuição Target A (Adimplência):")
unique_adimplencia, counts_adimplencia = np.unique(y_target_adimplencia, return_counts=True)
for val, count in zip(unique_adimplencia, counts_adimplencia):
    label = "Inadimplente" if val == 0 else "Adimplente"
    print(f"  - Classe {val} ({label}): {count:,} ({count/len(df)*100:.2f}%)")

# Distribuição Target B (Lucro em Inadimplentes)
num_inadimplentes = inadimplentes_mask.sum()
num_inadimplentes_lucrativos = (inadimplentes_mask & (df['lucro'] > 0)).sum()
num_inadimplentes_prejuizo = num_inadimplentes - num_inadimplentes_lucrativos

print(f"\nDistribuição Target B (Lucro em Inadimplentes):")
print(f"  - Total Inadimplentes: {num_inadimplentes:,}")
print(f"    → Inadimplentes Lucrativos (Target=1): {num_inadimplentes_lucrativos:,} ({num_inadimplentes_lucrativos/num_inadimplentes*100:.2f}%)")
print(f"    → Inadimplentes Prejuízo (Target=0): {num_inadimplentes_prejuizo:,} ({num_inadimplentes_prejuizo/num_inadimplentes*100:.2f}%)")

# Distribuição Target Final (Lucro Geral)
print(f"\nDistribuição Target Final (Lucro Geral - LGBM):")
unique_lucro, counts_lucro = np.unique(y_target_lucro, return_counts=True)
for val, count in zip(unique_lucro, counts_lucro):
    label = "Prejuízo" if val == 0 else "Lucrativo"
    print(f"  - Classe {val} ({label}): {count:,} ({count/len(df)*100:.2f}%)")

# Tabela cruzada: Adimplência vs Lucro
crosstab = pd.crosstab(
    pd.Series(y_target_adimplencia, name='Adimplência'),
    pd.Series(y_target_lucro, name='Lucro (Lucrativo)'),
    margins=True
)
print(f"\nTabela Cruzada Adimplência vs Lucro:")
print(crosstab)
print("="*70)
print("="*70)

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
 y_train_lucro_target, y_test_lucro_target,  # Para Auditor LGBM (target final)
 y_train_adimplencia_target, y_test_adimplencia_target,  # Para NN-A
 y_train_lucro_inadimplente_target, y_test_lucro_inadimplente_target,  # Para NN-B
 y_train_lucro, y_test_lucro, 
 y_train_pago_perc, y_test_pago_perc) = train_test_split(
    X_original, y_target_lucro, y_target_adimplencia, y_target_lucro_inadimplente, y_lucro_real, y_pago_perc,
    test_size=0.2, random_state=42, stratify=y_target_lucro
)

lucro_max_test = y_test_lucro[y_test_lucro > 0].sum()
lucro_aceitar_todos = y_test_lucro.sum()
print(f"\nSplit: {len(X_train_orig)} treino, {len(X_test_orig)} teste")
print(f"Lucro Máximo Teórico (Teste): R$ {lucro_max_test:.2f}")

# --- 2b. PIPELINE DE FEATURES PARA REDE NEURAL (NN) ---
print("\n[NN] Criando features (Numeric + Dummies + Target Encoding)...")

# Target Encoding será usado APENAS para o Especialista B (Lucro)
# O Especialista A (Risco) usará o mesmo TE para simplicidade.
te_mappings = {}
global_mean = y_train_lucro_target.mean()

X_train_nn = X_train_orig[numeric_features].copy()
X_test_nn = X_test_orig[numeric_features].copy()

y_train_lucro_target_s = pd.Series(y_train_lucro_target, index=X_train_orig.index, name='lucrativo')

# 1. Target Encoding (baseado no target de LUCRO)
for col in HI_CARD_COLS:
    train_data_with_target = X_train_orig[[col]].join(y_train_lucro_target_s)
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
# Pesos para Auditor LGBM (Lucro Final)
weights_lucro = compute_class_weight('balanced', classes=np.unique(y_train_lucro_target), y=y_train_lucro_target)
class_weight_lucro = {0: weights_lucro[0], 1: weights_lucro[1]}
print(f"Pesos de Classe (Lucro Final - LGBM): {class_weight_lucro}")

# Pesos para Especialista A (Adimplência)
weights_adimplencia = compute_class_weight('balanced', classes=np.unique(y_train_adimplencia_target), y=y_train_adimplencia_target)
class_weight_adimplencia = {0: weights_adimplencia[0], 1: weights_adimplencia[1]}
print(f"Pesos de Classe (Adimplência): {class_weight_adimplencia}")

# Pesos para Especialista B (Lucro em Inadimplentes)
# Importante: Calcular apenas sobre inadimplentes
inadimplentes_train_mask = (y_train_pago_perc < 1.0)
y_train_lucro_inadimplente_only = y_train_lucro_inadimplente_target[inadimplentes_train_mask]
weights_lucro_inadimplente = compute_class_weight('balanced', classes=np.unique(y_train_lucro_inadimplente_only), y=y_train_lucro_inadimplente_only)
class_weight_lucro_inadimplente = {0: weights_lucro_inadimplente[0], 1: weights_lucro_inadimplente[1]}
print(f"Pesos de Classe (Lucro-Inadimplente): {class_weight_lucro_inadimplente}")
print(f"  - Calculado sobre {inadimplentes_train_mask.sum():,} inadimplentes do treino")


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


# ---------- 4. ARQUITETURA DA REDE NEURAL (Compartilhada) ----------
def criar_modelo(input_dim, model_name):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.2)(x)
    # Damos nomes únicos às camadas de embedding
    embeddings = layers.Dense(32, activation='relu', name=f'{model_name}_embeddings')(x) 
    outputs = layers.Dense(1, activation='sigmoid')(embeddings)
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.AUC(name='auc')]
    )
    return model

# ---------- 5. CALLBACKS (Especialista B - Lucro) ----------
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
            if self.best_weights is None:
                 print(f"\n  Parada antecipada na Epoca {epoch+1}. Nenhum modelo bom foi encontrado.")
            else:
                if self.verbose:
                    print(f"\n  Parada antecipada na Epoca {epoch+1}. Restaurando melhor modelo com lucro de R$ {self.best_lucro:,.2f}")
                self.model.set_weights(self.best_weights)

# ---------- 6. TREINAMENTO (Etapa 1: Especialistas) ----------
input_dim = X_train_scaled.shape[1]

# --- 6a. Treinar Especialista A (Adimplência) ---
print("\n" + "="*70)
print("ETAPA 1A: TREINANDO ESPECIALISTA A (ADIMPLÊNCIA)")
print("="*70)
model_adimplencia = criar_modelo(input_dim, model_name='NN_Adimplencia')
model_adimplencia.summary()

early_stop_adimplencia = callbacks.EarlyStopping(
    monitor='val_auc', mode='max', patience=15, restore_best_weights=True
)
reduce_lr_adimplencia = callbacks.ReduceLROnPlateau(
    monitor='val_auc', mode='max', factor=0.5, patience=15, min_lr=1e-6, verbose=1
)

model_adimplencia.fit(
    X_train_scaled, y_train_adimplencia_target, 
    validation_data=(X_test_scaled, y_test_adimplencia_target),
    epochs=100, batch_size=256,
    callbacks=[early_stop_adimplencia, reduce_lr_adimplencia],
    class_weight=class_weight_adimplencia,
    verbose=1
)
print("\nTreinamento do Especialista A (Adimplência) concluído!")

# --- 6b. Treinar Especialista B (Lucro em Inadimplentes) ---
print("\n" + "="*70)
print("ETAPA 1B: TREINANDO ESPECIALISTA B (LUCRO EM INADIMPLENTES)")
print("="*70)
print(f"📊 Treinando APENAS com {inadimplentes_train_mask.sum():,} inadimplentes...")

model_lucro_inadimplente = criar_modelo(input_dim, model_name='NN_Lucro_Inadimplente')
model_lucro_inadimplente.summary()

# Treinar APENAS com dados de inadimplentes
X_train_inadimplentes = X_train_scaled[inadimplentes_train_mask]
y_train_lucro_inadim_only = y_train_lucro_inadimplente_target[inadimplentes_train_mask]

# Para validação, também usar apenas inadimplentes
inadimplentes_test_mask = (y_test_pago_perc < 1.0)
X_test_inadimplentes = X_test_scaled[inadimplentes_test_mask]
y_test_lucro_inadim_only = y_test_lucro_inadimplente_target[inadimplentes_test_mask]

print(f"   Treino: {len(X_train_inadimplentes):,} inadimplentes")
print(f"   Validação: {len(X_test_inadimplentes):,} inadimplentes")

early_stop_lucro_inadim = callbacks.EarlyStopping(
    monitor='val_auc', mode='max', patience=15, restore_best_weights=True
)
reduce_lr_lucro_inadim = callbacks.ReduceLROnPlateau(
    monitor='val_auc', mode='max', factor=0.5, patience=15, min_lr=1e-6, verbose=1
)

model_lucro_inadimplente.fit(
    X_train_inadimplentes, y_train_lucro_inadim_only, 
    validation_data=(X_test_inadimplentes, y_test_lucro_inadim_only),
    epochs=100, batch_size=256,
    callbacks=[early_stop_lucro_inadim, reduce_lr_lucro_inadim],
    class_weight=class_weight_lucro_inadimplente,
    verbose=1
)
print("\nTreinamento do Especialista B (Lucro-Inadimplente) concluído!")

# ---------- 7. EXTRAÇÃO DE FEATURES (NNs) ----------
print("\n" + "="*70)
print("ETAPA 1C: EXTRAINDO EMBEDDINGS E PREVISÕES DAS NNs")
print("="*70)

# --- Extrair de Especialista A (Adimplência) ---
feature_extractor_adimplencia = Model(
    inputs=model_adimplencia.inputs,
    outputs=model_adimplencia.get_layer('NN_Adimplencia_embeddings').output,
)
nn_embeddings_train_adimplencia = feature_extractor_adimplencia.predict(X_train_scaled, verbose=0)
nn_embeddings_test_adimplencia = feature_extractor_adimplencia.predict(X_test_scaled, verbose=0)
y_pred_prob_train_adimplencia = model_adimplencia.predict(X_train_scaled, verbose=0).flatten()
y_pred_prob_test_adimplencia = model_adimplencia.predict(X_test_scaled, verbose=0).flatten()

print(f"Shape dos embeddings (Adimplência): {nn_embeddings_train_adimplencia.shape}")

# --- Extrair de Especialista B (Lucro-Inadimplente) ---
feature_extractor_lucro_inadim = Model(
    inputs=model_lucro_inadimplente.inputs,
    outputs=model_lucro_inadimplente.get_layer('NN_Lucro_Inadimplente_embeddings').output,
)
nn_embeddings_train_lucro_inadim = feature_extractor_lucro_inadim.predict(X_train_scaled, verbose=0)
nn_embeddings_test_lucro_inadim = feature_extractor_lucro_inadim.predict(X_test_scaled, verbose=0)
y_pred_prob_train_lucro_inadim = model_lucro_inadimplente.predict(X_train_scaled, verbose=0).flatten()
y_pred_prob_test_lucro_inadim = model_lucro_inadimplente.predict(X_test_scaled, verbose=0).flatten()

print(f"Shape dos embeddings (Lucro-Inadimplente): {nn_embeddings_train_lucro_inadim.shape}")

# ---------- 7b. ANÁLISE INTERMEDIÁRIA ----------
print("\n⚠️  Nota: NN-B foi treinada APENAS com inadimplentes, não tem sentido isolada.")
print("   A decisão final será feita pelo LGBM combinando as duas visões.")

# ---------- 8. PREPARAÇÃO DE DADOS (Etapa 2: Auditor Fusor) ----------
print("\nPreparando dados para o Auditor (LGBM + Optuna)...")

# DataFrames para embeddings de Adimplência
cols_embed_adimplencia = [f'embed_adimplencia_{i}' for i in range(nn_embeddings_train_adimplencia.shape[1])]
df_embed_train_adimplencia = pd.DataFrame(nn_embeddings_train_adimplencia, columns=cols_embed_adimplencia, index=X_train_lgbm.index)
df_embed_test_adimplencia = pd.DataFrame(nn_embeddings_test_adimplencia, columns=cols_embed_adimplencia, index=X_test_lgbm.index)

# DataFrames para embeddings de Lucro-Inadimplente
cols_embed_lucro_inadim = [f'embed_lucro_inadim_{i}' for i in range(nn_embeddings_train_lucro_inadim.shape[1])]
df_embed_train_lucro_inadim = pd.DataFrame(nn_embeddings_train_lucro_inadim, columns=cols_embed_lucro_inadim, index=X_train_lgbm.index)
df_embed_test_lucro_inadim = pd.DataFrame(nn_embeddings_test_lucro_inadim, columns=cols_embed_lucro_inadim, index=X_test_lgbm.index)

# Concatenar TUDO para o LGBM Fusor
X_train_stacked = pd.concat([
    X_train_lgbm, 
    df_embed_train_adimplencia, 
    pd.Series(y_pred_prob_train_adimplencia, name='nn_prob_adimplencia', index=X_train_lgbm.index),
    df_embed_train_lucro_inadim,
    pd.Series(y_pred_prob_train_lucro_inadim, name='nn_prob_lucro_inadim', index=X_train_lgbm.index)
], axis=1)

X_test_stacked = pd.concat([
    X_test_lgbm, 
    df_embed_test_adimplencia, 
    pd.Series(y_pred_prob_test_adimplencia, name='nn_prob_adimplencia', index=X_test_lgbm.index),
    df_embed_test_lucro_inadim,
    pd.Series(y_pred_prob_test_lucro_inadim, name='nn_prob_lucro_inadim', index=X_test_lgbm.index)
], axis=1)

X_train_stacked.columns = [str(col) for col in X_train_stacked.columns]
X_test_stacked.columns = [str(col) for col in X_test_stacked.columns]

categorical_features_lgbm_final = categorical_features
print(f"Features do Auditor: {X_train_stacked.shape[1]}")
print(f"  - {len(X_train_lgbm.columns)} originais")
print(f"  - {len(cols_embed_adimplencia)} embeddings adimplência + 1 probabilidade")
print(f"  - {len(cols_embed_lucro_inadim)} embeddings lucro-inadim + 1 probabilidade")
print(f"Features Categóricas para LGBM: {len(categorical_features_lgbm_final)}")

# O Auditor LGBM será treinado para prever LUCRO FINAL
lgb_train = lgb.Dataset(
    X_train_stacked, 
    label=y_train_lucro_target,
    categorical_feature=categorical_features_lgbm_final,
    free_raw_data=False
)
lgb_eval = lgb.Dataset(
    X_test_stacked, 
    label=y_test_lucro_target, 
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
        'scale_pos_weight': class_weight_lucro[0] / class_weight_lucro[1],
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
    'scale_pos_weight': class_weight_lucro[0] / class_weight_lucro[1],
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
print(f"{'3. Modelo (NN-Dupla+LGBM) com threshold':<45} R$ {lucro_lgbm:>12,.2f} {lucro_lgbm/lucro_max_test:>11.2%} {num_aceitos_lgbm:>11,}")
print(f"{'4. Modelo (NN-Dupla+LGBM) ponderado':<45} R$ {lucro_lgbm_ponderado:>12,.2f} {lucro_lgbm_ponderado/lucro_max_test:>11.2%} {'N/A':>11}")
print("="*85)
ganho_vs_todos_lgbm = lucro_lgbm - lucro_aceitar_todos
ganho_perc_vs_todos = (lucro_lgbm / lucro_aceitar_todos - 1) * 100 if lucro_aceitar_todos > 0 else float('inf')
print(f"\nGanho vs Aceitar Todos: R$ {ganho_vs_todos_lgbm:,.2f} ({ganho_perc_vs_todos:+.2f}%)")
print(f"Distância do Máximo Teórico: R$ {lucro_max_test - lucro_lgbm:,.2f}")


# ---------- 14. ANÁLISE DETALHADA (Teste / Validação) ----------
print("\n" + "="*70)
print("ANÁLISE DETALHADA (SAÍDA FINAL - LGBM OTIMIZADO V7)")
print("="*70)
df_analise = pd.DataFrame({
    'lucro_real': y_test_lucro,
    'prob_aceitar': y_pred_prob_test,
    'aceito': aceitar_lgbm,
    'lucrativo': y_test_lucro_target, # Target Final
    'adimplente': y_test_adimplencia_target, # Target A
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
    fdr_lgbm, rejeitados_lgbm, FP_lgbm, FN_lgbm = 0, 0, 0, 0, 0

print("\n" + "="*70)
print("ANÁLISE DO AUDITOR: LGBM (Dual-NN V7.1)")
print("="*70)
print(f"\n{'Métrica':<35} {'Auditor LGBM (Final V7.1)':>25}")
print("-"*62)
print(f"{'Lucro Otimizado':<35} R$ {lucro_opt_lgbm:>24,.2f}")
print(f"{'Ganho vs Aceitar Todos':<35} R$ {ganho_vs_todos_lgbm:>24,.2f}")
print(f"{'Erro em Aceitos (FDR)':<35} {fdr_lgbm:>24.2%}")
print(f"{'Falsos Positivos (Prejuízo Aceito)':<35} {FP_lgbm:>25,}")
print(f"{'Falsos Negativos (Lucro Rejeitado)':<35} {FN_lgbm:>25,}")
print(f"{'Total Rejeitados':<35} {rejeitados_lgbm:>25,}")

print("\n--- Análise de Pagamento (Contratos ACEITOS pelo LGBM V7.1) ---")
df_aceitos = df_analise[df_analise['aceito'] == True]
total_aceitos = len(df_aceitos)
if total_aceitos > 0:
    # adimplente=1 significa pagou 100%, adimplente=0 significa inadimplente
    adimplentes_totais_aceitos = (df_aceitos['adimplente'] == 1).sum()
    inadimplentes_aceitos = (df_aceitos['adimplente'] == 0).sum()
    
    inadimplentes_lucrativos_aceitos = ((df_aceitos['adimplente'] == 0) & (df_aceitos['lucrativo'] == True)).sum()
    inadimplentes_prejuizo_aceitos = ((df_aceitos['adimplente'] == 0) & (df_aceitos['lucrativo'] == False)).sum()
    
    print(f"Total de Aceitos: {total_aceitos:,}")
    print(f"  - 1. Adimplentes Totais (pagaram 100%): {adimplentes_totais_aceitos:,} ({adimplentes_totais_aceitos/total_aceitos:.2%})")
    print(f"  - 2. Inadimplentes Totais (pagaram < 100%): {inadimplentes_aceitos:,} ({inadimplentes_aceitos/total_aceitos:.2%})")
    print(f"     - Destes, Lucrativos ('Bom Risco'): {inadimplentes_lucrativos_aceitos:,}")
    print(f"     - Destes, Prejuízo ('Mau Risco'): {inadimplentes_prejuizo_aceitos:,}")
else:
    print("Nenhum contrato aceito.")

# ---------- 15. SALVAR MODELO ----------
GRAFICO_PATH = '../graficos/NN_LGBM_AUDITOR_DUPLO_V7.1'
MODELO_PATH = '../modelos/V7.1'
os.makedirs(GRAFICO_PATH, exist_ok=True)
os.makedirs(MODELO_PATH, exist_ok=True)

model_adimplencia.save(os.path.join(MODELO_PATH, 'nn_especialista_adimplencia.h5'))
print(f"\nModelo NN (Especialista A - Adimplência) salvo: {os.path.join(MODELO_PATH, 'nn_especialista_adimplencia.h5')}")
model_lucro_inadimplente.save(os.path.join(MODELO_PATH, 'nn_especialista_lucro_inadimplente.h5'))
print(f"Modelo NN (Especialista B - Lucro-Inadimplente) salvo: {os.path.join(MODELO_PATH, 'nn_especialista_lucro_inadimplente.h5')}")

lgbm_final.save_model(os.path.join(MODELO_PATH, 'lgbm_auditor_fusor.txt'))
print(f"Modelo LGBM (Auditor Fusor) salvo: {os.path.join(MODELO_PATH, 'lgbm_auditor_fusor.txt')}")

# ---------- 16. GRÁFICOS ----------
print("\n--- Gerando Gráficos ---")
# Não vamos gerar o histórico de treino das NNs para simplificar
# Foco nos gráficos de resultado do Auditor LGBM
# 1. Otimização de threshold (do LGBM Otimizado)
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
plt.savefig(os.path.join(GRAFICO_PATH, 'analise_completa.png'), dpi=300)
plt.close()
print("Análise completa (LGBM) salva")

# 2. Matriz de confusão visual
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_lgbm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Rejeitar', 'Aceitar'],
            yticklabels=['Prejuízo', 'Lucro'], ax=ax)
ax.set_xlabel('Decisão do Modelo (LGBM)', fontsize=12)
ax.set_ylabel('Realidade', fontsize=12)
ax.set_title('Matriz de Confusão - Lucro vs Decisão (LGBM)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICO_PATH, 'confusion_matrix.png'), dpi=300)
plt.close()
print("Matriz de confusão (LGBM) salva")

# ---------- 17. ANÁLISE SHAP (INTERPRETABILIDADE) ----------
print("\n" + "="*70)
print("ANÁLISE SHAP - INTERPRETABILIDADE DO AUDITOR LGBM")
print("="*70)

# Limitar amostra para performance
SHAP_SAMPLE = min(1000, len(X_test_stacked))
X_test_shap = X_test_stacked.iloc[:SHAP_SAMPLE]
y_test_shap = y_test_lucro_target[:SHAP_SAMPLE]
y_pred_test_shap = aceitar_lgbm[:SHAP_SAMPLE]

print(f"Calculando SHAP values para {SHAP_SAMPLE} amostras...")
explainer = shap.TreeExplainer(lgbm_final)
shap_values = explainer.shap_values(X_test_shap)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

# 1. Summary Plot (Dot)
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(shap_values, X_test_shap, show=False, max_display=20)
plt.title('SHAP Feature Importance - Top 20 (LGBM V7.1)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICO_PATH, 'shap_summary_dot.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ SHAP Summary Plot (dot) salvo")

# 2. Summary Plot (Bar)
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(shap_values, X_test_shap, plot_type='bar', show=False, max_display=20)
plt.title('SHAP Feature Importance - Top 20 (LGBM V7.1)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICO_PATH, 'shap_summary_bar.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ SHAP Summary Plot (bar) salvo")

# 3. Análise de Importância: Embeddings vs Probabilidades vs Features Originais
feature_importance = np.abs(shap_values).mean(axis=0)
feature_names = X_test_shap.columns.tolist()
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Categorizar features
importance_df['tipo'] = 'Original'
importance_df.loc[importance_df['feature'].str.startswith('embed_adimplencia'), 'tipo'] = 'Embed_Adimplencia'
importance_df.loc[importance_df['feature'].str.startswith('embed_lucro_inadim'), 'tipo'] = 'Embed_Lucro_Inadim'
importance_df.loc[importance_df['feature'] == 'nn_prob_adimplencia', 'tipo'] = 'Prob_Adimplencia'
importance_df.loc[importance_df['feature'] == 'nn_prob_lucro_inadim', 'tipo'] = 'Prob_Lucro_Inadim'

print("\n📊 Top 20 Features por SHAP Importance:")
print(importance_df.head(20).to_string(index=False))

# Análise agregada por tipo
print("\n🔍 Importância Agregada por Tipo de Feature:")
tipo_summary = importance_df.groupby('tipo')['importance'].agg(['sum', 'mean', 'count'])
tipo_summary = tipo_summary.sort_values('sum', ascending=False)
tipo_summary['pct_total'] = tipo_summary['sum'] / tipo_summary['sum'].sum() * 100
print(tipo_summary.to_string())

# Verificar se as probabilidades estão entre os top features
prob_adimplencia_rank = importance_df[importance_df['feature'] == 'nn_prob_adimplencia'].index[0] + 1 if 'nn_prob_adimplencia' in importance_df['feature'].values else None
prob_lucro_inadim_rank = importance_df[importance_df['feature'] == 'nn_prob_lucro_inadim'].index[0] + 1 if 'nn_prob_lucro_inadim' in importance_df['feature'].values else None

print(f"\n🎯 Ranking das Probabilidades:")
if prob_adimplencia_rank:
    prob_adimplencia_imp = importance_df[importance_df['feature'] == 'nn_prob_adimplencia']['importance'].values[0]
    print(f"  - nn_prob_adimplencia: #{prob_adimplencia_rank} (importance: {prob_adimplencia_imp:.4f})")
if prob_lucro_inadim_rank:
    prob_lucro_inadim_imp = importance_df[importance_df['feature'] == 'nn_prob_lucro_inadim']['importance'].values[0]
    print(f"  - nn_prob_lucro_inadim: #{prob_lucro_inadim_rank} (importance: {prob_lucro_inadim_imp:.4f})")

# 4. Análise de Correlação entre Embeddings
print("\n🔗 Analisando correlação entre embeddings Adimplência vs Lucro-Inadimplente...")
embed_adimplencia_cols = [col for col in X_test_shap.columns if col.startswith('embed_adimplencia_')]
embed_lucro_inadim_cols = [col for col in X_test_shap.columns if col.startswith('embed_lucro_inadim_')]

if len(embed_adimplencia_cols) > 0 and len(embed_lucro_inadim_cols) > 0:
    correlations = []
    for i in range(min(len(embed_adimplencia_cols), len(embed_lucro_inadim_cols))):
        corr = np.corrcoef(
            X_test_shap[embed_adimplencia_cols[i]], 
            X_test_shap[embed_lucro_inadim_cols[i]]
        )[0, 1]
        correlations.append({
            'embed_idx': i,
            'correlation': corr,
            'adimplencia_importance': importance_df[importance_df['feature'] == embed_adimplencia_cols[i]]['importance'].values[0],
            'lucro_inadim_importance': importance_df[importance_df['feature'] == embed_lucro_inadim_cols[i]]['importance'].values[0]
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df['avg_importance'] = (corr_df['adimplencia_importance'] + corr_df['lucro_inadim_importance']) / 2
    corr_df = corr_df.sort_values('avg_importance', ascending=False)
    
    print("\n🔗 Correlação entre Embeddings (Top 10 por importância):")
    print(corr_df.head(10).to_string(index=False))
    
    high_corr = corr_df[corr_df['correlation'].abs() > 0.8]
    if len(high_corr) > 0:
        print(f"\n⚠️  ALERTA: {len(high_corr)} pares de embeddings com correlação >0.8 (redundância)")
        print(high_corr[['embed_idx', 'correlation', 'avg_importance']].to_string(index=False))
    
    low_corr = corr_df[corr_df['correlation'].abs() < 0.3]
    print(f"\n✅ {len(low_corr)} pares de embeddings com correlação <0.3 (complementares)")

# 5. Gráfico de Importância por Tipo
fig, ax = plt.subplots(figsize=(10, 6))
tipo_plot = tipo_summary.reset_index()
ax.bar(tipo_plot['tipo'], tipo_plot['pct_total'], color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
ax.set_ylabel('% da Importância Total', fontsize=12)
ax.set_xlabel('Tipo de Feature', fontsize=12)
ax.set_title('Contribuição por Tipo de Feature (SHAP)', fontsize=14, fontweight='bold')
for i, row in tipo_plot.iterrows():
    ax.text(i, row['pct_total'] + 1, f"{row['pct_total']:.1f}%", ha='center', fontsize=10, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICO_PATH, 'shap_importance_by_type.png'), dpi=300)
plt.close()
print("✓ Gráfico de importância por tipo salvo")

# 6. Waterfall plots para casos específicos
# True Positives
tp_indices = np.where((y_test_shap == 1) & (y_pred_test_shap == True))[0]
if len(tp_indices) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[tp_indices[0]],
        base_values=explainer.expected_value,
        data=X_test_shap.iloc[tp_indices[0]],
        feature_names=X_test_shap.columns.tolist()
    ), max_display=15, show=False)
    plt.title('SHAP Waterfall - True Positive (Acerto: Lucrativo Aceito)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICO_PATH, 'shap_waterfall_tp.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Waterfall TP salvo")

# False Positives
fp_indices = np.where((y_test_shap == 0) & (y_pred_test_shap == True))[0]
if len(fp_indices) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[fp_indices[0]],
        base_values=explainer.expected_value,
        data=X_test_shap.iloc[fp_indices[0]],
        feature_names=X_test_shap.columns.tolist()
    ), max_display=15, show=False)
    plt.title('SHAP Waterfall - False Positive (Erro: Prejuízo Aceito)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICO_PATH, 'shap_waterfall_fp.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Waterfall FP salvo")

# False Negatives
fn_indices = np.where((y_test_shap == 1) & (y_pred_test_shap == False))[0]
if len(fn_indices) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[fn_indices[0]],
        base_values=explainer.expected_value,
        data=X_test_shap.iloc[fn_indices[0]],
        feature_names=X_test_shap.columns.tolist()
    ), max_display=15, show=False)
    plt.title('SHAP Waterfall - False Negative (Erro: Lucrativo Rejeitado)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICO_PATH, 'shap_waterfall_fn.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Waterfall FN salvo")

# 7. Salvar relatório de interpretabilidade
report_path = os.path.join(GRAFICO_PATH, 'interpretability_report_v7.1.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("RELATÓRIO DE INTERPRETABILIDADE - DUAL-NN + LGBM (V7.1)\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Amostra SHAP: {SHAP_SAMPLE} contratos\n\n")
    
    f.write("--- ESTRATÉGIA V7.1: INADIMPLENTES LUCRATIVOS ---\n")
    f.write(f"NN-A: Adimplente (100%) vs Inadimplente (<100%)\n")
    f.write(f"NN-B: Lucro entre Inadimplentes (treinada com {inadimplentes_train_mask.sum():,} inadimplentes)\n")
    f.write(f"LGBM: Fusão das duas visões para decisão final\n\n")
    
    f.write("--- TOP 20 FEATURES (SHAP) ---\n")
    f.write(importance_df.head(20).to_string(index=False))
    f.write("\n\n")
    
    f.write("--- IMPORTÂNCIA POR TIPO ---\n")
    f.write(tipo_summary.to_string())
    f.write("\n\n")
    
    f.write("--- RANKING DAS PROBABILIDADES ---\n")
    if prob_adimplencia_rank:
        f.write(f"nn_prob_adimplencia: #{prob_adimplencia_rank} (importance: {prob_adimplencia_imp:.4f})\n")
    if prob_lucro_inadim_rank:
        f.write(f"nn_prob_lucro_inadim: #{prob_lucro_inadim_rank} (importance: {prob_lucro_inadim_imp:.4f})\n")
    f.write("\n")
    
    if len(embed_adimplencia_cols) > 0 and len(embed_lucro_inadim_cols) > 0:
        f.write("--- CORRELAÇÃO EMBEDDINGS (TOP 10) ---\n")
        f.write(corr_df.head(10).to_string(index=False))
        f.write("\n\n")
        
        if len(high_corr) > 0:
            f.write(f"⚠️  {len(high_corr)} pares com correlação >0.8 (redundância)\n")
        f.write(f"✅ {len(low_corr)} pares com correlação <0.3 (complementares)\n")

print(f"\n✓ Relatório de interpretabilidade salvo: {report_path}")

print("\n" + "="*70)
print("ANÁLISE COMPLETA (V7.0)")
print("="*70)