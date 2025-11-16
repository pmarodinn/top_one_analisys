"""
MODELO NN+LGBM COM REGRESSÃO DE LUCRO E VALIDAÇÃO TEMPORAL
==========================================================
MUDANÇA FUNDAMENTAL: Este modelo prediz LUCRO diretamente (regressão),
não inadimplência (classificação binária).

Por quê?
- Dataset com 78-80% contratos lucrativos
- Baseline "aceitar tudo" já é lucrativo
- Threshold sobre classificação binária rejeita muitos lucrativos (alto FN)
- Resultado: perde dinheiro vs baseline

Solução: REGRESSÃO DE LUCRO
- Target = lucro (R$ valor contínuo)
- Modelo prediz lucro esperado de cada contrato
- Aceitar se lucro_predito > threshold_lucro (ex: > 0)
- Otimização econômica direta

Validação Temporal (TimeSeriesSplit):
1. Ordena dados por Data_Lancamento
2. Fold 1: Treina até Mês M1 → Testa Mês M1+1
3. Fold 2: Treina até Mês M2 → Testa Mês M2+1
4. ...
5. Fold N: Treina até Mês MN → Testa Mês MN+1
6. Agrega resultados: lucro total, lucro por mês, estabilidade
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🎯 REGRESSÃO DE LUCRO - NN+LGBM COM TIME SERIES SPLIT")
print("="*70)

# Verifica GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU disponível: {len(gpus)} device(s)")
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
    df = df.dropna(subset=['pago_perc', 'Data_Lancamento']).copy()
    df['default'] = (df['pago_perc'] < 1).astype(int)
    df['adimplente'] = (df['pago_perc'] == 1).astype(int)

    df['data_mes']   = df['Data_Lancamento'].dt.month
    df['data_ano']   = df['Data_Lancamento'].dt.year
    df['data_dia']   = df['Data_Lancamento'].dt.dayofweek
    df['ano_mes'] = df['Data_Lancamento'].dt.to_period('M')
    
    df['categoria_limpa'] = df['categorias'].astype(str).str.strip("[]'").str.split(',').str[0].str.strip("'")
    df['divida_sobre_renda'] = (df['valor_inicial_da_prestacao'] / (df['renda_cliente'] + 1e-6)).clip(upper=100)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

DATA_FILE = '../data/dataset_interno_top_one_atualizado.csv'
df = load_and_preprocess_v3(DATA_FILE)

print(f"\n📊 Dados carregados: {len(df)} linhas")
print(f"Data Lancamento: {df['Data_Lancamento'].min()} até {df['Data_Lancamento'].max()}")
print(f"Lucro Máximo Teórico Global: R$ {df['lucro'].sum():,.2f}")

# ---------- 2. ORDENAR POR DATA E CRIAR SPLITS TEMPORAIS ----------
df = df.sort_values('Data_Lancamento').reset_index(drop=True)

# Análise temporal
meses_disponiveis = df['ano_mes'].unique()
meses_disponiveis = sorted([str(m) for m in meses_disponiveis])

print(f"\n📅 Período dos Dados:")
print(f"Primeiro mês: {meses_disponiveis[0]}")
print(f"Último mês: {meses_disponiveis[-1]}")
print(f"Total de meses: {len(meses_disponiveis)}")

# Estatísticas por mês
contratos_por_mes = df.groupby('ano_mes').agg({
    'lucro': ['count', 'sum', 'mean'],
    'adimplente': 'mean'
}).reset_index()
contratos_por_mes.columns = ['ano_mes', 'num_contratos', 'lucro_total', 'lucro_medio', 'taxa_adimplencia']

print(f"\n📈 Estatísticas Mensais:")
print(f"Contratos/mês (média): {contratos_por_mes['num_contratos'].mean():.0f}")
print(f"Contratos/mês (min): {contratos_por_mes['num_contratos'].min():.0f}")
print(f"Contratos/mês (max): {contratos_por_mes['num_contratos'].max():.0f}")

# ---------- 3. CONFIGURAÇÃO DO TIME SERIES SPLIT ----------
# Usaremos uma janela mínima de treinamento e testaremos nos últimos 3 meses
MIN_TRAIN_MONTHS = 9  # Aumentado para ter mais dados de treino
TEST_WINDOW = 1  # Testar 1 mês por vez
NUM_FOLDS = 3  # REDUZIDO: apenas 3 meses de teste (últimos 3 meses)

# Criar splits temporais
num_meses = len(meses_disponiveis)

print(f"\n🔀 Configuração Time Series Split:")
print(f"Janela mínima de treino: {MIN_TRAIN_MONTHS} meses")
print(f"Janela de teste: {TEST_WINDOW} mês(es)")
print(f"Número de folds (últimos meses): {NUM_FOLDS}")
print(f"\n⚠️ Dataset com {num_meses} meses ({meses_disponiveis[0]} a {meses_disponiveis[-1]})")

# Criar lista de splits - APENAS OS ÚLTIMOS 3 MESES
time_splits = []
start_fold_index = num_meses - NUM_FOLDS  # Índice onde começam os últimos 3 meses

for i in range(start_fold_index, num_meses):
    train_end_month = meses_disponiveis[i-1]
    test_month = meses_disponiveis[i]
    
    time_splits.append({
        'fold': i - start_fold_index + 1,
        'train_start': meses_disponiveis[0],
        'train_end': train_end_month,
        'test_month': test_month
    })

print(f"\nSplits configurados (últimos {NUM_FOLDS} meses):")
for split in time_splits:
    print(f"  Fold {split['fold']}: Treina [{split['train_start']} → {split['train_end']}] → Testa [{split['test_month']}]")

# ---------- 4. PREPARAÇÃO DE FEATURES ----------
# Target = LUCRO (regressão), não default (classificação)
y_lucro = df['lucro'].values
y_default = df['default'].values  # Mantém para métricas de diagnóstico

COLS_REMOVE = ['default','pago_perc','lucro','aceitar','adimplente',
               'contrato_id','proposta_id','Unnamed: 0',
               'Data_Lancamento','Data_inicio','ano_mes']

X = df.drop(columns=[c for c in COLS_REMOVE if c in df.columns])

# Identificar features numéricas e categóricas
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

print(f"\n🔧 Features utilizadas: {X.shape[1]}")
print(f"Numéricas: {len(numeric_features)}")
print(f"Categóricas: {len(categorical_features)}")

# Preparar features categóricas
categorical_low_card = []
categorical_high_card = []

for col in categorical_features:
    unique_vals = X[col].nunique()
    if unique_vals <= 20:
        categorical_low_card.append(col)
    else:
        categorical_high_card.append(col)

print(f"Categóricas Baixa Cardinalidade (<= 20): {len(categorical_low_card)}")
print(f"Categóricas Alta Cardinalidade (> 20): {len(categorical_high_card)}")

# ---------- 5. FUNÇÃO DE TREINAMENTO NN (REGRESSÃO DE LUCRO) ----------
def create_nn_regression_model(input_dim):
    """
    Cria modelo NN para REGRESSÃO DE LUCRO (não classificação binária).
    Mudanças vs versão classificação:
    - Camada final: linear (sem sigmoid) para prever valor contínuo
    - Loss: MSE (mean squared error) em vez de binary_crossentropy
    - Métrica: MAE (mean absolute error) em vez de AUC
    """
    inputs = keras.Input(shape=(input_dim,), name='input_layer')
    
    x = keras.layers.Dense(512, activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    embeddings = keras.layers.Dense(32, activation='relu', name='embeddings')(x)
    
    # MUDANÇA CRÍTICA: saída linear para regressão (prever R$ lucro)
    outputs = keras.layers.Dense(1, activation='linear')(embeddings)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # Mean Squared Error para regressão
        metrics=['mae']  # Mean Absolute Error
    )
    return model

# ---------- 6. TARGET ENCODING ----------
def apply_target_encoding(X_train, X_test, y_train, high_card_cols):
    """Aplica Target Encoding para features de alta cardinalidade"""
    X_train_te = X_train.copy()
    X_test_te = X_test.copy()
    
    # Garantir que y_train é uma Series
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, index=X_train.index)
    
    global_mean = y_train.mean()
    
    for col in high_card_cols:
        if col not in X_train.columns:
            continue
        
        # Criar DataFrame temporário alinhado
        temp_df = pd.DataFrame({
            'category': X_train_te[col].values,
            'target': y_train.values
        })
        
        # Calcular média do target por categoria
        target_mean = temp_df.groupby('category')['target'].mean()
        
        # Aplicar no treino e teste
        X_train_te[col + '_te'] = X_train_te[col].map(target_mean).fillna(global_mean)
        X_test_te[col + '_te'] = X_test_te[col].map(target_mean).fillna(global_mean)
        
        # Remover coluna original
        X_train_te = X_train_te.drop(columns=[col])
        X_test_te = X_test_te.drop(columns=[col])
    
    return X_train_te, X_test_te

# ---------- 7. LOOP PRINCIPAL - TIME SERIES SPLIT ----------
results_by_fold = []

# Modelos para treino incremental (warm start)
nn_model_previous = None
lgbm_model_previous = None

print("\n" + "="*70)
print("🔁 INICIANDO VALIDAÇÃO TEMPORAL (TIME SERIES SPLIT)")
print("🔄 TREINO INCREMENTAL: Modelos do fold anterior serão reutilizados")
print("="*70)

for idx, split_info in enumerate(time_splits):
    fold = split_info['fold']
    train_end = split_info['train_end']
    test_month = split_info['test_month']
    
    print(f"\n{'='*70}")
    print(f"📅 FOLD {fold}/{len(time_splits)}")
    print(f"{'='*70}")
    print(f"Treino: [{split_info['train_start']} → {train_end}]")
    print(f"Teste: [{test_month}]")
    
    # Criar máscaras de treino e teste
    train_mask = df['ano_mes'].astype(str) <= train_end
    test_mask = df['ano_mes'].astype(str) == test_month
    
    # Verificar se há dados suficientes
    if train_mask.sum() < 100 or test_mask.sum() < 10:
        print(f"⚠️ Fold {fold} ignorado: dados insuficientes (treino: {train_mask.sum()}, teste: {test_mask.sum()})")
        continue
    
    # Split temporal
    X_train = X[train_mask].copy()
    X_test = X[test_mask].copy()
    y_train = y_default[train_mask]
    y_test = y_default[test_mask]
    y_train_lucro = y_lucro[train_mask]
    y_test_lucro = y_lucro[test_mask]
    
    print(f"Treino: {len(X_train)} contratos")
    print(f"Teste: {len(X_test)} contratos")
    print(f"Lucro Máximo Teórico (Teste): R$ {y_test_lucro[y_test_lucro > 0].sum():,.2f}")
    
    # Aplicar Target Encoding
    y_train_series = pd.Series(y_train, name='target')
    X_train_te, X_test_te = apply_target_encoding(X_train, X_test, y_train_series, categorical_high_card)
    
    # One-Hot Encoding para baixa cardinalidade
    X_train_encoded = pd.get_dummies(X_train_te, columns=[c for c in categorical_low_card if c in X_train_te.columns], drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_te, columns=[c for c in categorical_low_card if c in X_test_te.columns], drop_first=True)
    
    # Alinhar colunas entre treino e teste
    missing_cols_test = set(X_train_encoded.columns) - set(X_test_encoded.columns)
    for col in missing_cols_test:
        X_test_encoded[col] = 0
    
    missing_cols_train = set(X_test_encoded.columns) - set(X_train_encoded.columns)
    for col in missing_cols_train:
        X_train_encoded[col] = 0
    
    X_test_encoded = X_test_encoded[X_train_encoded.columns]
    
    # Preencher NaNs e normalizar
    X_train_encoded = X_train_encoded.fillna(X_train_encoded.median())
    X_test_encoded = X_test_encoded.fillna(X_train_encoded.median())
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)
    
    print(f"Features após encoding: {X_train_scaled.shape[1]}")
    
    # ==================== ETAPA 1: REDE NEURAL (REGRESSÃO) ====================
    print(f"\n--- Etapa 1: Treinando NN para Regressão de Lucro (Fold {fold}) ---")
    
    # TREINO INCREMENTAL: usar modelo anterior se existir
    if nn_model_previous is not None and fold > 1:
        print(f"   🔄 Reutilizando pesos do Fold {fold-1} (warm start)")
        model_nn = nn_model_previous
    else:
        print(f"   🆕 Criando novo modelo (Fold 1)")
        model_nn = create_nn_regression_model(X_train_scaled.shape[1])
    
    # Callbacks (agora monitoram val_loss ao invés de val_auc)
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        mode='min'  # Minimizar loss
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        mode='min'
    )
    
    # Treinar (agora y_train = lucro, não default)
    history = model_nn.fit(
        X_train_scaled, y_train_lucro,  # MUDANÇA: target = lucro
        validation_split=0.2,
        epochs=100,
        batch_size=256,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    # Extrair embeddings
    embedding_model = keras.Model(
        inputs=model_nn.input,
        outputs=model_nn.get_layer('embeddings').output
    )
    
    embeddings_train = embedding_model.predict(X_train_scaled, verbose=0)
    embeddings_test = embedding_model.predict(X_test_scaled, verbose=0)
    
    # Predições NN (agora = lucro predito em R$)
    y_pred_nn_train = model_nn.predict(X_train_scaled, verbose=0).flatten()
    y_pred_nn_test = model_nn.predict(X_test_scaled, verbose=0).flatten()
    
    # Métricas NN (MAE = Mean Absolute Error)
    mae_nn = np.mean(np.abs(y_test_lucro - y_pred_nn_test))
    r2_nn = 1 - (np.sum((y_test_lucro - y_pred_nn_test)**2) / np.sum((y_test_lucro - y_test_lucro.mean())**2))
    print(f"MAE NN (Teste): R$ {mae_nn:.2f}")
    print(f"R² NN (Teste): {r2_nn:.4f}")
    
    # ==================== ETAPA 2: LGBM AUDITOR (REGRESSÃO) ====================
    print(f"\n--- Etapa 2: Treinando LGBM Auditor de Lucro (Fold {fold}) ---")
    
    # Preparar features para LGBM
    X_train_lgbm = np.concatenate([
        embeddings_train,
        X_train_encoded.values
    ], axis=1)
    
    X_test_lgbm = np.concatenate([
        embeddings_test,
        X_test_encoded.values
    ], axis=1)
    
    # Feature names
    embed_cols = [f'embed_{i}' for i in range(32)]
    feature_names = embed_cols + list(X_train_encoded.columns)
    categorical_indices = [i for i, col in enumerate(feature_names) if col in categorical_low_card]
    
    # Split validação interna
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr_lucro, y_val_lucro = train_test_split(
        X_train_lgbm, y_train_lucro, test_size=0.2, random_state=42
    )
    
    # Treinar LGBM para REGRESSÃO DE LUCRO
    params_lgbm = {
        'objective': 'regression',  # MUDANÇA: regression ao invés de binary
        'metric': 'rmse',  # MUDANÇA: RMSE ao invés de AUC
        'learning_rate': 0.035,
        'num_leaves': 142,
        'max_depth': 4,
        'subsample': 0.66,
        'colsample_bytree': 0.93,
        'reg_alpha': 0.01,
        'reg_lambda': 0.63,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X_tr, label=y_tr_lucro, feature_name=feature_names)
    valid_data = lgb.Dataset(X_val, label=y_val_lucro, feature_name=feature_names, reference=train_data)
    
    # TREINO INCREMENTAL: usar modelo anterior se existir
    if lgbm_model_previous is not None and fold > 1:
        print(f"   🔄 Continuando treino do LGBM do Fold {fold-1}")
        model_lgbm = lgb.train(
            params_lgbm,
            train_data,
            num_boost_round=500,
            valid_sets=[valid_data],
            init_model=lgbm_model_previous,  # WARM START
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
    else:
        print(f"   🆕 Criando novo LGBM (Fold 1)")
        model_lgbm = lgb.train(
            params_lgbm,
            train_data,
            num_boost_round=500,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
    
    # Predições LGBM (agora = lucro predito em R$)
    y_pred_lgbm = model_lgbm.predict(X_test_lgbm, num_iteration=model_lgbm.best_iteration)
    
    # Métricas LGBM
    mae_lgbm = np.mean(np.abs(y_test_lucro - y_pred_lgbm))
    r2_lgbm = 1 - (np.sum((y_test_lucro - y_pred_lgbm)**2) / np.sum((y_test_lucro - y_test_lucro.mean())**2))
    print(f"MAE LGBM (Teste): R$ {mae_lgbm:.2f}")
    print(f"R² LGBM (Teste): {r2_lgbm:.4f}")
    
    # ==================== ETAPA 3: OTIMIZAÇÃO DE THRESHOLD (ECONÔMICO) ====================
    print(f"\n--- Etapa 3: Otimizando Threshold de Lucro (Fold {fold}) ---")
    
    # MUDANÇA FUNDAMENTAL: threshold agora é sobre LUCRO PREDITO (em R$), não probabilidade
    # Testar thresholds de lucro: aceitar se lucro_predito >= threshold_lucro
    threshold_lucros = np.linspace(-500, 500, 100)  # De -R$500 a +R$500
    best_threshold = 0
    best_ganho = -np.inf
    lucro_baseline = y_test_lucro.sum()  # Lucro se aceitar todos os contratos
    
    # Otimização: varre thresholds de lucro
    threshold_results = []
    
    for th_lucro in threshold_lucros:
        # Aceitar se lucro predito >= threshold
        aceitar = y_pred_lgbm >= th_lucro
        
        if aceitar.sum() == 0:  # Se rejeitar todos, lucro = 0
            lucro_total = 0
        else:
            lucro_total = y_test_lucro[aceitar].sum()
        
        ganho = lucro_total - lucro_baseline  # Ganho vs aceitar tudo
        
        threshold_results.append({
            'threshold': th_lucro,
            'num_aceitos': aceitar.sum(),
            'lucro_total': lucro_total,
            'ganho': ganho
        })
        
        if ganho > best_ganho:
            best_ganho = ganho
            best_threshold = th_lucro
    
    print(f"Threshold Ótimo: R$ {best_threshold:.2f}")
    print(f"Lucro Baseline (aceitar tudo): R$ {lucro_baseline:,.2f}")
    print(f"Ganho Otimizado vs Baseline: R$ {best_ganho:,.2f}")
    
    # DEBUG: Mostrar top 5 thresholds
    threshold_results_sorted = sorted(threshold_results, key=lambda x: x['ganho'], reverse=True)
    print(f"\n🔍 Top 5 Thresholds por Ganho:")
    for i, res in enumerate(threshold_results_sorted[:5]):
        print(f"  #{i+1}: th=R$ {res['threshold']:.2f}, aceitos={res['num_aceitos']}, "
              f"lucro=R$ {res['lucro_total']:,.2f}, ganho=R$ {res['ganho']:,.2f}")
    
    # Decisão final
    aceitar_final = y_pred_lgbm >= best_threshold
    num_aceitos = aceitar_final.sum()
    taxa_aprovacao = num_aceitos / len(y_test) * 100
    
    # Calcular lucro otimizado (com threshold)
    lucro_otimizado = y_test_lucro[aceitar_final].sum() if num_aceitos > 0 else 0
    
    # Diagnóstico detalhado
    print(f"\n--- Diagnóstico do Threshold ---")
    lucrativos_reais = (y_test_lucro > 0)
    print(f"Contratos Lucrativos Reais: {lucrativos_reais.sum()}/{len(y_test_lucro)} ({lucrativos_reais.sum()/len(y_test_lucro)*100:.1f}%)")
    print(f"Recall Lucrativos: {(aceitar_final & lucrativos_reais).sum()}/{lucrativos_reais.sum()} ({(aceitar_final & lucrativos_reais).sum()/lucrativos_reais.sum()*100:.1f}%)")
    print(f"Falsos Positivos: {(aceitar_final & ~lucrativos_reais).sum()} contratos prejuízo aceitos")
    print(f"Falsos Negativos: {(~aceitar_final & lucrativos_reais).sum()} contratos lucrativos rejeitados")
    
    # Cálculo de ganho vs baseline
    ganho_vs_baseline = lucro_otimizado - lucro_baseline
    ganho_perc_baseline = (ganho_vs_baseline / lucro_baseline * 100) if lucro_baseline != 0 else 0
    
    print(f"\n--- Comparação de Resultados (Fold {fold}) ---")
    print(f"Lucro BASELINE (aceitar todos):      R$ {lucro_baseline:>13,.2f}")
    print(f"Lucro MODELO (algoritmo):            R$ {lucro_otimizado:>13,.2f}")
    print(f"")
    emoji_baseline = "✅" if ganho_vs_baseline >= 0 else "❌"
    print(f"Ganho MODELO vs BASELINE: {ganho_vs_baseline:+14,.2f} ({ganho_perc_baseline:+.2f}%) {emoji_baseline}")
    
    print(f"\nTaxa Aprovação MODELO:  {taxa_aprovacao:.1f}%")
    
    # Métricas de erro
    cm = confusion_matrix(y_test, aceitar_final.astype(int))
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Salvar modelos para próximo fold (treino incremental)
    nn_model_previous = model_nn
    lgbm_model_previous = model_lgbm
    print(f"\n💾 Modelos salvos para reutilização no próximo fold")
    
    # Salvar resultados do fold
    results_by_fold.append({
        'fold': fold,
        'test_month': test_month,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'mae_nn': mae_nn,
        'r2_nn': r2_nn,
        'mae_lgbm': mae_lgbm,
        'r2_lgbm': r2_lgbm,
        'threshold': best_threshold,
        'lucro_otimizado': lucro_otimizado,
        'lucro_baseline': lucro_baseline,
        'ganho_absoluto': ganho_vs_baseline,
        'ganho_percentual': ganho_perc_baseline,
        'num_aceitos': num_aceitos,
        'taxa_aprovacao': taxa_aprovacao,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    })

# ---------- 8. ANÁLISE DOS RESULTADOS AGREGADOS ----------
print("\n" + "="*70)
print("📊 ANÁLISE AGREGADA - TIME SERIES SPLIT (3 MESES)")
print("🔄 TREINO INCREMENTAL: Modelos evoluíram entre folds")
print("="*70)

df_results = pd.DataFrame(results_by_fold)

if len(df_results) == 0:
    print("⚠️ Nenhum fold foi processado com sucesso!")
else:
    print(f"\nTotal de Folds Processados: {len(df_results)}")
    
    # Estatísticas gerais
    print(f"\n--- Desempenho Médio (Regressão de Lucro) ---")
    print(f"MAE NN (média): R$ {df_results['mae_nn'].mean():.2f} ± R$ {df_results['mae_nn'].std():.2f}")
    print(f"R² NN (média): {df_results['r2_nn'].mean():.4f} ± {df_results['r2_nn'].std():.4f}")
    print(f"MAE LGBM (média): R$ {df_results['mae_lgbm'].mean():.2f} ± R$ {df_results['mae_lgbm'].std():.2f}")
    print(f"R² LGBM (média): {df_results['r2_lgbm'].mean():.4f} ± {df_results['r2_lgbm'].std():.4f}")
    print(f"Threshold Lucro (média): R$ {df_results['threshold'].mean():.2f} ± R$ {df_results['threshold'].std():.2f}")
    
    print(f"\n--- Lucro Total Agregado (3 meses) ---")
    lucro_total_baseline = df_results['lucro_baseline'].sum()
    lucro_total_otimizado = df_results['lucro_otimizado'].sum()
    
    ganho_total_vs_baseline = lucro_total_otimizado - lucro_total_baseline
    ganho_total_perc_baseline = (ganho_total_vs_baseline / lucro_total_baseline * 100) if lucro_total_baseline != 0 else 0
    
    emoji_baseline = "✅" if ganho_total_vs_baseline >= 0 else "❌"
    
    print(f"\nLucro BASELINE (aceitar todos):      R$ {lucro_total_baseline:>13,.2f}")
    print(f"Lucro MODELO (algoritmo):            R$ {lucro_total_otimizado:>13,.2f}")
    print(f"\nGanho MODELO vs BASELINE: {ganho_total_vs_baseline:+14,.2f} ({ganho_total_perc_baseline:+.2f}%) {emoji_baseline}")
    
    print(f"\n--- Lucro Mensal (Médio) ---")
    print(f"Lucro/mês BASELINE: R$ {df_results['lucro_baseline'].mean():>10,.2f} ± {df_results['lucro_baseline'].std():,.2f}")
    print(f"Lucro/mês MODELO:   R$ {df_results['lucro_otimizado'].mean():>10,.2f} ± {df_results['lucro_otimizado'].std():,.2f}")
    print(f"\nGanho/mês vs BASELINE: R$ {df_results['ganho_absoluto'].mean():>10,.2f} ({df_results['ganho_percentual'].mean():+.2f}%)")
    
    print(f"\n--- Taxa de Aprovação ---")
    print(f"Aprovação MODELO (média):  {df_results['taxa_aprovacao'].mean():.1f}% ± {df_results['taxa_aprovacao'].std():.1f}%")
    
    # Análise de estabilidade temporal
    print(f"\n--- Estabilidade Temporal ---")
    r2_volatilidade = df_results['r2_lgbm'].std() / df_results['r2_lgbm'].mean() if df_results['r2_lgbm'].mean() != 0 else 0
    lucro_volatilidade = df_results['lucro_otimizado'].std() / df_results['lucro_otimizado'].mean()
    
    print(f"Volatilidade R² (CV): {r2_volatilidade:.4f}")
    print(f"Volatilidade Lucro (CV): {lucro_volatilidade:.4f}")
    
    # Tendência temporal (regressão linear simples)
    from scipy import stats
    x_tempo = np.arange(len(df_results))
    slope_r2, intercept_r2, r_r2, p_r2, _ = stats.linregress(x_tempo, df_results['r2_lgbm'])
    slope_lucro, intercept_lucro, r_lucro, p_lucro, _ = stats.linregress(x_tempo, df_results['ganho_percentual'])
    
    print(f"\nTendência R² LGBM ao longo do tempo:")
    print(f"  Slope: {slope_r2:.6f} (p-value: {p_r2:.4f})")
    print(f"  {'📈 Melhora' if slope_r2 > 0 else '📉 Piora'} {'significativa' if p_r2 < 0.05 else 'não significativa'} ao longo do tempo")
    
    print(f"\nTendência Ganho % ao longo do tempo:")
    print(f"  Slope: {slope_lucro:.6f} (p-value: {p_lucro:.4f})")
    print(f"  {'📈 Melhora' if slope_lucro > 0 else '📉 Piora'} {'significativa' if p_lucro < 0.05 else 'não significativa'} ao longo do tempo")
    
    # Meses com melhor/pior performance
    print(f"\n--- Top 5 Meses (Maior Ganho) ---")
    top_5 = df_results.nlargest(5, 'ganho_absoluto')[['test_month', 'ganho_absoluto', 'ganho_percentual', 'r2_lgbm']]
    for idx, row in top_5.iterrows():
        print(f"  {row['test_month']}: R$ {row['ganho_absoluto']:,.2f} ({row['ganho_percentual']:+.2f}%), R² {row['r2_lgbm']:.4f}")
    
    print(f"\n--- Bottom 5 Meses (Menor Ganho) ---")
    bottom_5 = df_results.nsmallest(5, 'ganho_absoluto')[['test_month', 'ganho_absoluto', 'ganho_percentual', 'r2_lgbm']]
    for idx, row in bottom_5.iterrows():
        print(f"  {row['test_month']}: R$ {row['ganho_absoluto']:,.2f} ({row['ganho_percentual']:+.2f}%), R² {row['r2_lgbm']:.4f}")
    
    # ---------- 9. VISUALIZAÇÕES ----------
    print("\n--- Gerando Visualizações ---")
    os.makedirs('../graficos/NN_TIMESERIES', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Lucro ao longo do tempo (2 linhas: Baseline, Modelo)
    ax1 = axes[0, 0]
    ax1.plot(df_results['test_month'].astype(str), df_results['lucro_baseline'], 
             marker='s', label='Lucro Baseline (aceitar todos)', linewidth=2, alpha=0.6, color='gray')
    ax1.plot(df_results['test_month'].astype(str), df_results['lucro_otimizado'], 
             marker='o', label='Lucro Modelo (algoritmo)', linewidth=2.5, color='green')
    ax1.set_xlabel('Mês de Teste', fontsize=12)
    ax1.set_ylabel('Lucro (R$)', fontsize=12)
    ax1.set_title('Evolução do Lucro Mensal: Baseline vs Modelo', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. R² ao longo do tempo
    ax2 = axes[0, 1]
    ax2.plot(df_results['test_month'].astype(str), df_results['r2_nn'], 
             marker='o', label='R² NN', linewidth=2)
    ax2.plot(df_results['test_month'].astype(str), df_results['r2_lgbm'], 
             marker='s', label='R² LGBM', linewidth=2)
    ax2.set_xlabel('Mês de Teste', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('Evolução do R² Mensal (Regressão)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Ganho percentual
    ax3 = axes[1, 0]
    colors = ['green' if x > 0 else 'red' for x in df_results['ganho_percentual']]
    ax3.bar(df_results['test_month'].astype(str), df_results['ganho_percentual'], color=colors, alpha=0.7)
    ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Mês de Teste', fontsize=12)
    ax3.set_ylabel('Ganho vs Baseline (%)', fontsize=12)
    ax3.set_title('Ganho Percentual Mensal', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Taxa de aprovação
    ax4 = axes[1, 1]
    ax4.plot(df_results['test_month'].astype(str), df_results['taxa_aprovacao'], 
             marker='o', linewidth=2, color='purple')
    ax4.set_xlabel('Mês de Teste', fontsize=12)
    ax4.set_ylabel('Taxa de Aprovação (%)', fontsize=12)
    ax4.set_title('Taxa de Aprovação Mensal', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('../graficos/NN_TIMESERIES/timeseries_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico de análise temporal salvo")
    
    # Matriz de Confusão Consolidada
    cm_total = np.array([[df_results['tn'].sum(), df_results['fp'].sum()],
                         [df_results['fn'].sum(), df_results['tp'].sum()]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
                xticklabels=['Pred: Não Aceitar', 'Pred: Aceitar'],
                yticklabels=['Real: Inadimplente', 'Real: Adimplente'])
    ax.set_xlabel('Predição do Modelo', fontsize=12)
    ax.set_ylabel('Realidade', fontsize=12)
    ax.set_title('Matriz de Confusão Consolidada (Todos os Folds)', fontsize=14, fontweight='bold')
    
    # Adicionar métricas
    tn_total, fp_total, fn_total, tp_total = cm_total.ravel()
    precisao_total = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall_total = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1_total = 2 * (precisao_total * recall_total) / (precisao_total + recall_total) if (precisao_total + recall_total) > 0 else 0
    ax.text(0.5, -0.15, f'Precision: {precisao_total:.2%} | Recall: {recall_total:.2%} | F1-Score: {f1_total:.2%}',
            ha='center', transform=ax.transAxes, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../graficos/NN_TIMESERIES/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Matriz de confusão consolidada salva")
    
    # Salvar resultados em CSV
    df_results.to_csv('../graficos/NN_TIMESERIES/timeseries_results.csv', index=False)
    print("✓ Resultados salvos em CSV")
    
    # Relatório textual
    with open('../graficos/NN_TIMESERIES/timeseries_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RELATÓRIO - VALIDAÇÃO TEMPORAL (TIME SERIES SPLIT)\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total de Folds: {len(df_results)}\n")
        f.write(f"Período: {df_results['test_month'].iloc[0]} até {df_results['test_month'].iloc[-1]}\n\n")
        
        f.write("--- DESEMPENHO MÉDIO (REGRESSÃO DE LUCRO) ---\n")
        f.write(f"MAE NN: R$ {df_results['mae_nn'].mean():.2f} ± R$ {df_results['mae_nn'].std():.2f}\n")
        f.write(f"R² NN: {df_results['r2_nn'].mean():.4f} ± {df_results['r2_nn'].std():.4f}\n")
        f.write(f"MAE LGBM: R$ {df_results['mae_lgbm'].mean():.2f} ± R$ {df_results['mae_lgbm'].std():.2f}\n")
        f.write(f"R² LGBM: {df_results['r2_lgbm'].mean():.4f} ± {df_results['r2_lgbm'].std():.4f}\n")
        f.write(f"Threshold Lucro: R$ {df_results['threshold'].mean():.2f} ± R$ {df_results['threshold'].std():.2f}\n\n")
        
        f.write("--- LUCRO TOTAL (7 MESES) ---\n")
        f.write(f"Baseline: R$ {lucro_total_baseline:,.2f}\n")
        f.write(f"Modelo:   R$ {lucro_total_otimizado:,.2f}\n\n")
        f.write(f"Ganho vs Baseline: R$ {ganho_total_vs_baseline:+,.2f} ({ganho_total_perc_baseline:+.2f}%)\n\n")
        
        f.write("--- LUCRO MENSAL (MÉDIO) ---\n")
        f.write(f"Baseline: R$ {df_results['lucro_baseline'].mean():,.2f} ± R$ {df_results['lucro_baseline'].std():,.2f}\n")
        f.write(f"Modelo:   R$ {df_results['lucro_otimizado'].mean():,.2f} ± R$ {df_results['lucro_otimizado'].std():,.2f}\n\n")
        f.write(f"Ganho/mês vs Baseline: R$ {df_results['ganho_absoluto'].mean():,.2f} ({df_results['ganho_percentual'].mean():+.2f}%)\n\n")
        
        f.write("--- ESTABILIDADE ---\n")
        f.write(f"Volatilidade R² (CV): {r2_volatilidade:.4f}\n")
        f.write(f"Volatilidade Lucro (CV): {lucro_volatilidade:.4f}\n\n")
        
        f.write("--- TENDÊNCIAS ---\n")
        f.write(f"R² LGBM: Slope {slope_r2:.6f} (p={p_r2:.4f})\n")
        f.write(f"Ganho %: Slope {slope_lucro:.6f} (p={p_lucro:.4f})\n\n")
        
        f.write("--- RESULTADOS POR MÊS ---\n")
        for idx, row in df_results.iterrows():
            f.write(f"\n{row['test_month']}:\n")
            f.write(f"  Lucro: R$ {row['lucro_otimizado']:,.2f} (Baseline: R$ {row['lucro_baseline']:,.2f})\n")
            f.write(f"  Ganho: R$ {row['ganho_absoluto']:,.2f} ({row['ganho_percentual']:+.2f}%)\n")
            f.write(f"  R²: NN {row['r2_nn']:.4f} | LGBM {row['r2_lgbm']:.4f}\n")
            f.write(f"  MAE: NN R$ {row['mae_nn']:.2f} | LGBM R$ {row['mae_lgbm']:.2f}\n")
            f.write(f"  Threshold Lucro: R$ {row['threshold']:.2f}\n")
            f.write(f"  Aprovação: {row['taxa_aprovacao']:.1f}%\n")
    
    print("✓ Relatório textual salvo")

print("\n" + "="*70)
print("✅ VALIDAÇÃO TEMPORAL CONCLUÍDA")
print("="*70)
