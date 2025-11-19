"""
MODELO HURDLE: CLASSIFICAÇÃO + REGRESSÃO CONDICIONAL
====================================================

Estratégia em dois estágios:
1. Modelo A (Classificação): Probabilidade de pagamento (1 - default)
2. Modelo B (Regressão condicional): Valor pago (apenas clientes lucrativos)
Lucro esperado = P(pagar) * valor_esperado    -   P(default) * custo_médio_default

Estrutura de saída replica `inadimplente_ENSEMBLE_PARETO_FINAL.py`.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import xgboost as xgb
import lightgbm as lgbm
from catboost import CatBoostClassifier, CatBoostRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

import optuna
from optuna.samplers import NSGAIISampler

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_and_preprocess_v3 import load_and_preprocess_v3

# Configurações TensorFlow (CPU-friendly)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

print("="*80)
print("🏆 MODELO HURDLE: CLASSIFICAÇÃO + REGRESSÃO CONDICIONAL")
print("="*80)

# --------------------------------------
# 1. CARREGAMENTO E PREPARAÇÃO
# --------------------------------------
DATA_FILE = '../data/dataset_interno_top_one_atualizado.csv'
df = load_and_preprocess_v3(DATA_FILE)
print(f"\n📊 Dados carregados: {len(df)} contratos")

# Targets
y_lucro = df['lucro'].values.astype(float)
y_default = df['default'].values.astype(int)
y_pagador = 1 - y_default

# Engenharia temporal
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

COLS_REMOVE = ['default', 'pago_perc', 'lucro', 'aceitar', 'adimplente',
               'contrato_id', 'proposta_id', 'Unnamed: 0',
               'Data_Lancamento', 'Data_inicio']

X = df.drop(columns=[c for c in COLS_REMOVE if c in df.columns])
print(f"\n🔧 Features antes do encoding: {X.shape[1]}")
X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.median())
print(f"Features após encoding: {X.shape[1]}")

pago_perc_values = df['pago_perc'].values.astype(float)

X_train, X_test, y_train_lucro, y_test_lucro, y_train_default, y_test_default, y_train_pagador, y_test_pagador, pago_perc_train, pago_perc_test = train_test_split(
    X,
    y_lucro,
    y_default,
    y_pagador,
    pago_perc_values,
    test_size=0.2,
    random_state=42
)

print("\n📊 Split 80/20...")
print(f"Treino: {len(X_train):,} contratos")
print(f"Teste: {len(X_test):,} contratos")

feature_names = X.columns.tolist()
X_train_matrix = X_train.values.astype(np.float32)
X_test_matrix = X_test.values.astype(np.float32)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_matrix)
X_test_scaled = scaler.transform(X_test_matrix)

custo_default_medio = np.abs(df[df['lucro'] < 0]['lucro'].mean())
custo_default_medio = custo_default_medio if not np.isnan(custo_default_medio) else 100.0
print(f"Custo médio estimado de default: R$ {custo_default_medio:.2f}")

# --------------------------------------
# 2. DEFINIÇÃO DAS ARQUITETURAS
# --------------------------------------

def create_nn_classifier(input_dim, hidden_dim=128, dropout=0.3):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.LayerNormalization()(inputs)
    x = layers.Dense(hidden_dim, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(hidden_dim//2, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(dropout)(x)

    embeddings = layers.Dense(hidden_dim//2, activation='linear', name='emb_clf')(x)
    x = layers.Activation('swish')(embeddings)
    output = layers.Dense(1, activation='sigmoid', name='prob_pagador')(x)

    model = keras.Model(inputs, output)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    embedding_model = keras.Model(inputs=inputs, outputs=embeddings)
    return model, embedding_model


def create_nn_regressor(input_dim, hidden_dim=128, dropout=0.3):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.LayerNormalization()(inputs)
    x = layers.Dense(hidden_dim, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(hidden_dim//2, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(dropout)(x)

    embeddings = layers.Dense(hidden_dim//2, activation='linear', name='emb_reg')(x)
    x = layers.Activation('swish')(embeddings)
    x = layers.Dropout(dropout/1.5)(x)
    output = layers.Dense(1, activation='linear', name='lucro_condicional')(x)

    model = keras.Model(inputs, output)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.0015, weight_decay=1e-4),
        loss=keras.losses.Huber(delta=200.0),
        metrics=['mae']
    )
    embedding_model = keras.Model(inputs=inputs, outputs=embeddings)
    return model, embedding_model


kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --------------------------------------
# 3. MODELO A - CLASSIFICAÇÃO
# --------------------------------------
print("\n" + "="*80)
print("🤖 MODELO A: PROBABILIDADE DE PAGAMENTO")
print("="*80)

# 3.1 Neural Network Classifier
print("1️⃣ NN Classifier (OOF + embeddings)...")

tmp_clf, tmp_embed = create_nn_classifier(X_train_scaled.shape[1])
emb_dim_clf = tmp_embed.output_shape[-1]
del tmp_clf, tmp_embed

oof_nn_clf = np.zeros(len(X_train_scaled))
embeddings_clf_oof = np.zeros((len(X_train_scaled), emb_dim_clf), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_scaled), 1):
    nn_clf, emb_clf = create_nn_classifier(X_train_scaled.shape[1])
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-5)
    ]
    nn_clf.fit(
        X_train_scaled[tr_idx], y_train_pagador[tr_idx],
        validation_data=(X_train_scaled[va_idx], y_train_pagador[va_idx]),
        epochs=250,
        batch_size=128,
        callbacks=cb,
        verbose=0
    )
    oof_nn_clf[va_idx] = nn_clf.predict(X_train_scaled[va_idx], verbose=0).flatten()
    embeddings_clf_oof[va_idx] = emb_clf.predict(X_train_scaled[va_idx], verbose=0)
    auc_fold = roc_auc_score(y_train_pagador[va_idx], oof_nn_clf[va_idx])
    print(f"   • Fold {fold}: AUC = {auc_fold:.4f}")

nn_clf_full, emb_clf_full = create_nn_classifier(X_train_scaled.shape[1])
nn_cb = [
    callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-5)
]
nn_clf_full.fit(
    X_train_scaled, y_train_pagador,
    validation_split=0.15,
    epochs=300,
    batch_size=128,
    callbacks=nn_cb,
    verbose=0
)
pred_nn_clf_train = nn_clf_full.predict(X_train_scaled, verbose=0).flatten()
pred_nn_clf_test = nn_clf_full.predict(X_test_scaled, verbose=0).flatten()
embeddings_clf_train_full = emb_clf_full.predict(X_train_scaled, verbose=0)
embeddings_clf_test = emb_clf_full.predict(X_test_scaled, verbose=0)

auc_nn = roc_auc_score(y_test_pagador, pred_nn_clf_test)
print(f"   ✓ NN Classifier: AUC = {auc_nn:.4f}")

# 3.2 LightGBM Classifier
print("2️⃣ LightGBM Classifier...")

lgb_clf_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.02,
    'n_estimators': 3000,
    'num_leaves': 120,
    'min_child_samples': 90,
    'subsample': 0.85,
    'subsample_freq': 5,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.3,
    'reg_lambda': 1.2,
    'random_state': 42,
    'n_jobs': -1
}

oof_lgb_clf = np.zeros(len(X_train_matrix))
lgb_clf_iters = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_matrix), 1):
    model_lgb_clf = lgbm.LGBMClassifier(**lgb_clf_params)
    model_lgb_clf.fit(
        X_train_matrix[tr_idx], y_train_pagador[tr_idx],
        eval_set=[(X_train_matrix[va_idx], y_train_pagador[va_idx])],
        eval_metric='auc',
        callbacks=[lgbm.early_stopping(200, verbose=False)]
    )
    oof_lgb_clf[va_idx] = model_lgb_clf.predict_proba(X_train_matrix[va_idx])[:, 1]
    best_iter = model_lgb_clf.best_iteration_ or lgb_clf_params['n_estimators']
    lgb_clf_iters.append(best_iter)
    auc_fold = roc_auc_score(y_train_pagador[va_idx], oof_lgb_clf[va_idx])
    print(f"   • Fold {fold}: AUC = {auc_fold:.4f} (iters={best_iter})")

avg_iter = int(np.mean(lgb_clf_iters)) if lgb_clf_iters else lgb_clf_params['n_estimators']
final_lgb_clf = lgbm.LGBMClassifier(**{**lgb_clf_params, 'n_estimators': max(avg_iter, 300)})
final_lgb_clf.fit(X_train_matrix, y_train_pagador)
pred_lgb_clf_train = final_lgb_clf.predict_proba(X_train_matrix)[:, 1]
pred_lgb_clf_test = final_lgb_clf.predict_proba(X_test_matrix)[:, 1]
auc_lgb = roc_auc_score(y_test_pagador, pred_lgb_clf_test)
print(f"   ✓ LGBM Classifier: AUC = {auc_lgb:.4f}")

# 3.3 XGBoost Classifier
print("3️⃣ XGBoost Classifier...")

xgb_clf_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.02,
    'n_estimators': 3500,
    'max_depth': 7,
    'subsample': 0.85,
    'colsample_bytree': 0.7,
    'min_child_weight': 20,
    'gamma': 0.2,
    'reg_alpha': 0.2,
    'reg_lambda': 1.0,
    'n_jobs': 4,
    'tree_method': 'hist',
    'random_state': 42
}

oof_xgb_clf = np.zeros(len(X_train_matrix))
xgb_clf_iters = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_matrix), 1):
    model_xgb_clf = xgb.XGBClassifier(**xgb_clf_params, early_stopping_rounds=200)
    model_xgb_clf.fit(
        X_train_matrix[tr_idx], y_train_pagador[tr_idx],
        eval_set=[(X_train_matrix[va_idx], y_train_pagador[va_idx])],
        verbose=False
    )
    oof_xgb_clf[va_idx] = model_xgb_clf.predict_proba(X_train_matrix[va_idx])[:, 1]
    best_iter = getattr(model_xgb_clf, 'best_iteration', None) or xgb_clf_params['n_estimators']
    xgb_clf_iters.append(best_iter)
    auc_fold = roc_auc_score(y_train_pagador[va_idx], oof_xgb_clf[va_idx])
    print(f"   • Fold {fold}: AUC = {auc_fold:.4f} (iters={best_iter})")

avg_xgb_clf_iter = int(np.mean(xgb_clf_iters)) if xgb_clf_iters else xgb_clf_params['n_estimators']
final_xgb_clf = xgb.XGBClassifier(**{**xgb_clf_params, 'n_estimators': max(avg_xgb_clf_iter, 400)})
final_xgb_clf.fit(X_train_matrix, y_train_pagador, verbose=False)
pred_xgb_clf_train = final_xgb_clf.predict_proba(X_train_matrix)[:, 1]
pred_xgb_clf_test = final_xgb_clf.predict_proba(X_test_matrix)[:, 1]
auc_xgb = roc_auc_score(y_test_pagador, pred_xgb_clf_test)
print(f"   ✓ XGB Classifier: AUC = {auc_xgb:.4f}")

# 3.4 CatBoost Classifier
print("4️⃣ CatBoost Classifier...")

oof_cat_clf = np.zeros(len(X_train_matrix))
cat_clf_iters = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_matrix), 1):
    model_cat_clf = CatBoostClassifier(
        depth=7,
        learning_rate=0.03,
        iterations=3000,
        loss_function='Logloss',
        eval_metric='AUC',
        l2_leaf_reg=4.0,
        random_seed=42,
        verbose=False
    )
    model_cat_clf.fit(
        X_train_matrix[tr_idx], y_train_pagador[tr_idx],
        eval_set=(X_train_matrix[va_idx], y_train_pagador[va_idx]),
        use_best_model=True,
        verbose=False
    )
    oof_cat_clf[va_idx] = model_cat_clf.predict_proba(X_train_matrix[va_idx])[:, 1]
    best_iter = model_cat_clf.get_best_iteration() or 3000
    cat_clf_iters.append(best_iter)
    auc_fold = roc_auc_score(y_train_pagador[va_idx], oof_cat_clf[va_idx])
    print(f"   • Fold {fold}: AUC = {auc_fold:.4f} (iters={best_iter})")

avg_cat_clf_iter = int(np.mean(cat_clf_iters)) if cat_clf_iters else 3000
cat_clf_final = CatBoostClassifier(
    depth=7,
    learning_rate=0.03,
    iterations=avg_cat_clf_iter,
    loss_function='Logloss',
    eval_metric='AUC',
    l2_leaf_reg=4.0,
    random_seed=42,
    verbose=False
)
cat_clf_final.fit(X_train_matrix, y_train_pagador, verbose=False)
pred_cat_clf_train = cat_clf_final.predict_proba(X_train_matrix)[:, 1]
pred_cat_clf_test = cat_clf_final.predict_proba(X_test_matrix)[:, 1]
auc_cat = roc_auc_score(y_test_pagador, pred_cat_clf_test)
print(f"   ✓ CatBoost Classifier: AUC = {auc_cat:.4f}")

# Meta features - classificação
feature_stds = X_train_matrix.std(axis=0)
top_features_idx = np.argsort(feature_stds)[-10:]

meta_clf_train = np.column_stack([
    oof_nn_clf,
    oof_lgb_clf,
    oof_xgb_clf,
    oof_cat_clf,
    embeddings_clf_oof,
    X_train_scaled[:, top_features_idx]
])

meta_clf_test = np.column_stack([
    pred_nn_clf_test,
    pred_lgb_clf_test,
    pred_xgb_clf_test,
    pred_cat_clf_test,
    embeddings_clf_test,
    X_test_scaled[:, top_features_idx]
])

meta_clf_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.01,
    'n_estimators': 3000,
    'max_depth': 4,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'min_child_weight': 8,
    'gamma': 0.1,
    'reg_alpha': 0.2,
    'reg_lambda': 1.0,
    'tree_method': 'hist',
    'n_jobs': 4,
    'random_state': 42
}

meta_clf_X_train, meta_clf_X_val, meta_clf_y_train, meta_clf_y_val = train_test_split(
    meta_clf_train,
    y_train_pagador,
    test_size=0.15,
    random_state=42
)

meta_clf_model_cv = xgb.XGBClassifier(**meta_clf_params, early_stopping_rounds=300)
meta_clf_model_cv.fit(
    meta_clf_X_train,
    meta_clf_y_train,
    eval_set=[(meta_clf_X_val, meta_clf_y_val)],
    verbose=False
)
best_meta_clf_iter = getattr(meta_clf_model_cv, 'best_iteration', None) or meta_clf_params['n_estimators']
meta_clf_model = xgb.XGBClassifier(**{**meta_clf_params, 'n_estimators': best_meta_clf_iter})
meta_clf_model.fit(meta_clf_train, y_train_pagador, verbose=False)

prob_pay_train = meta_clf_model.predict_proba(meta_clf_train)[:, 1]
prob_pay_test = meta_clf_model.predict_proba(meta_clf_test)[:, 1]
prob_default_train = 1 - prob_pay_train
prob_default_test = 1 - prob_pay_test

auc_meta_clf = roc_auc_score(y_test_pagador, prob_pay_test)
print(f"   ✓ Meta-Classifier: AUC = {auc_meta_clf:.4f}")

# --------------------------------------
# 4. MODELO B - REGRESSÃO CONDICIONAL
# --------------------------------------
print("\n" + "="*80)
print("💰 MODELO B: VALOR CONDICIONAL PAGO")
print("="*80)

mask_pos_train = y_train_lucro > 0
mask_pos_test = y_test_lucro > 0

X_train_pos_scaled = X_train_scaled[mask_pos_train]
y_train_pos = y_train_lucro[mask_pos_train]

X_train_pos_matrix = X_train_matrix[mask_pos_train]
X_test_pos_matrix = X_test_matrix

# 4.1 NN Regressor
print("1️⃣ NN Regressor...")

tmp_reg, tmp_reg_emb = create_nn_regressor(X_train_scaled.shape[1])
emb_dim_reg = tmp_reg_emb.output_shape[-1]
del tmp_reg, tmp_reg_emb

oof_nn_reg = np.zeros(len(y_train_pos))
embeddings_reg_oof = np.zeros((len(y_train_pos), emb_dim_reg), dtype=np.float32)

pos_indices = np.where(mask_pos_train)[0]
kf_reg = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, va_idx) in enumerate(kf_reg.split(X_train_pos_scaled), 1):
    nn_reg, emb_reg = create_nn_regressor(X_train_scaled.shape[1])
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-5)
    ]
    nn_reg.fit(
        X_train_pos_scaled[tr_idx], y_train_pos[tr_idx],
        validation_data=(X_train_pos_scaled[va_idx], y_train_pos[va_idx]),
        epochs=250,
        batch_size=128,
        callbacks=cb,
        verbose=0
    )
    oof_nn_reg[va_idx] = nn_reg.predict(X_train_pos_scaled[va_idx], verbose=0).flatten()
    embeddings_reg_oof[va_idx] = emb_reg.predict(X_train_pos_scaled[va_idx], verbose=0)
    mae_fold = mean_absolute_error(y_train_pos[va_idx], oof_nn_reg[va_idx])
    print(f"   • Fold {fold}: MAE = R$ {mae_fold:.2f}")

nn_reg_full, emb_reg_full = create_nn_regressor(X_train_scaled.shape[1])
nn_reg_cb = [
    callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=18, min_lr=1e-5)
]
nn_reg_full.fit(
    X_train_pos_scaled,
    y_train_pos,
    validation_split=0.2,
    epochs=320,
    batch_size=128,
    callbacks=nn_reg_cb,
    verbose=0
)

pred_nn_reg_train = nn_reg_full.predict(X_train_scaled, verbose=0).flatten()
pred_nn_reg_test = nn_reg_full.predict(X_test_scaled, verbose=0).flatten()
embeddings_reg_train_full = emb_reg_full.predict(X_train_scaled, verbose=0)
embeddings_reg_test = emb_reg_full.predict(X_test_scaled, verbose=0)

# 4.2 LightGBM Regressor
print("2️⃣ LightGBM Regressor...")

lgb_reg_params = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.02,
    'n_estimators': 4000,
    'num_leaves': 140,
    'min_child_samples': 60,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'reg_alpha': 0.3,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1
}

oof_lgb_reg = np.zeros(len(y_train_pos))
lgb_reg_iters = []

for fold, (tr_idx, va_idx) in enumerate(kf_reg.split(X_train_pos_matrix), 1):
    model_lgb_reg = lgbm.LGBMRegressor(**lgb_reg_params)
    model_lgb_reg.fit(
        X_train_pos_matrix[tr_idx], y_train_pos[tr_idx],
        eval_set=[(X_train_pos_matrix[va_idx], y_train_pos[va_idx])],
        eval_metric='mae',
        callbacks=[lgbm.early_stopping(200, verbose=False)]
    )
    oof_lgb_reg[va_idx] = model_lgb_reg.predict(X_train_pos_matrix[va_idx])
    best_iter = model_lgb_reg.best_iteration_ or lgb_reg_params['n_estimators']
    lgb_reg_iters.append(best_iter)
    mae_fold = mean_absolute_error(y_train_pos[va_idx], oof_lgb_reg[va_idx])
    print(f"   • Fold {fold}: MAE = R$ {mae_fold:.2f} (iters={best_iter})")

avg_lgb_reg_iter = int(np.mean(lgb_reg_iters)) if lgb_reg_iters else lgb_reg_params['n_estimators']
final_lgb_reg = lgbm.LGBMRegressor(**{**lgb_reg_params, 'n_estimators': max(avg_lgb_reg_iter, 400)})
final_lgb_reg.fit(X_train_pos_matrix, y_train_pos)
pred_lgb_reg_train = final_lgb_reg.predict(X_train_matrix)
pred_lgb_reg_test = final_lgb_reg.predict(X_test_matrix)

# 4.3 XGBoost Regressor
print("3️⃣ XGBoost Regressor...")

xgb_reg_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.02,
    'n_estimators': 3500,
    'max_depth': 7,
    'subsample': 0.85,
    'colsample_bytree': 0.7,
    'min_child_weight': 15,
    'gamma': 0.2,
    'reg_alpha': 0.2,
    'reg_lambda': 1.2,
    'tree_method': 'hist',
    'n_jobs': 4,
    'random_state': 42
}

oof_xgb_reg = np.zeros(len(y_train_pos))
xgb_reg_iters = []

for fold, (tr_idx, va_idx) in enumerate(kf_reg.split(X_train_pos_matrix), 1):
    model_xgb_reg = xgb.XGBRegressor(**xgb_reg_params, early_stopping_rounds=200)
    model_xgb_reg.fit(
        X_train_pos_matrix[tr_idx], y_train_pos[tr_idx],
        eval_set=[(X_train_pos_matrix[va_idx], y_train_pos[va_idx])],
        verbose=False
    )
    oof_xgb_reg[va_idx] = model_xgb_reg.predict(X_train_pos_matrix[va_idx])
    best_iter = getattr(model_xgb_reg, 'best_iteration', None) or xgb_reg_params['n_estimators']
    xgb_reg_iters.append(best_iter)
    mae_fold = mean_absolute_error(y_train_pos[va_idx], oof_xgb_reg[va_idx])
    print(f"   • Fold {fold}: MAE = R$ {mae_fold:.2f} (iters={best_iter})")

avg_xgb_reg_iter = int(np.mean(xgb_reg_iters)) if xgb_reg_iters else xgb_reg_params['n_estimators']
final_xgb_reg = xgb.XGBRegressor(**{**xgb_reg_params, 'n_estimators': max(avg_xgb_reg_iter, 400)})
final_xgb_reg.fit(X_train_pos_matrix, y_train_pos, verbose=False)
pred_xgb_reg_train = final_xgb_reg.predict(X_train_matrix)
pred_xgb_reg_test = final_xgb_reg.predict(X_test_matrix)

# 4.4 CatBoost Regressor
print("4️⃣ CatBoost Regressor...")

oof_cat_reg = np.zeros(len(y_train_pos))
cat_reg_iters = []

for fold, (tr_idx, va_idx) in enumerate(kf_reg.split(X_train_pos_matrix), 1):
    model_cat_reg = CatBoostRegressor(
        depth=8,
        learning_rate=0.03,
        iterations=4000,
        loss_function='MAE',
        l2_leaf_reg=5.0,
        random_state=42,
        verbose=False
    )
    model_cat_reg.fit(
        X_train_pos_matrix[tr_idx], y_train_pos[tr_idx],
        eval_set=(X_train_pos_matrix[va_idx], y_train_pos[va_idx]),
        use_best_model=True,
        verbose=False
    )
    oof_cat_reg[va_idx] = model_cat_reg.predict(X_train_pos_matrix[va_idx])
    best_iter = model_cat_reg.get_best_iteration() or 4000
    cat_reg_iters.append(best_iter)
    mae_fold = mean_absolute_error(y_train_pos[va_idx], oof_cat_reg[va_idx])
    print(f"   • Fold {fold}: MAE = R$ {mae_fold:.2f} (iters={best_iter})")

avg_cat_reg_iter = int(np.mean(cat_reg_iters)) if cat_reg_iters else 4000
final_cat_reg = CatBoostRegressor(
    depth=8,
    learning_rate=0.03,
    iterations=avg_cat_reg_iter,
    loss_function='MAE',
    l2_leaf_reg=5.0,
    random_state=42,
    verbose=False
)
final_cat_reg.fit(X_train_pos_matrix, y_train_pos, verbose=False)
pred_cat_reg_train = final_cat_reg.predict(X_train_matrix)
pred_cat_reg_test = final_cat_reg.predict(X_test_matrix)

# Meta regressor features (treinados apenas em positivos)
meta_reg_train = np.column_stack([
    oof_nn_reg,
    oof_lgb_reg,
    oof_xgb_reg,
    oof_cat_reg,
    embeddings_reg_oof,
    X_train_pos_scaled[:, top_features_idx]
])

meta_reg_test_full = np.column_stack([
    pred_nn_reg_test,
    pred_lgb_reg_test,
    pred_xgb_reg_test,
    pred_cat_reg_test,
    embeddings_reg_test,
    X_test_scaled[:, top_features_idx]
])

meta_reg_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'n_estimators': 2500,
    'max_depth': 4,
    'min_child_weight': 10,
    'subsample': 0.85,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.2,
    'reg_lambda': 1.0,
    'gamma': 0.1,
    'tree_method': 'hist',
    'n_jobs': 4,
    'random_state': 42
}

meta_reg_X_train, meta_reg_X_val, meta_reg_y_train, meta_reg_y_val = train_test_split(
    meta_reg_train,
    y_train_pos,
    test_size=0.15,
    random_state=42
)

meta_reg_model_cv = xgb.XGBRegressor(**meta_reg_params, early_stopping_rounds=250)
meta_reg_model_cv.fit(
    meta_reg_X_train,
    meta_reg_y_train,
    eval_set=[(meta_reg_X_val, meta_reg_y_val)],
    verbose=False
)
best_meta_reg_iter = getattr(meta_reg_model_cv, 'best_iteration', None) or meta_reg_params['n_estimators']
meta_reg_model = xgb.XGBRegressor(**{**meta_reg_params, 'n_estimators': best_meta_reg_iter})
meta_reg_model.fit(meta_reg_train, y_train_pos, verbose=False)

# Para previsão no conjunto completo, concatenamos as features equivalentes
meta_reg_features_train_full = np.column_stack([
    pred_nn_reg_train,
    pred_lgb_reg_train,
    pred_xgb_reg_train,
    pred_cat_reg_train,
    embeddings_reg_train_full,
    X_train_scaled[:, top_features_idx]
])

pred_amount_train = meta_reg_model.predict(meta_reg_features_train_full)
pred_amount_test = meta_reg_model.predict(meta_reg_test_full)

pred_amount_train = np.clip(pred_amount_train, 0, None)
pred_amount_test = np.clip(pred_amount_test, 0, None)

# --------------------------------------
# 5. COMBINAÇÃO HURDLE
# --------------------------------------
expected_profit_train = prob_pay_train * pred_amount_train - prob_default_train * custo_default_medio
expected_profit_test = prob_pay_test * pred_amount_test - prob_default_test * custo_default_medio

mae_hurdle = mean_absolute_error(y_test_lucro, expected_profit_test)
r2_hurdle = r2_score(y_test_lucro, expected_profit_test)
print("\n" + "="*80)
print("🎯 HURDLE FINAL - MÉTRICAS")
print("="*80)
print(f"MAE = R$ {mae_hurdle:.2f}")
print(f"R²  = {r2_hurdle:.4f}")

# --------------------------------------
# 6. OTIMIZAÇÃO PARETO
# --------------------------------------
print("\n" + "="*80)
print("🔄 OTIMIZAÇÃO MULTI-OBJETIVO (PARETO)")
print("="*80)

pred_meta_test = expected_profit_test

def objective_pareto(trial):
    threshold = trial.suggest_float('threshold', pred_meta_test.min(), pred_meta_test.max())
    aceitar = pred_meta_test >= threshold
    if aceitar.sum() == 0:
        return -1e9, 1.0
    lucro_total = y_test_lucro[aceitar].sum()
    inadimplentes_aceitos = y_test_default[aceitar].sum()
    fdr = inadimplentes_aceitos / aceitar.sum()
    return lucro_total, fdr

study = optuna.create_study(
    directions=["maximize", "minimize"],
    sampler=NSGAIISampler(seed=42)
)
study.optimize(objective_pareto, n_trials=400, show_progress_bar=False)

pareto_trials = study.best_trials
print(f"   ✓ Pareto Frontier: {len(pareto_trials)} soluções")

pareto_scores = []
for trial in pareto_trials:
    lucro_vals = [t.values[0] for t in pareto_trials]
    fdr_vals = [t.values[1] for t in pareto_trials]
    lucro_norm = (trial.values[0] - min(lucro_vals)) / (max(lucro_vals) - min(lucro_vals) + 1e-6)
    fdr_norm = 1 - (trial.values[1] - min(fdr_vals)) / (max(fdr_vals) - min(fdr_vals) + 1e-6)
    score = 0.7 * lucro_norm + 0.3 * fdr_norm
    pareto_scores.append(score)

best_idx = int(np.argmax(pareto_scores))
best_trial = pareto_trials[best_idx]
best_threshold = best_trial.params['threshold']
best_lucro = best_trial.values[0]
best_fdr = best_trial.values[1]

print(f"   🎯 Threshold selecionado: R$ {best_threshold:.2f}")
print(f"   Lucro: R$ {best_lucro:,.2f}")
print(f"   FDR: {best_fdr*100:.2f}%")

aceitar_final = pred_meta_test >= best_threshold
num_aceitos = aceitar_final.sum()
taxa_aceitacao = num_aceitos / len(y_test_lucro) * 100

lucro_otimizado = y_test_lucro[aceitar_final].sum()
lucro_baseline = y_test_lucro.sum()
lucro_maximo = y_test_lucro[y_test_lucro > 0].sum()

ganho_vs_baseline = lucro_otimizado - lucro_baseline
ganho_perc = (ganho_vs_baseline / lucro_baseline * 100) if lucro_baseline != 0 else 0

# --------------------------------------
# 7. RELATÓRIO FINAL
# --------------------------------------
print("\n" + "="*80)
print("📊 RESULTADOS FINAIS")
print("="*80)
print(f"AUC ProbPagamento (Meta): {auc_meta_clf:.4f}")
print(f"MAE Valor Condicional (Meta): R$ {mean_absolute_error(y_train_pos, meta_reg_model.predict(meta_reg_train)):.2f}")
print(f"HURDLE - MAE: R$ {mae_hurdle:.2f} | R²: {r2_hurdle:.4f}")
print("\n--- Comparação de Cenários ---")
print(f"{'Cenário':<45} {'Lucro':>15} {'Eficiência':>12} {'Contratos':>12}")
print("="*80)
print(f"{'1. Baseline (aceitar todos)':<45} R$ {lucro_baseline:>12,.2f} {lucro_baseline/lucro_maximo:>11.2%} {len(y_test_lucro):>11,}")
print(f"{'2. Máximo Teórico (lucro>0)':<45} R$ {lucro_maximo:>12,.2f} {100.0:>11.2%} {(y_test_lucro>0).sum():>11,}")
print(f"{'3. HURDLE + PARETO (otimizado)':<45} R$ {lucro_otimizado:>12,.2f} {lucro_otimizado/lucro_maximo:>11.2%} {num_aceitos:>11,}")
print("="*80)
print(f"💰 Ganho vs Baseline: R$ {ganho_vs_baseline:+,.2f} ({ganho_perc:+.2f}%)")
print(f"📊 Taxa de Aprovação: {taxa_aceitacao:.1f}%")
print(f"📉 FDR: {best_fdr*100:.2f}%")

# Matriz de confusão
if hasattr(y_test_default, 'values'):
    y_true_default = y_test_default
else:
    y_true_default = np.array(y_test_default)

y_pred_rejeitar = (pred_meta_test < best_threshold).astype(int)
cm = confusion_matrix(y_true_default, y_pred_rejeitar)
tn, fp, fn, tp = cm.ravel()
precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0

# --------------------------------------
# 8. VISUALIZAÇÕES
# --------------------------------------
print("\n--- Gerando Visualizações ---")
os.makedirs('../graficos/HURDLE_MODEL', exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. AUC dos classificadores
ax1 = axes[0, 0]
clf_models = ['NN', 'LGBM', 'XGB', 'CAT', 'META']
auc_values = [auc_nn, auc_lgb, auc_xgb, auc_cat, auc_meta_clf]
ax1.bar(clf_models, auc_values, color=['blue', 'green', 'red', 'orange', 'purple'], alpha=0.7)
ax1.set_ylabel('AUC')
ax1.set_title('Comparação AUC - Classificadores')
ax1.grid(alpha=0.3, axis='y')
for i, v in enumerate(auc_values):
    ax1.text(i, v + 0.005, f"{v:.3f}", ha='center', fontweight='bold')

# 2. MAE regressão
ax2 = axes[0, 1]
reg_models = ['NN', 'LGBM', 'XGB', 'CAT']
mae_reg_values = [
    mean_absolute_error(y_train_pos, oof_nn_reg),
    mean_absolute_error(y_train_pos, oof_lgb_reg),
    mean_absolute_error(y_train_pos, oof_xgb_reg),
    mean_absolute_error(y_train_pos, oof_cat_reg)
]
ax2.bar(reg_models, mae_reg_values, color=['blue', 'green', 'red', 'orange'], alpha=0.7)
ax2.set_ylabel('MAE (R$)')
ax2.set_title('MAE Condicional (OOF)')
ax2.grid(alpha=0.3, axis='y')
for i, v in enumerate(mae_reg_values):
    ax2.text(i, v + 5, f"R$ {v:.0f}", ha='center', fontweight='bold')

# 3. Matriz de confusão
aq3 = axes[0, 2]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=aq3, cbar=True,
            xticklabels=['Aceito', 'Rejeitado'],
            yticklabels=['Adimplente', 'Inadimplente'])
aq3.set_title('Matriz de Confusão (Hurdle)')
aq3.set_xlabel('Predição do Modelo')
aq3.set_ylabel('Realidade')
aq3.text(0.5, -0.15, f'Precision: {precisao:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}',
         transform=aq3.transAxes, ha='center', fontsize=10, fontweight='bold')

# 4. Pareto
ax4 = axes[1, 0]
lucros = [trial.values[0] for trial in pareto_trials]
fdrs = [trial.values[1] * 100 for trial in pareto_trials]
ax4.scatter(fdrs, lucros, c='purple', alpha=0.6)
ax4.scatter([best_fdr*100], [best_lucro], c='red', s=150, marker='*', label='Selecionado')
ax4.set_xlabel('FDR (%)')
ax4.set_ylabel('Lucro Total (R$)')
ax4.set_title('Pareto Frontier')
ax4.grid(alpha=0.3)
ax4.legend()

# 5. Cenários
ax5 = axes[1, 1]
cenarios = ['Baseline', 'Máx. Teórico', 'HURDLE+PARETO']
lucros_cenarios = [lucro_baseline, lucro_maximo, lucro_otimizado]
ax5.bar(cenarios, lucros_cenarios, color=['gray', 'lightblue', 'purple'], alpha=0.7)
ax5.set_ylabel('Lucro (R$)')
ax5.set_title('Comparação de Cenários')
ax5.grid(alpha=0.3, axis='y')
for i, val in enumerate(lucros_cenarios):
    ax5.text(i, val + max(lucros_cenarios)*0.02, f"R$ {val:,.0f}", ha='center', fontweight='bold')

# 6. Distribuição decisões
ax6 = axes[1, 2]
decisoes = ['Aceitos', 'Rejeitados']
contagens = [aceitar_final.sum(), (~aceitar_final).sum()]
ax6.pie(contagens, labels=decisoes, autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
ax6.set_title('Distribuição de Decisões')

plt.tight_layout()
plt.savefig('../graficos/HURDLE_MODEL/analise_hurdle.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráficos salvos em graficos/HURDLE_MODEL")

# Salvar matriz separada
fig_cm = plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Aceito', 'Rejeitado'],
            yticklabels=['Adimplente', 'Inadimplente'])
plt.title('Matriz de Confusão - Modelo Hurdle')
plt.tight_layout()
plt.savefig('../graficos/HURDLE_MODEL/matriz_confusao.png', dpi=300, bbox_inches='tight')
plt.close(fig_cm)

# Resumo
with open('../graficos/HURDLE_MODEL/resumo.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("MODELO HURDLE: CLASSIFICAÇÃO + REGRESSÃO\n")
    f.write("="*80 + "\n\n")
    f.write(f"Treino/Teste: 80/20 | Contratos teste: {len(y_test_lucro):,}\n\n")
    f.write("--- MODELO A (Classificação) ---\n")
    f.write(f"NN AUC: {auc_nn:.4f}\n")
    f.write(f"LGBM AUC: {auc_lgb:.4f}\n")
    f.write(f"XGB AUC: {auc_xgb:.4f}\n")
    f.write(f"CAT AUC: {auc_cat:.4f}\n")
    f.write(f"META AUC: {auc_meta_clf:.4f}\n\n")
    f.write("--- MODELO B (Regressão Condicional) ---\n")
    f.write(f"NN MAE: R$ {mean_absolute_error(y_train_pos, oof_nn_reg):.2f}\n")
    f.write(f"LGBM MAE: R$ {mean_absolute_error(y_train_pos, oof_lgb_reg):.2f}\n")
    f.write(f"XGB MAE: R$ {mean_absolute_error(y_train_pos, oof_xgb_reg):.2f}\n")
    f.write(f"CAT MAE: R$ {mean_absolute_error(y_train_pos, oof_cat_reg):.2f}\n\n")
    f.write("--- COMBINAÇÃO HURDLE ---\n")
    f.write(f"MAE Final: R$ {mae_hurdle:.2f}\n")
    f.write(f"R² Final: {r2_hurdle:.4f}\n")
    f.write(f"Custo médio default: R$ {custo_default_medio:.2f}\n\n")
    f.write("--- PARETO ---\n")
    f.write(f"Threshold: R$ {best_threshold:.2f}\n")
    f.write(f"Lucro otimizado: R$ {lucro_otimizado:,.2f}\n")
    f.write(f"FDR: {best_fdr*100:.2f}%\n")
    f.write(f"Taxa de aprovação: {taxa_aceitacao:.1f}%\n\n")
    f.write("--- MATRIZ DE CONFUSÃO ---\n")
    f.write(f"TN: {tn:,} | FP: {fp:,}\n")
    f.write(f"FN: {fn:,} | TP: {tp:,}\n")
    f.write(f"Precision: {precisao:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}\n")

print("\n" + "="*80)
print("✅ MODELO HURDLE FINALIZADO")
print("="*80)
print(f"HURDLE - Ganho vs baseline: {ganho_perc:+.2f}%")
print("Arquivos em graficos/HURDLE_MODEL/ (analise_hurdle, matriz_confusao, resumo)")
