import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

# Criar diretório para gráficos
os.makedirs('../../graficos/analise_modelos/NN', exist_ok=True)

# --- FASE 0: FUNÇÃO DE PREPARAÇÃO DE DADOS ---
def load_and_preprocess_v3(filepath):
    """
    Carrega e prepara o dataset para modelagem.
    Assume que TODOS os contratos no arquivo são "maduros".
    """
    try:
        df = pd.read_csv(filepath, sep=';', decimal=',') 
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {filepath}")
        return None
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

    print(f"Dados brutos carregados: {df.shape[0]} linhas")

    # --- 1. Limpeza de Tipos ---
    cols_comma_decimal = [
        'valor_inicial_da_prestacao', 'salario_perc', 'lucro', 'IPCA',
        'Score_MC', 'idhm_2010', 'idhm_renda_2010', 
        'idhm_longevidade_2010', 'idhm_educacao_2010',
        'populacao', 'area', 'densidade_pop', 'preco_combustivel',
        'valor_cestabasica', 'preco_cb_perc'
    ]
    
    for col in cols_comma_decimal:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    cols_datetime = ['Data_Lancamento', 'Data_inicio']
    for col in cols_datetime:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # --- 2. Definição da Variável Alvo ---
    if 'pago_perc' not in df.columns:
        print("Erro: A coluna 'pago_perc' é necessária para definir o alvo.")
        return None

    df['pago_perc'] = pd.to_numeric(df['pago_perc'], errors='coerce')
    df_modelagem = df.dropna(subset=['pago_perc']).copy()
    df_modelagem['default'] = (df_modelagem['pago_perc'] < 1).astype(int)
    
    print(f"Dados filtrados (pago_perc preenchido): {df_modelagem.shape[0]} linhas")
    print("Distribuição da variável alvo 'default' (1 = Inadimplente):")
    print(df_modelagem['default'].value_counts(normalize=True))

    # --- 3. Engenharia de Features ---
    if 'Data_Lancamento' in df_modelagem.columns:
        df_modelagem['data_mes'] = df_modelagem['Data_Lancamento'].dt.month
        df_modelagem['data_ano'] = df_modelagem['Data_Lancamento'].dt.year
        df_modelagem['data_dia_semana'] = df_modelagem['Data_Lancamento'].dt.dayofweek

    if 'categorias' in df_modelagem.columns:
        df_modelagem['categoria_limpa'] = df_modelagem['categorias'].astype(str).str.strip("[]'").str.split(',').str[0].str.strip("'")
    else:
        df_modelagem['categoria_limpa'] = 'desconhecido'
        
    df_modelagem['divida_sobre_renda'] = df_modelagem['valor_inicial_da_prestacao'] / (df_modelagem['renda_cliente'] + 1e-6)
    
    # Substituir infinitos e valores muito grandes por NaN
    df_modelagem = df_modelagem.replace([np.inf, -np.inf], np.nan)
    
    if 'divida_sobre_renda' in df_modelagem.columns:
        df_modelagem['divida_sobre_renda'] = df_modelagem['divida_sobre_renda'].clip(upper=100)

    return df_modelagem

# --- FASE 1: FUNÇÃO DE CRIAÇÃO DO MODELO (MLP) ---
def criar_modelo_pd_nn(input_shape):
    """
    Constrói a arquitetura da Rede Neural (MLP).
    """
    inputs = Input(shape=(input_shape,))
    
    # Camadas Ocultas
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.3)(x) # Dropout para prevenir overfitting
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Camada de Saída
    outputs = Dense(1, activation='sigmoid')(x)
    
    modelo = Model(inputs=inputs, outputs=outputs)
    
    # Compilar o modelo
    modelo.compile(
        optimizer='adam',                 
        loss='binary_crossentropy',       # Equivalente a 'logloss'
        metrics=[tf.keras.metrics.AUC(name='auc')] # Monitorar o AUC
    )
    return modelo

# --- FUNÇÃO PRINCIPAL DE EXECUÇÃO ---
def run_pd_model():
    # --- Carregar os dados (Versão 3 - Simplificada) ---
    dataset_maduro = load_and_preprocess_v3('../../data/dataset_interno_top_one_atualizado.csv')

    if dataset_maduro is None:
        print("Falha ao carregar os dados. Encerrando.")
        return

    # --- 1. Definir Features (X) e Alvo (y) ---
    y = dataset_maduro['default']
    
    # Colunas a serem EXCLUÍDAS
    colunas_excluir = [
        'default', 'pago_perc', 'lucro', 'aceitar', # Vazamento de informação do futuro
        'contrato_id', 'proposta_id', 'Unnamed: 0', # IDs
        'Data_Lancamento', 'Data_inicio', #Datas
    ]
    
    X = dataset_maduro.drop(columns=colunas_excluir, errors='ignore')
    
    print("\n--- Iniciando Fase 1: Modelo Preditivo (Rede Neural) ---")
    print(f"Features utilizadas ({X.shape[1]}): {X.columns.tolist()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    
    print(f"\nDetectadas {len(numeric_features)} features numéricas.")
    print(f"Detectadas {len(categorical_features)} features categóricas.")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Processar os dados para a Rede Neural
    print("\nProcessando dados para a Rede Neural...")
    preprocessor.fit(X_train) 
    X_train_nn = preprocessor.transform(X_train)
    X_test_nn = preprocessor.transform(X_test)
    n_features = X_train_nn.shape[1]
    print(f"Número de features de entrada para a NN (pós-processamento): {n_features}")


    # --- 4. Calcular o Peso Financeiro (Sua lógica) ---
    train_indices = X_train.index
    df_train_data = dataset_maduro.loc[train_indices].copy()
    
    if 'lucro' in df_train_data.columns:
        custo_medio_inadimplente = abs(df_train_data[df_train_data['lucro'] < 0]['lucro'].mean())
        ganho_medio_adimplente = df_train_data[df_train_data['lucro'] > 0]['lucro'].mean()
        
        # Lidar com caso de ganho/custo ser 0 ou NaN
        if pd.isna(custo_medio_inadimplente) or custo_medio_inadimplente == 0:
            custo_medio_inadimplente = 1.0
        if pd.isna(ganho_medio_adimplente) or ganho_medio_adimplente == 0:
            ganho_medio_adimplente = 1.0

        peso_linear = custo_medio_inadimplente / ganho_medio_adimplente
        scale_pos_weight = peso_linear * (1 + np.log1p(peso_linear))
        
        print(f"\n Pesos baseados em impacto financeiro:")
        print(f"   Custo médio por inadimplente:     R$ {custo_medio_inadimplente:,.2f}")
        print(f"   Ganho médio por adimplente:       R$ {ganho_medio_adimplente:,.2f}")
        print(f"   Peso Linear:                      {peso_linear:.2f}")
        print(f"   Scale Pos Weight (Log ajustado):  {scale_pos_weight:.2f}")
    else:
        print("\nAviso: Coluna 'lucro' não encontrada. Usando peso de classe padrão (1.0).")
        scale_pos_weight = 1.0

    # --- 5. Criar e Treinar o Modelo ---
    pd_model_nn = criar_modelo_pd_nn(n_features)
    pd_model_nn.summary()

    # O Keras usa um dicionário para 'class_weight'
    class_weights_dict = {
        0: 0.9,  # Peso normal para classe 0 (Adimplente)
        1: scale_pos_weight # Peso calculado para classe 1 (Inadimplente)
    }
    print(f"\nUsando pesos no Keras: {class_weights_dict}")

    early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True)

    print("\nTreinando o modelo de Probabilidade de Default (Rede Neural)...")
    history = pd_model_nn.fit(
        X_train_nn,
        y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2, # Usar 20% dos dados de treino para validação
        callbacks=[early_stop],
        class_weight=class_weights_dict,
        verbose=1
    )
    print("Treinamento concluído.")

    # --- 7. Análise Financeira dos Dados de TREINO (Sua lógica) ---
    train_indices = X_train.index
    df_train_results = dataset_maduro.loc[train_indices].copy()
    
    if 'lucro' in df_train_results.columns:
        arrecadacoes_treino = df_train_results[df_train_results['lucro'] > 0]['lucro'].sum()
        perdas_treino = df_train_results[df_train_results['lucro'] < 0]['lucro'].sum()
        lucro_treino = arrecadacoes_treino + perdas_treino
        percentual_treino = (arrecadacoes_treino / abs(perdas_treino)) * 100 if perdas_treino != 0 else 0
        
        print("\n" + "="*70)
        print("💰 ANÁLISE FINANCEIRA - DADOS DE TREINO")
        print("="*70)
        print(f"📊 Total de contratos no treino: {len(df_train_results):,}")
        print(f"💚 Arrecadações Totais (lucros positivos):  R$ {arrecadacoes_treino:,.2f}")
        print(f"📉 Perdas Totais (lucros negativos):        R$ {perdas_treino:,.2f}")
        print(f"📈 Arrecadação / Perda:                      {percentual_treino:.2f}%")
        print(f"{'─'*70}")
        print(f"💵 Lucro Líquido Total:                      R$ {lucro_treino:,.2f}")
        print("="*70)

    # --- 8. Avaliar o Modelo ---
    print("\n--- Avaliação do Modelo (Dados de Teste) ---")
    y_pred_proba = pd_model_nn.predict(X_test_nn).ravel() # .ravel() para achatar
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"**ROC AUC Score (Rede Neural): {auc_score:.4f}**")
    
        # --- 9. Análise Financeira dos Dados de TESTE (Sua lógica) ---
    test_indices = X_test.index
    df_test_results = dataset_maduro.loc[test_indices].copy()
    
    df_test_results['pred_inadimplente'] = (y_pred_proba >= 0.5).astype(int)
    df_test_results['real_inadimplente'] = y_test.values
    
    if 'lucro' in df_test_results.columns:
        arrecadacoes_teste = df_test_results[df_test_results['lucro'] > 0]['lucro'].sum()
        perdas_teste = df_test_results[df_test_results['lucro'] < 0]['lucro'].sum()
        lucro_teste = arrecadacoes_teste + perdas_teste
        percentual_teste = (arrecadacoes_teste / abs(perdas_teste)) * 100 if perdas_teste != 0 else 0
        
        print("\n" + "="*70)
        print("💰 ANÁLISE FINANCEIRA - DADOS DE TESTE (VALIDAÇÃO)")
        print("="*70)
        print(f"📊 Total de contratos no teste: {len(df_test_results):,}")
        print(f"💚 Arrecadações Totais (lucros positivos):  R$ {arrecadacoes_teste:,.2f}")
        print(f"📉 Perdas Totais (lucros negativos):        R$ {perdas_teste:,.2f}")
        print(f"📈 Arrecadação / Perda:                      {percentual_teste:.2f}%")
        print(f"{'─'*70}")
        print(f"💵 Lucro Líquido Total:                      R$ {lucro_teste:,.2f}")
        print("="*70)
    
    # --- 10. Gerar Gráficos ---
    print("\n--- Gerando Gráficos ---")
    
    # 1. ROC Curve - Cost-Insensitive (modelo sem peso)
    print("Gerando ROC Curve (Cost-Insensitive)...")
    pd_model_nn_no_weight = criar_modelo_pd_nn(n_features)
    
    history_no_weight = pd_model_nn_no_weight.fit(
        X_train_nn,
        y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True)],
        class_weight={0: 1.0, 1: 1.0},  # SEM peso
        verbose=0
    )
    
    y_pred_proba_no_weight = pd_model_nn_no_weight.predict(X_test_nn, verbose=0).ravel()
    
    # Plot ROC Curves
    fpr_sensitive, tpr_sensitive, _ = roc_curve(y_test, y_pred_proba)
    fpr_insensitive, tpr_insensitive, _ = roc_curve(y_test, y_pred_proba_no_weight)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_insensitive, tpr_insensitive, label=f'Cost-Insensitive (AUC = {roc_auc_score(y_test, y_pred_proba_no_weight):.4f})', linewidth=2)
    plt.plot(fpr_sensitive, tpr_sensitive, label=f'Cost-Sensitive (AUC = {auc_score:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Neural Network', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../../graficos/analise_modelos/NN/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ ROC Curves salvas em graficos/NN/roc_curves.png")
    
    # 2. Confusion Matrix
    print("Gerando Confusion Matrix...")
    y_pred = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Adimplente', 'Inadimplente'],
                yticklabels=['Adimplente', 'Inadimplente'])
    plt.xlabel('Predito', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    plt.title('Confusion Matrix - Neural Network (Cost-Sensitive)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../../graficos/analise_modelos/NN/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Confusion Matrix salva em graficos/NN/confusion_matrix.png")
    
    # 3. SHAP Values para Neural Network
    print("Calculando SHAP values (pode demorar bastante para NNs)...")
    print("⚠️ SHAP para redes neurais é muito lento. Pulando...")
    print("   Gerando gráfico de treinamento alternativo...")
    
    try:
        # Plot Training History
        plt.figure(figsize=(12, 5))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Model Loss', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # AUC
        plt.subplot(1, 2, 2)
        plt.plot(history.history['auc'], label='Train AUC')
        plt.plot(history.history['val_auc'], label='Validation AUC')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('AUC', fontsize=12)
        plt.title('Model AUC', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../../graficos/analise_modelos/NN/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Training History Plot salvo em graficos/NN/training_history.png")
        
    except Exception as e:
        print(f"⚠️ Erro ao gerar gráfico de treinamento: {e}")
    
    print("\n✅ Todos os gráficos foram salvos em graficos/NN/")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    run_pd_model()