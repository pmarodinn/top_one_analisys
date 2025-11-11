# --- Carregar os dados (Versão 3 - Simplificada) ---
# Substitua 'seu_dataset.csv' pelo caminho real do seu arquivo

from load_and_preprocess_v3 import load_and_preprocess_v3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

# Criar diretório para gráficos
os.makedirs('../graficos/XGBOOST', exist_ok=True)

dataset_maduro = load_and_preprocess_v3('../data/dataset_interno_top_one.csv')

if dataset_maduro is not None:
    
    # --- 1. Definir Features (X) e Alvo (y) ---
    y = dataset_maduro['default']
    
    # Colunas a serem EXCLUÍDAS
    colunas_excluir = [
        'default', 'pago_perc', 'lucro', 'aceitar', # Vazamento de informação do futuro
        'contrato_id', 'proposta_id', 'Unnamed: 0', # IDs
        'Data_Lancamento', 'Data_inicio', #Datas
    ]
    
    X = dataset_maduro.drop(columns=colunas_excluir, errors='ignore')
    
    print("\n--- Iniciando Fase 1: Modelo Preditivo (XGBoost) ---")
    print(f"Features utilizadas ({X.shape[1]}): {X.columns.tolist()}")

    # --- 2. Dividir em Treino e Teste ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.5, 
        random_state=42, 
        stratify=y
    )

    # --- 3. Criar Pipeline de Pré-processamento ---
    
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    
    print(f"\nDetectadas {len(numeric_features)} features numéricas.")
    print(f"Detectadas {len(categorical_features)} features categóricas.")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Preenche NaNs (ex: Score_MC)
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # Preenche NaNs (ex: Grupo_MC)
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # --- 4. Criar o Modelo XGBoost ---
    # Calcular peso baseado no impacto financeiro real (não na quantidade de exemplos)
    # Pegar índices de treino para calcular o custo/ganho médio
    train_indices = X_train.index
    df_train_data = dataset_maduro.loc[train_indices].copy()
    
    if 'lucro' in df_train_data.columns:
        custo_medio_inadimplente = abs(df_train_data[df_train_data['lucro'] < 0]['lucro'].mean())
        ganho_medio_adimplente = df_train_data[df_train_data['lucro'] > 0]['lucro'].mean()
        
        # Peso base (linear)
        peso_linear = custo_medio_inadimplente / ganho_medio_adimplente
        
        # Aplicar escala logarítmica mais agressiva para penalizar inadimplentes
        # Inadimplente = perda de arrecadação + calote (duplo prejuízo)
        # Fórmula: peso_linear * (1 + log(1 + peso_linear))
        scale_pos_weight = peso_linear * (1 + np.log1p(peso_linear))
        
        print(f"\n Pesos baseados em impacto financeiro:")
        print(f"   Custo médio por inadimplente:     R$ {custo_medio_inadimplente:,.2f}")
        print(f"   Ganho médio por adimplente:       R$ {ganho_medio_adimplente:,.2f}")
        print(f"   Peso Linear:                      {peso_linear:.2f}")
        print(f"   Scale Pos Weight (Log ajustado):  {scale_pos_weight:.2f}")
        print(f"   Fator de amplificação:            {scale_pos_weight/peso_linear:.2f}x")

    model_xgb = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    pd_model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model_xgb)
    ])

    print("\nTreinando o modelo de Probabilidade de Default (PD)...")
    pd_model_pipeline.fit(X_train, y_train)
    print("Treinamento concluído.")

    # --- 7. Análise Financeira dos Dados de TREINO ---
    train_indices = X_train.index
    df_train_results = dataset_maduro.loc[train_indices].copy()
    
    if 'lucro' in df_train_results.columns:
        arrecadacoes_treino = df_train_results[df_train_results['lucro'] > 0]['lucro'].sum()
        perdas_treino = df_train_results[df_train_results['lucro'] < 0]['lucro'].sum()
        lucro_treino = arrecadacoes_treino + perdas_treino
        percentual_treino = (arrecadacoes_treino / abs(perdas_treino)) * 100 if perdas_treino != 0 else 0
        
        print("\n" + "="*70)
        print(" ANÁLISE FINANCEIRA - DADOS DE TREINO")
        print("="*70)
        print(f" Total de contratos no treino: {len(df_train_results):,}")
        print(f" Arrecadações Totais (lucros positivos):  R$ {arrecadacoes_treino:,.2f}")
        print(f" Perdas Totais (lucros negativos):        R$ {perdas_treino:,.2f}")
        print(f" Arrecadação / Perda:                      {percentual_treino:.2f}%")
        print(f"{'─'*70}")
        print(f" Lucro Líquido Total:                      R$ {lucro_treino:,.2f}")
        print("="*70)

    # --- 8. Avaliar o Modelo ---
    print("\n--- Avaliação do Modelo (Dados de Teste) ---")
    y_pred_proba = pd_model_pipeline.predict_proba(X_test)[:, 1]
    
    # ROC AUC: A principal métrica
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"**ROC AUC Score (Poder de Separação): {auc_score:.4f}**")
    
    # --- 9. Análise Financeira dos Dados de TESTE ---
    test_indices = X_test.index
    df_test_results = dataset_maduro.loc[test_indices].copy()
    
    # Adicionar as previsões
    df_test_results['pred_inadimplente'] = (y_pred_proba >= 0.5).astype(int)
    df_test_results['real_inadimplente'] = y_test.values
    
    # Calcular arrecadação, perda e lucro com base nos dados REAIS (não nas previsões)
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
    
    try:
        cat_features_out = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
        all_feature_names = numeric_features + list(cat_features_out)
        importances = pd_model_pipeline.named_steps['classifier'].feature_importances_
        feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
        print("\n--- Top 10 Features Mais Importantes ---")
        print(feature_importance_df.head(10))
    except Exception as e:
        print(f"\nNão foi possível extrair feature importance: {e}")
    
    # --- 10. Gerar Gráficos ---
    print("\n--- Gerando Gráficos ---")
    
    # 1. ROC Curve - Cost-Insensitive (modelo sem peso)
    print("Gerando ROC Curve (Cost-Insensitive)...")
    model_xgb_no_weight = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        scale_pos_weight=1.0,  # SEM peso
        random_state=42
    )
    
    pd_model_pipeline_no_weight = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model_xgb_no_weight)
    ])
    
    pd_model_pipeline_no_weight.fit(X_train, y_train)
    y_pred_proba_no_weight = pd_model_pipeline_no_weight.predict_proba(X_test)[:, 1]
    
    # Plot ROC Curves
    fpr_sensitive, tpr_sensitive, _ = roc_curve(y_test, y_pred_proba)
    fpr_insensitive, tpr_insensitive, _ = roc_curve(y_test, y_pred_proba_no_weight)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_insensitive, tpr_insensitive, label=f'Cost-Insensitive (AUC = {roc_auc_score(y_test, y_pred_proba_no_weight):.4f})', linewidth=2)
    plt.plot(fpr_sensitive, tpr_sensitive, label=f'Cost-Sensitive (AUC = {auc_score:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - XGBoost', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../graficos/XGBOOST/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ ROC Curves salvas em graficos/XGBOOST/roc_curves.png")
    
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
    plt.title('Confusion Matrix - XGBoost (Cost-Sensitive)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../graficos/XGBOOST/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Confusion Matrix salva em graficos/XGBOOST/confusion_matrix.png")
    
    # 3. SHAP Values
    print("Calculando SHAP values (pode demorar)...")
    try:
        # Usar apenas uma amostra para SHAP (mais rápido)
        sample_size = min(500, len(X_test))
        X_test_sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_test_sample = X_test.iloc[X_test_sample_indices]
        X_test_transformed = preprocessor.transform(X_test_sample)
        
        explainer = shap.TreeExplainer(pd_model_pipeline.named_steps['classifier'])
        shap_values = explainer.shap_values(X_test_transformed)
        
        # SHAP Summary Plot com nomes simplificados
        feature_names_short = [f[:30] for f in all_feature_names]  # Limitar tamanho dos nomes
        
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_test_transformed, 
                         feature_names=feature_names_short,
                         show=False, max_display=15)
        plt.tight_layout()
        plt.savefig('../graficos/XGBOOST/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ SHAP Summary Plot salvo em graficos/XGBOOST/shap_summary.png")
        
    except Exception as e:
        print(f"⚠️ Erro ao gerar SHAP values: {e}")
        print("   Gerando gráfico de importância de features alternativo...")
        
        # Gráfico alternativo: Feature Importance
        try:
            plt.figure(figsize=(10, 8))
            top_n = 20
            top_features = feature_importance_df.head(top_n)
            plt.barh(range(top_n), top_features['importance'].values)
            plt.yticks(range(top_n), [f[:40] for f in top_features['feature'].values])
            plt.xlabel('Importance', fontsize=12)
            plt.title('Top 20 Feature Importances - XGBoost', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('../graficos/XGBOOST/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Feature Importance Plot salvo em graficos/XGBOOST/feature_importance.png")
        except Exception as e2:
            print(f"⚠️ Erro ao gerar gráfico alternativo: {e2}")
    
    print("\n✅ Todos os gráficos foram salvos em graficos/XGBOOST/")