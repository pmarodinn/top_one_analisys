import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import re

def load_and_preprocess_v3(filepath):
    try:
        df = pd.read_csv(filepath, sep=';', decimal=',') 
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

    # --- 2. Definição da Variável Alvo (Lógica de Contrato Maduro) ---
    
    if 'pago_perc' not in df.columns:
        print("Erro: A coluna 'pago_perc' é necessária para definir o alvo.")
        return None

    # Lógica simplificada:
    # 1.0 (ou 100) = Adimplente (0)
    # < 1.0 (ou < 100) = Inadimplente (1)
    
    # Primeiro, vamos garantir que pago_perc é numérico
    df['pago_perc'] = pd.to_numeric(df['pago_perc'], errors='coerce')

    # Remover linhas onde não sabemos o resultado (pago_perc é Nulo)
    df_modelagem = df.dropna(subset=['pago_perc']).copy()

    # Assumindo que 1 significa 100% pago. 
    # Se 100 for 100%, mude para: df_modelagem['pago_perc'] < 100
    df_modelagem['default'] = (df_modelagem['pago_perc'] < 1).astype(int)
    
    print(f"Dados filtrados (pago_perc preenchido): {df_modelagem.shape[0]} linhas")
    print("Distribuição da variável alvo 'default' (1 = Inadimplente):")
    print(df_modelagem['default'].value_counts(normalize=True))

    # --- 3. Engenharia de Features (Originação) ---
    
    if 'Data_Lancamento' in df_modelagem.columns:
        df_modelagem['data_mes'] = df_modelagem['Data_Lancamento'].dt.month
        df_modelagem['data_ano'] = df_modelagem['Data_Lancamento'].dt.year
        df_modelagem['data_dia_semana'] = df_modelagem['Data_Lancamento'].dt.dayofweek

    if 'categorias' in df_modelagem.columns:
        df_modelagem['categoria_limpa'] = df_modelagem['categorias'].astype(str).str.strip("[]'").str.split(',').str[0].str.strip("'")
    else:
        df_modelagem['categoria_limpa'] = 'desconhecido'
        
    # Calcular divida_sobre_renda com proteção contra divisão por zero
    df_modelagem['divida_sobre_renda'] = df_modelagem['valor_inicial_da_prestacao'] / (df_modelagem['renda_cliente'] + 1e-6)
    
    # Substituir infinitos e valores muito grandes por NaN (serão tratados pelo pipeline)
    df_modelagem = df_modelagem.replace([np.inf, -np.inf], np.nan)
    
    # Limitar valores muito altos na divida_sobre_renda (cap em 100)
    if 'divida_sobre_renda' in df_modelagem.columns:
        df_modelagem['divida_sobre_renda'] = df_modelagem['divida_sobre_renda'].clip(upper=100)

    return df_modelagem

# Bloco de execução principal
if __name__ == "__main__":
    filepath = '../../data/dataset_interno_top_one_atualizado.csv'
    df_processado = load_and_preprocess_v3(filepath)
    
    if df_processado is not None:
        print(f"\n✓ Dados processados com sucesso!")
        print(f"Shape final: {df_processado.shape}")
        
        # Análise Financeira
        if 'lucro' in df_processado.columns:
            perdas_totais = df_processado[df_processado['lucro'] < 0]['lucro'].sum()
            arrecadacoes_totais = df_processado[df_processado['lucro'] > 0]['lucro'].sum()
            lucro_liquido = arrecadacoes_totais + perdas_totais  # perdas já são negativas
            
            print("\n" + "="*60)
            print(" ANÁLISE FINANCEIRA")
            print("="*60)
            print(f" Arrecadações Totais (valores positivos): R$ {arrecadacoes_totais:,.2f}")
            print(f" Perdas Totais (valores negativos):       R$ {perdas_totais:,.2f}")
            print(f"{'─'*60}")
            print(f" Lucro Líquido:                            R$ {lucro_liquido:,.2f}")
            print("="*60)
        
        print(f"\nPrimeiras linhas:")
        print(df_processado.head())
        print(f"\nColunas disponíveis:")
        print(df_processado.columns.tolist())
