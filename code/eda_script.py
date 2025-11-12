import pandas as pd
import numpy as np

# 1. Carregar o dataset
file_path = '/home/ubuntu/upload/dataset_interno_top_one.csv'
try:
    # Tentativa de carregar com separador ';' e decimal ','
    df = pd.read_csv(file_path, sep=';', decimal=',')
except Exception as e:
    print(f"Erro ao carregar o arquivo: {e}")
    exit()

print("--- Informações Iniciais do Dataset ---")
print(f"Número de linhas: {len(df)}")
print(f"Número de colunas: {len(df.columns)}")
print("\nPrimeiras 5 linhas:")
print(df.head())
print("\nTipos de Dados:")
print(df.info())

# 2. Limpeza e Conversão de Colunas Chave (Baseado na análise do código do usuário)
# As colunas 'lucro' e 'pago_perc' são cruciais e parecem ter sido carregadas corretamente
# devido ao uso de sep=';' e decimal=',' no read_csv.

# 3. Análise da Variável de Inadimplência (default) e Lucro
# Assumindo que inadimplência é pago_perc < 1 (ou seja, não pagou 100%)
df['default'] = (df['pago_perc'] < 1).astype(int)
df['lucrativo'] = (df['lucro'] > 0).astype(int)

print("\n--- Análise da Variável de Inadimplência (default) ---")
print(df['default'].value_counts(normalize=True) * 100)

print("\n--- Análise da Variável Lucrativo ---")
print(df['lucrativo'].value_counts(normalize=True) * 100)

# 4. Criação da Variável Alvo de 3 Classes
# Classe 0: Adimplente (default=0)
# Classe 1: Inadimplente Lucrativo (default=1 e lucrativo=1)
# Classe 2: Inadimplente Não Lucrativo (default=1 e lucrativo=0)
df['target_class'] = np.select(
    [
        df['default'] == 0,
        (df['default'] == 1) & (df['lucrativo'] == 1),
        (df['default'] == 1) & (df['lucrativo'] == 0)
    ],
    [
        0, # Adimplente
        1, # Inadimplente Lucrativo
        2  # Inadimplente Não Lucrativo
    ],
    default=-1 # Erro
)

print("\n--- Distribuição da Variável Alvo de 3 Classes ---")
print(df['target_class'].value_counts())
print("0: Adimplente")
print("1: Inadimplente Lucrativo")
print("2: Inadimplente Não Lucrativo")

# 5. Análise de Lucro por Classe
lucro_por_classe = df.groupby('target_class')['lucro'].agg(['sum', 'mean', 'count'])
print("\n--- Lucro Agregado por Classe ---")
print(lucro_por_classe)

# 6. Identificação de Colunas com Muitos Valores Ausentes (NaN)
print("\n--- Valores Ausentes (Top 10) ---")
missing_values = df.isnull().sum().sort_values(ascending=False)
print(missing_values[missing_values > 0].head(10))

# 7. Salvando o DataFrame processado para a próxima fase
df.to_csv('processed_data_step1.csv', index=False, sep=';', decimal=',')
print("\nDataFrame processado salvo em 'processed_data_step1.csv'")
