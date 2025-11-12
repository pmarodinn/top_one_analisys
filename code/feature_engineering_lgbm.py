import pandas as pd
import numpy as np

# Carregar o DataFrame processado da Fase 1
df = pd.read_csv('processed_data_step1.csv', sep=';', decimal=',')

# 1. Tratamento de Valores Ausentes (Imputação)
df['Score_MC'] = df['Score_MC'].fillna(df['Score_MC'].median())
df['Grupo_MC'] = df['Grupo_MC'].fillna('DESCONHECIDO')
df['categorias_da_Profissao'] = df['categorias_da_Profissao'].fillna('DESCONHECIDO')
df['Tipo_Cliente'] = df['Tipo_Cliente'].fillna('DESCONHECIDO')
df['descricao_da_Profissao'] = df['descricao_da_Profissao'].fillna('DESCONHECIDO')

# 2. Engenharia de Features
# Conversão de Datas
df['Data_Lancamento'] = pd.to_datetime(df['Data_Lancamento'], errors='coerce')
df['Data_inicio'] = pd.to_datetime(df['Data_inicio'], errors='coerce')
df['tempo_relacionamento_dias'] = (df['Data_inicio'] - df['Data_Lancamento']).dt.days.fillna(0)

# Features de Renda/Dívida
df['divida_sobre_renda'] = (df['valor_financiado'] / (df['renda_cliente'] + 1e-6)).clip(upper=100)
df['prestacao_sobre_renda'] = (df['valor_inicial_da_prestacao'] / (df['renda_cliente'] + 1e-6)).clip(upper=50)

# Features de Score
df['score_binario_baixo'] = (df['Score_MC'] <= df['Score_MC'].quantile(0.25)).astype(int)

# 3. Definição das Features (X) e Variável Alvo (y)
COLS_TO_DROP = [
    'Unnamed: 0', 'contrato_id', 'proposta_id', 'Data_Lancamento', 'Data_inicio',
    'default', 'lucrativo', 'target_class', 'aceitar', 'categorias', 'uf', 'municipio',
    'Cidade_Loja', 'descricao_da_Profissao', 'Cargo', 'Mercadoria' # Colunas de alta cardinalidade que serão tratadas pelo LGBM
]

X = df.drop(columns=[col for col in COLS_TO_DROP if col in df.columns])
y = df['target_class']
y_lucro_real = df['lucro']

# Identificar features numéricas e categóricas
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Converter colunas categóricas para o tipo 'category' do pandas, que o LightGBM usa
for col in categorical_features:
    X[col] = X[col].astype('category')

# 4. Split de Treino e Teste (Manual, pois train_test_split não está disponível)
# Usaremos uma divisão simples baseada no índice para simular o split
# Isso não é ideal (sem estratificação), mas é um workaround.
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]
y_lucro_test = y_lucro_real.iloc[split_index:]

# 5. Salvar os datasets de treino e teste para a próxima fase
X_train.to_csv('X_train_lgbm.csv', index=False, sep=';', decimal=',')
X_test.to_csv('X_test_lgbm.csv', index=False, sep=';', decimal=',')
y_train.to_csv('y_train_lgbm.csv', index=False, header=True)
y_test.to_csv('y_test_lgbm.csv', index=False, header=True)
y_lucro_test.to_csv('y_lucro_test_lgbm.csv', index=False, header=True)

print("--- Resumo da Engenharia de Features ---")
print(f"Features Numéricas: {len(numeric_features)}")
print(f"Features Categóricas: {len(categorical_features)}")
print(f"Shape de X_train: {X_train.shape}")
print(f"Shape de X_test: {X_test.shape}")
print("\nDistribuição da Variável Alvo (Treino):")
print(y_train.value_counts(normalize=True))
print("\nArquivos de treino/teste salvos com sucesso para LightGBM.")