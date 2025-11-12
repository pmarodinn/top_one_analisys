import pandas as pd
import numpy as np


def softmax(z):
    """Função Softmax para classificação multiclasse."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot_encode(y, num_classes):
    """Codificação One-Hot."""
    # Garante que y é um array de inteiros
    y = y.astype(int)
    return np.eye(num_classes)[y]

def compute_loss(X, Y, W, num_classes, reg_lambda):
    """Calcula a perda de entropia cruzada e a perda de regularização."""
    m = X.shape[0]
    z = X @ W
    probs = softmax(z)
    
    # Perda de Entropia Cruzada
    # Usar np.clip para evitar log(0)
    probs = np.clip(probs, 1e-12, 1.0)
    log_likelihood = -np.log(probs[range(m), np.argmax(Y, axis=1)])
    data_loss = np.sum(log_likelihood) / m
    
    # Perda de Regularização L2
    reg_loss = (reg_lambda / 2) * np.sum(W * W)
    
    return data_loss + reg_loss

def train_softmax_regression(X, y, num_classes, learning_rate=0.01, reg_lambda=0.001, epochs=5000):
    """Treina o modelo de Regressão Logística Softmax."""
    m, n = X.shape
    Y_one_hot = one_hot_encode(y, num_classes)
    
    # Inicialização dos pesos (W)
    np.random.seed(42)
    W = np.random.randn(n, num_classes).astype(np.float64) * 0.01
    
    for epoch in range(epochs):
        # Forward pass
        z = X @ W
        probs = softmax(z)
        
        # Backward pass (Cálculo do gradiente)
        error = probs - Y_one_hot
        dW = (X.T @ error) / m
        
        # Adicionar regularização ao gradiente
        dW += reg_lambda * W
        
        # Atualização dos pesos
        W -= learning_rate * dW
        
        if epoch % 500 == 0:
            loss = compute_loss(X, Y_one_hot, W, num_classes, reg_lambda)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
    return W

def predict_proba(X, W):
    """Calcula as probabilidades de classe."""
    return softmax(X @ W)

def predict(X, W):
    """Calcula a classe prevista."""
    return np.argmax(predict_proba(X, W), axis=1)

def otimizar_threshold_lucro(y_prob, y_true, y_lucro_real, target_class_id):
    """
    Otimiza o threshold de probabilidade para maximizar o lucro para uma classe específica.
    target_class_id: 1 (Inadimplente Lucrativo) ou 2 (Inadimplente Não Lucrativo)
    """
    thresholds = np.arange(0.01, 1.0, 0.01)
    melhor_threshold = 0.5
    melhor_lucro = -np.inf
    
    prob_col = target_class_id
    
    # Para a Classe 1 (Inadimplente Lucrativo): Queremos aceitar
    if target_class_id == 1:
        lucro_max_teorico = y_lucro_real[y_true == 1].sum()
        for threshold in thresholds:
            aceitar = (y_prob[:, prob_col] >= threshold)
            lucro_total = y_lucro_real[aceitar].sum()
            
            if lucro_total > melhor_lucro:
                melhor_lucro = lucro_total
                melhor_threshold = threshold
        
        return melhor_threshold, melhor_lucro, lucro_max_teorico

    # Para a Classe 2 (Inadimplente Não Lucrativo): Queremos rejeitar
    elif target_class_id == 2:
        # Lucro máximo teórico é o lucro total do conjunto de teste (aceitando todos)
        lucro_max_teorico = y_lucro_real.sum()
        
        for threshold in thresholds:
            rejeitar = (y_prob[:, prob_col] >= threshold)
            lucro_obtido = y_lucro_real[~rejeitar].sum()
            
            if lucro_obtido > melhor_lucro:
                melhor_lucro = lucro_obtido
                melhor_threshold = threshold
                
        return melhor_threshold, melhor_lucro, lucro_max_teorico
    
    return 0.5, 0, 0

# --- Carregar Dados ---
X_train = pd.read_csv('X_train_lgbm.csv', sep=';', decimal=',')
X_test = pd.read_csv('X_test_lgbm.csv', sep=';', decimal=',')
# Corrigido: Ignorar o cabeçalho (header=None) e usar skiprows=1
y_train = pd.read_csv('y_train_lgbm.csv', header=None, skiprows=1).iloc[:, 0].values.astype(int)
y_test = pd.read_csv('y_test_lgbm.csv', header=None, skiprows=1).iloc[:, 0].values.astype(int)
y_lucro_test = pd.read_csv('y_lucro_test_lgbm.csv', header=None, skiprows=1).iloc[:, 0].values

# --- Pré-processamento para Softmax Regression ---

# 1. One-Hot Encoding para Categóricas
# Identificar colunas categóricas que ainda são do tipo 'object' (strings)
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
X_train = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

# Alinhar colunas
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
X_test = X_test[X_train.columns]

# 2. Normalização (Manual)
# Agora todas as colunas devem ser numéricas (float ou int)
numeric_features = X_train.columns.tolist()
X_train_num = X_train[numeric_features]
X_test_num = X_test[numeric_features]

mean = X_train_num.mean()
std = X_train_num.std()
std[std == 0] = 1 # Evitar divisão por zero

X_train[numeric_features] = (X_train_num - mean) / std
X_test[numeric_features] = (X_test_num - mean) / std

# 3. Tratamento de NaNs e Conversão para numpy arrays
# Garante que o array final é float64 para a multiplicação de matrizes
X_train_np = X_train.fillna(0).values.astype(np.float64)
X_test_np = X_test.fillna(0).values.astype(np.float64)

# Adicionar o termo de bias (coluna de 1s)
X_train_np = np.hstack([np.ones((X_train_np.shape[0], 1)), X_train_np])
X_test_np = np.hstack([np.ones((X_test_np.shape[0], 1)), X_test_np])

# --- Treinamento do Modelo ---
num_classes = 3
print("Iniciando Treinamento do Modelo Softmax Regression...")
W = train_softmax_regression(X_train_np, y_train, num_classes, epochs=5000, learning_rate=0.01, reg_lambda=0.001)

# Previsões no conjunto de teste
y_pred_proba = predict_proba(X_test_np, W)
y_pred = predict(X_test_np, W)

# --- Avaliação e Otimização de Lucro ---

# 1. Avaliação Geral (Acurácia)
accuracy = np.mean(y_pred == y_test)

print("\n--- Avaliação do Modelo Softmax Regression (Conjunto de Teste) ---")
print(f"Accuracy: {accuracy:.4f}")

# 2. Otimização de Lucro para Inadimplentes Lucrativos (Classe 1)
thresh_lucro, lucro_obtido_c1, lucro_max_c1 = otimizar_threshold_lucro(
    y_pred_proba, y_test, y_lucro_test, target_class_id=1
)

print("\n--- Otimização de Lucro (Classe 1: Inadimplente Lucrativo) ---")
print(f"Lucro Máximo Teórico (C1): R$ {lucro_max_c1:,.2f}")
print(f"Melhor Threshold (C1): {thresh_lucro:.2f}")
print(f"Lucro Obtido (C1): R$ {lucro_obtido_c1:,.2f}")
print(f"Eficiência (C1): {lucro_obtido_c1 / lucro_max_c1 * 100:.2f}%")

# 3. Otimização de Lucro para Inadimplentes Não Lucrativos (Classe 2)
thresh_prejuizo, lucro_obtido_c2, lucro_max_c2 = otimizar_threshold_lucro(
    y_pred_proba, y_test, y_lucro_test, target_class_id=2
)

print("\n--- Otimização de Lucro (Classe 2: Inadimplente Não Lucrativo) ---")
print(f"Lucro Máximo Teórico (C2 - Aceitar Todos): R$ {lucro_max_c2:,.2f}")
print(f"Melhor Threshold (C2 - Rejeição): {thresh_prejuizo:.2f}")
print(f"Lucro Obtido (C2 - Rejeição): R$ {lucro_obtido_c2:,.2f}")
print(f"Eficiência (C2 - Rejeição): {lucro_obtido_c2 / lucro_max_c2 * 100:.2f}%")

# 4. Geração do Output Detalhado (Para a próxima fase)
results_df = X_test.copy()
results_df['y_true'] = y_test
results_df['lucro_real'] = y_lucro_test
results_df['prob_adimplente'] = y_pred_proba[:, 0]
results_df['prob_lucrativo'] = y_pred_proba[:, 1]
results_df['prob_nao_lucrativo'] = y_pred_proba[:, 2]
results_df['y_pred'] = y_pred

# Salvar resultados para a próxima fase de análise detalhada
results_df.to_csv('model_predictions_numpy.csv', index=False, sep=';', decimal=',')
print("\nPrevisões do modelo salvas em 'model_predictions_numpy.csv'")