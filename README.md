# Top One Model - Análise de Inadimplência

Modelos de Machine Learning para predição de inadimplência com otimização de custo financeiro.

## 📊 Modelos Implementados

### 1. XGBoost (Cost-Sensitive)
- Modelo baseado em árvores de decisão
- Otimizado com `scale_pos_weight` baseado em impacto financeiro
- ROC AUC: ~78%
- Localização: `code/inadimplente.py`

### 2. Neural Network (Cost-Sensitive)
- Rede Neural Multi-Layer Perceptron (MLP)
- Arquitetura: 64 → 32 → 1 (com Dropout)
- Otimizada com `class_weight` baseado em impacto financeiro
- ROC AUC: ~77%
- Localização: `code/inadimplente_NN.py`

## 🎯 Otimização de Custo

Ambos os modelos utilizam uma abordagem **cost-sensitive** baseada no impacto financeiro real:

```python
# Cálculo do peso
custo_medio_inadimplente = R$ 1.179,30
ganho_medio_adimplente = R$ 655,71
peso_linear = custo / ganho = 1.80
scale_pos_weight = peso_linear * (1 + log(1 + peso_linear)) = 3.65
```

Isto reflete o **duplo prejuízo** dos inadimplentes:
- Perda da arrecadação
- Calote do valor financiado

## 📈 Resultados Financeiros

### Dados de Teste (20%):
- **Arrecadações**: R$ 4.740.098,17
- **Perdas**: R$ -2.338.089,77
- **Lucro Líquido**: R$ 2.402.008,40
- **Proporção**: 203% (Arrecadação/Perda)

## 📁 Estrutura do Projeto

```
top_one_model_v2/
├── code/
│   ├── inadimplente.py          # Modelo XGBoost
│   ├── inadimplente_NN.py       # Modelo Neural Network
│   └── load_and_preprocess_v3.py # Pré-processamento de dados
├── data/
│   └── dataset_interno_top_one.csv (não versionado)
├── graficos/
│   ├── XGBOOST/
│   │   ├── roc_curves.png
│   │   ├── confusion_matrix.png
│   │   └── feature_importance.png
│   └── NN/
│       ├── roc_curves.png
│       ├── confusion_matrix.png
│       └── training_history.png
└── README.md
```

## 🚀 Como Usar

### 1. Instalar Dependências
```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn shap
```

### 2. Preparar Dados
Colocar o arquivo CSV em `data/dataset_interno_top_one.csv`
- Formato: separador `;`, decimal `,`

### 3. Rodar Modelos
```bash
# XGBoost
python code/inadimplente.py

# Neural Network
python code/inadimplente_NN.py
```

## 📊 Gráficos Gerados

Ambos os modelos geram automaticamente:
- **ROC Curves**: Comparação Cost-Insensitive vs Cost-Sensitive
- **Confusion Matrix**: Matriz de confusão do modelo otimizado
- **Feature Importance** (XGBoost) ou **Training History** (NN)

## 🔑 Features Principais

Top 10 features mais importantes (XGBoost):
1. plano_financiamento (9.36%)
2. Score_MC (3.00%)
3. Tipo_Cliente_NORMAL (1.92%)
4. categoria_limpa_estetica (1.68%)
5. estado_civil_Casado (1.63%)

## 📝 Notas

- O modelo utiliza **80/20** (XGBoost) ou **50/50** (NN) de split treino/teste
- Early stopping implementado para prevenir overfitting
- Todas as features categóricas são One-Hot encoded
- Features numéricas são padronizadas (StandardScaler)

## 👥 Autores

Top One Analysis Team

## 📅 Data

Novembro 2025
