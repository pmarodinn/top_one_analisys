# Top One Model - Análise de Inadimplência

Modelos de Machine Learning para predição de inadimplência com otimização de custo financeiro.

## Modelos Implementados

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

## Custo

```python
# Cálculo do peso
custo_medio_inadimplente = R$ 1.179,30
ganho_medio_adimplente = R$ 655,71
peso_linear = custo / ganho = 1.80
scale_pos_weight = peso_linear * (1 + log(1 + peso_linear)) = 3.65
```

tentar refletir o **duplo prejuízo** dos inadimplentes:
- Perda da arrecadação
- Calote do valor financiado

## Resultados Financeiros

### Dados de Teste (20%):
- **Arrecadações**: R$ 4.740.098,17
- **Perdas**: R$ -2.338.089,77
- **Lucro Líquido**: R$ 2.402.008,40
- **Proporção**: 203% (Arrecadação/Perda)

## Estrutura do Projeto

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

## Gráficos Gerados

Ambos os modelos geram automaticamente:
- **ROC Curves**: Comparação Cost-Insensitive vs Cost-Sensitive
- **Confusion Matrix**: Matriz de confusão do modelo otimizado
- **Feature Importance** (XGBoost) ou **Training History** (NN)
## 📅 Data

Novembro 2025
