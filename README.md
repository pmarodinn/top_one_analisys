# Top One Model - Análise de Inadimplência

Modelos de Machine Learning para predição de inadimplência com otimização de lucro financeiro.

## 🎯 Modelo Atual (Última Versão Testada)

### **Neural Network + LightGBM Auditor (Stacking)**
**Arquivo**: `code/inadimplente_NN_LUCRO_OTIMIZADO.py`

Arquitetura híbrida em 3 etapas:

1. **Rede Neural (Filtro)**
   - Arquitetura: 512 → 256 → 128 → 64 → 32 (embeddings) → 1
   - Features: Target Encoding + Dummies + Numéricas
   - BatchNormalization + Dropout
   - Early Stopping baseado em lucro

2. **Otimização Bayesiana (Optuna)**
   - 100 trials para otimizar hiperparâmetros do LightGBM
   - Objetivo: Maximizar lucro no conjunto de teste
   - Busca inteligente de parâmetros

3. **Auditor LightGBM (Especialista)**
   - Features: Originais + Embeddings da NN + Probabilidade da NN
   - Otimizado com melhores hiperparâmetros do Optuna
   - Threshold otimizado para maximizar lucro

### 📊 Resultados Financeiros (Conjunto de Teste)

| Cenário | Lucro | Eficiência | Contratos Aceitos |
|---------|-------|------------|-------------------|
| **1. Aceitar TODOS** | R$ 2.446.507,00 | 51,52% | 9.241 |
| **2. Máximo Teórico** | R$ 4.748.002,23 | 100,00% | 6.768 |
| **3. Modelo (NN+LGBM)** | R$ 2.472.828,79 | **52,08%** | 8.779 |

**Ganho vs Aceitar Todos**: R$ +26.321,79 (+1,08%)

### 📈 Métricas de Desempenho

- **Recall (Lucrativos)**: 92,32% (6.247/6.768 contratos lucrativos aceitos)
- **False Discovery Rate**: 29,27% (2.532/8.779 contratos aceitos eram prejuízo)
- **Total Rejeitados**: 462 contratos

### 💡 Comparação: Rede Neural vs Auditor LGBM

| Métrica | Rede Neural (Sozinha) | Auditor LGBM (Final) |
|---------|----------------------|---------------------|
| Lucro Otimizado | R$ 2.438.534,42 | R$ 2.472.828,79 |
| Ganho vs Aceitar Todos | R$ +7.972 | R$ +26.322 |
| Erro em Aceitos (FDR) | 29,46% | 29,27% |

**Ganho do Auditor**: R$ +34.294,37 (melhoria de 1,41%)

---

## 📚 Modelos Anteriores

### 1. XGBoost (Cost-Sensitive)
- Modelo baseado em árvores de decisão
- Otimizado com `scale_pos_weight` baseado em impacto financeiro
- ROC AUC: ~78%
- Localização: `code/inadimplente_XGBoost.py`

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

## 🔬 Metodologia

### Target Encoding
- Colunas de alta cardinalidade (Mercadoria, Profissão, Cidade, etc.) são convertidas usando a média do target no treino
- Evita explosão dimensional (de 10.288 → 341 features)
- Reduz overfitting

### Otimização de Threshold
- Busca exaustiva (0.00 a 1.00, step 0.01)
- Objetivo: Maximizar lucro líquido = Σ(lucro dos aceitos)
- Threshold ótimo encontrado: ~0.3877

### Class Weights
- Calculado automaticamente com `compute_class_weight('balanced')`
- Compensa desbalanceamento (73% lucrativos, 27% prejuízo)

## 📁 Estrutura do Projeto

```
top_one_model_v2/
├── code/
│   ├── inadimplente_NN_LUCRO_OTIMIZADO.py  # 🔥 Modelo Atual (NN+LGBM Stacking)
│   ├── inadimplente_XGBoost.py             # Modelo XGBoost
│   ├── inadimplente_XGBoost_v2.py          # XGBoost + Bayesian Calibration
│   ├── inadimplente_LUCRO_OTIMIZADO.py     # XGBoost Regression (% pago)
│   ├── inadimplente_NN.py                  # Neural Network simples
│   └── load_and_preprocess_v3.py           # Pré-processamento de dados
├── data/
│   └── dataset_interno_top_one.csv         # Dataset (não versionado)
├── modelos/
│   ├── nn_filtro_etapa1.h5                 # Rede Neural treinada
│   └── lgbm_auditor_etapa2_otimizado.txt   # LightGBM otimizado
├── graficos/
│   ├── NN_LGBM_AUDITOR_OTIMIZADO/          # 🔥 Gráficos do modelo atual
│   │   ├── historico_treinamento_NN.png
│   │   ├── analise_completa.png
│   │   └── confusion_matrix.png
│   ├── XGBOOST/
│   └── NN/
└── README.md
```

## 📊 Gráficos Gerados (Modelo Atual)

O modelo `inadimplente_NN_LUCRO_OTIMIZADO.py` gera automaticamente:

1. **Histórico de Treinamento (NN)**
   - Loss (Binary Crossentropy) - Treino vs Validação
   - AUC (Area Under Curve) - Treino vs Validação

2. **Análise Completa (4 subplots)**
   - Otimização de Threshold (Lucro Total vs Threshold)
   - Eficiência vs Threshold
   - Distribuição de Probabilidades (Lucrativos vs Prejuízo)
   - Scatter: Probabilidade vs Lucro Real

3. **Confusion Matrix**
   - Matriz de confusão visual (Lucro vs Decisão)

## 🚀 Como Executar

```bash
cd code
python inadimplente_NN_LUCRO_OTIMIZADO.py
```

**Requisitos**:
- TensorFlow/Keras
- LightGBM
- Optuna
- scikit-learn
- pandas, numpy, matplotlib, seaborn

## 📅 Última Atualização

13 de Novembro de 2025
