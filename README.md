# Top One Model - Análise de Inadimplência

Modelos de Machine Learning para predição de inadimplência com otimização de lucro financeiro.

## 🎯 Modelo Atual (Última Versão Testada)

### **Neural Network + LightGBM Auditor (Stacking) - v6.2**
**Arquivo**: `code/inadimplente_NN_LUCRO_OTIMIZADO.py`

Arquitetura híbrida em 3 etapas com **aprendizado independente**:

1. **Rede Neural (Filtro)**
   - Arquitetura: 512 → 256 → 128 → 64 → 32 (embeddings) → 1
   - Features: Target Encoding + Dummies + Numéricas (99 features)
   - BatchNormalization + Dropout [0.30, 0.30, 0.20, 0.20]
   - Early Stopping baseado em lucro (patience=15)
   - Extraí embeddings de 32 dimensões

2. **Otimização Bayesiana (Optuna)**
   - 100 trials para otimizar hiperparâmetros do LightGBM
   - Objetivo: Maximizar lucro no conjunto de teste
   - Busca: learning_rate, num_leaves, max_depth, subsample, reg_alpha/lambda

3. **Auditor LightGBM (Especialista)**
   - **Features**: Originais + Embeddings da NN (**sem nn_prob** ✅)
   - LGBM aprende padrões complementares, não copia a NN
   - Otimizado com melhores hiperparâmetros do Optuna
   - Threshold otimizado para maximizar lucro

### � Análise de Interpretabilidade (SHAP)

**Gráficos Gerados** (14 total):
- Summary Plot (dot + bar) - Importância global das features
- Waterfall Plots (TP, FP, FN) - Análise de casos específicos
- Dependence Plots (top 3 features) - Relações não-lineares
- Comparação FP vs TP - Padrões de erro
- Correlação de Embeddings - Top 5 embeddings vs features originais
- t-SNE 2D - Visualização de clusters (lucrativos vs prejuízo)
- Densidade t-SNE - Regiões de risco

**Top 5 Features por SHAP Importance**:
1. `embed_26` (0.085) → Profissão + Mercadoria + Score
2. `embed_6` (0.059) → Profissão + Mercadoria + Categorias
3. `embed_1` (0.027) → Profissão + Mercadoria
4. `embed_29` (0.021) → Profissão + Mercadoria + Estética
5. `embed_15` (0.018) → Localização + IDH + Valor Financiado

### �📊 Resultados Financeiros (Conjunto de Teste)

| Cenário | Lucro | Eficiência | Contratos Aceitos |
|---------|-------|------------|-------------------|
| **1. Aceitar TODOS** | R$ 2.433.890 | 51,53% | 9.241 |
| **2. Máximo Teórico** | R$ 4.723.588 | 100,00% | 7.295 |
| **3. Modelo (NN+LGBM)** | R$ 2.571.854 | **54,45%** | 8.534 |

**Ganho vs Aceitar Todos**: R$ +137.964 (**+5,67%**) 🚀

### 📈 Métricas de Desempenho

- **Recall (Lucrativos)**: 95,00% (6.930/7.295 contratos lucrativos aceitos)
- **False Discovery Rate**: 18,80% (1.604/8.534 contratos aceitos eram prejuízo)
- **Falsos Positivos**: 157 por 1.000 contratos (reduzido!)
- **Falsos Negativos**: 31 por 1.000 contratos
- **Total Rejeitados**: 707 contratos

### 💡 Comparação: Rede Neural vs Auditor LGBM

| Métrica | Rede Neural (Sozinha) | Auditor LGBM (Final) | Melhoria |
|---------|----------------------|---------------------|----------|
| Lucro Otimizado | R$ 2.558.167 | R$ 2.571.854 | +0,53% |
| Ganho vs Aceitar Todos | R$ +124.277 | R$ +137.964 | +11% |
| Erro em Aceitos (FDR) | 18,91% | 18,80% | -0,11 p.p. |

**Ganho do Auditor**: R$ +13.687 (melhoria de 0,53%)

---

## 🚀 Evolução do Modelo

### **v6.2 - Aprendizado Independente (ATUAL)** ✅
**Data**: 13/11/2025  
**Mudança Principal**: Remoção do `nn_prob` das features do LGBM

**Resultados**:
- ✅ **+R$ 102.589** vs versão anterior (com nn_prob)
- ✅ **+1,45 p.p.** no ganho vs aceitar todos (4,22% → 5,67%)
- ✅ **-0,23 p.p.** no FDR (19,03% → 18,80%)
- ✅ Embeddings assumiram protagonismo (embed_26: 0.085 importance)

**Descoberta**: LGBM aprendia padrões **complementares** mais fortes sem copiar a NN.

### **v6.1 - Análise de Interpretabilidade**
**Data**: 13/11/2025  
**Adicionado**:
- 14 gráficos SHAP (Summary, Waterfall, Dependence, Comparação FP vs TP)
- Correlação de Embeddings com features originais
- t-SNE 2D para visualização de clusters
- Relatório de interpretabilidade completo

**Descoberta**: `nn_prob` dominava (0.237 importance), indicando que LGBM "copiava" a NN.

### **v6.0 - NN + LGBM Stacking**
**Data**: 12/11/2025  
**Resultados**: R$ 2.469.265 (+4,22% vs aceitar todos)  
**Arquitetura**: NN → Embeddings → LGBM (com nn_prob)

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

### **1. Gráficos de Treinamento (NN)**
- **Histórico de Loss** (Binary Crossentropy) - Treino vs Validação
- **Evolução da AUC** (Area Under Curve) - Treino vs Validação

### **2. Análise de Performance (LGBM) - 4 subplots**
- Otimização de Threshold (Lucro Total vs Threshold)
- Eficiência vs Threshold
- Distribuição de Probabilidades (Lucrativos vs Prejuízo)
- Scatter: Probabilidade vs Lucro Real

### **3. SHAP Interpretability (14 gráficos)** 🔬
- **Summary Plots**: Dot + Bar (importância global)
- **Waterfall Plots**: True Positive, False Positive, False Negative
- **Dependence Plots**: Top 3 features (embed_26, embed_6, embed_1)
- **Comparação FP vs TP**: Diferenças de importância em erros
- **Correlação de Embeddings**: Top 5 embeddings vs 99 features originais
- **t-SNE 2D**: Visualização de clusters lucrativos vs prejuízo
- **Densidade t-SNE**: Regiões de alto/baixo risco

### **4. Matriz de Confusão**
- Lucro Real vs Decisão do Modelo (Aceitar/Rejeitar)

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
