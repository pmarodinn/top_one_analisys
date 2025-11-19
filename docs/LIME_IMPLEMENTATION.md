# Implementação LIME no Ensemble Model

## Objetivo

Adicionar **interpretabilidade local** ao modelo ensemble, permitindo explicar decisões individuais de aceitação/rejeição de contratos através do LIME (Local Interpretable Model-agnostic Explanations).

## O que é LIME?

LIME é uma técnica que explica predições de modelos black-box (como nosso ensemble) criando um modelo interpretável local ao redor de uma instância específica.

### Como funciona:

1. **Perturbação**: Gera variações da instância original (mudando features)
2. **Predição**: Usa o modelo black-box para prever as variações
3. **Pesos**: Atribui pesos maiores para variações mais próximas da instância original
4. **Modelo Local**: Treina modelo linear simples (interpretável) com os dados ponderados
5. **Explicação**: Os coeficientes do modelo linear mostram a importância de cada feature

## Implementação no Código

### Arquivo: `inadimplente_ENSEMBLE_PARETO_FINAL_LIME.py`

### 1. Wrapper para o Meta-Learner

```python
def meta_learner_predict(X_features):
    """Wrapper para predição do meta-learner"""
    # 1. Predições dos modelos base (NN, LGBM, XGB)
    pred_nn = model_nn.predict(X_features, verbose=0).flatten()
    pred_lgbm = model_lgbm.predict(X_features)
    pred_xgb = model_xgb.predict(xgb.DMatrix(X_features))
    
    # 2. Extrair embeddings da NN
    emb = embedding_extractor.predict(X_features, verbose=0)
    
    # 3. Combinar com top features originais
    meta_feats = np.column_stack([
        pred_nn, pred_lgbm, pred_xgb, emb, X_features[:, top_features_idx]
    ])
    
    # 4. Predição final
    return meta_model.predict(xgb.DMatrix(meta_feats))
```

**Por que este wrapper?**
- LIME espera uma função que recebe features originais e retorna predições
- Nosso ensemble tem múltiplas etapas (base models → embeddings → meta-learner)
- O wrapper encapsula todo o pipeline

### 2. Criação do Explainer

```python
explainer = lime_tabular.LimeTabularExplainer(
    X_train_scaled,              # Dados de treino (para entender distribuições)
    feature_names=feature_names_list,  # Nomes das features originais
    mode='regression',           # Problema de regressão (predizer lucro)
    discretize_continuous=True,  # Discretizar features contínuas em bins
    random_state=42
)
```

### 3. Seleção de Exemplos Representativos

O código seleciona 6 contratos diversos:

1. **Aceito - Adimplente (maior lucro)**: Melhor caso - acertou totalmente
2. **Aceito - Adimplente (lucro médio)**: Caso típico bem-sucedido
3. **Aceito - Inadimplente Lucrativo**: Inadimplência parcial mas ainda lucrativo
4. **Aceito - Inadimplente Não Lucrativo**: Erro tipo I - aceitou quando não devia
5. **Rejeitado - Adimplente (FP)**: Erro tipo II - rejeitou bom cliente
6. **Rejeitado - Inadimplente (TP)**: Acerto - rejeitou mau cliente

```python
# Exemplo: Aceito adimplente com maior lucro
mask_aceitos_adim = aceitar_final & (pago_perc_test == 1.0)
if mask_aceitos_adim.sum() > 0:
    idx = np.where(mask_aceitos_adim)[0][np.argmax(y_test_lucro[mask_aceitos_adim])]
    exemplos.append(('Aceito - Adimplente (maior lucro)', idx))
```

### 4. Geração de Explicações

```python
for label, idx in exemplos:
    exp = explainer.explain_instance(
        X_test_scaled[idx],      # Instância a explicar
        meta_learner_predict,    # Função de predição
        num_features=10,         # Top 10 features mais importantes
        num_samples=5000         # Qtd de perturbações para treinar modelo local
    )
```

**Parâmetros importantes:**
- `num_features=10`: Mostra as 10 features mais influentes
- `num_samples=5000`: Quanto maior, mais precisa a explicação (mas mais lento)

## Outputs Gerados

### 1. Console Output

Para cada exemplo:
```
--- Aceito - Adimplente (maior lucro) (idx=123) ---
   Decisão: ACEITAR
   Lucro Predito: R$ 1,234.56
   Lucro Real: R$ 1,200.00
   Pagamento: 100.0%
   Top 5 Features Influentes:
      📈 idade > 45                                        | peso: +0.2345
      📉 num_parcelas <= 12                                | peso: -0.1234
      📈 score_SPC > 700                                   | peso: +0.1123
      ...
```

### 2. Visualização: `graficos/LIME_EXPLANATIONS/lime_top_examples.png`

Grid 3x2 com 6 subplots mostrando:
- Gráfico de barras horizontais
- Verde = feature aumenta lucro predito
- Vermelho = feature diminui lucro predito
- Comprimento da barra = magnitude do efeito

### 3. Relatório Textual: `graficos/LIME_EXPLANATIONS/lime_report.txt`

Relatório detalhado com:
- Informações do contrato
- Decisão do modelo
- Lucro predito vs real
- Top 10 features mais influentes com pesos

## Interpretação dos Resultados

### Peso Positivo (+)
- Feature contribui para **aumentar** o lucro predito
- Exemplo: `idade > 45` com peso +0.23 → clientes mais velhos aumentam lucro esperado

### Peso Negativo (-)
- Feature contribui para **diminuir** o lucro predito
- Exemplo: `num_parcelas > 24` com peso -0.15 → muitas parcelas diminuem lucro esperado

### Magnitude
- Peso absoluto alto (|peso| > 0.1): Feature muito influente
- Peso absoluto baixo (|peso| < 0.05): Feature pouco influente

## Casos de Uso

### 1. Justificar Rejeições

Para cliente rejeitado:
```
Cliente XYZ foi rejeitado porque:
- Score SPC baixo (<500): -R$ 150 no lucro esperado
- Histórico de inadimplência: -R$ 100
- Renda insuficiente: -R$ 80
→ Lucro esperado: -R$ 50 (abaixo do threshold)
```

### 2. Identificar Padrões

Comparando múltiplas explicações:
- Adimplentes: Score alto, renda estável, poucas parcelas
- Inadimplentes: Score baixo, muitas parcelas, profissão de risco

### 3. Validar Modelo

- Se features importantes fazem sentido de negócio → modelo confiável
- Se features irrelevantes dominam → modelo pode estar overfitting

## Limitações do LIME

1. **Instabilidade**: Executar 2x na mesma instância pode dar pesos ligeiramente diferentes
   - Solução: Usar `random_state` fixo e `num_samples` alto

2. **Aproximação Local**: Explicação válida apenas perto da instância
   - Não assume que mesmas features importam globalmente

3. **Custo Computacional**: 5000 predições por instância
   - Para explicar 100 contratos = 500k predições

4. **Features Correlacionadas**: LIME pode distribuir peso entre features correlacionadas
   - Ex: `idade` e `tempo_emprego` podem dividir importância

## Alternativas e Complementos

### SHAP (SHapley Additive exPlanations)
- Mais robusto matematicamente
- Garante consistência
- Mais lento que LIME

```python
import shap
explainer_shap = shap.KernelExplainer(meta_learner_predict, X_train_scaled[:100])
shap_values = explainer_shap.shap_values(X_test_scaled[idx])
```

### Anchors
- Cria regras simples tipo "IF-THEN"
- Mais interpretável para leigos

### Counterfactuals
- "O que precisaria mudar para decisão reverter?"
- Útil para recomendações ao cliente

## Próximos Passos

1. **Análise de Estabilidade**: Rodar LIME 10x na mesma instância, verificar variação
2. **Comparação LIME vs SHAP**: Validar se explicações convergem
3. **Dashboard Interativo**: Streamlit/Dash para explorar explicações
4. **Explicações em Batch**: Explicar todos os 9,243 contratos (pode levar horas)
5. **Feature Importance Global**: Agregar explicações locais para ranking global

## Comandos para Executar

```bash
# Instalar LIME (se necessário)
pip install lime

# Executar modelo com LIME
cd code/
python inadimplente_ENSEMBLE_PARETO_FINAL_LIME.py

# Outputs em:
# - graficos/LIME_EXPLANATIONS/lime_top_examples.png
# - graficos/LIME_EXPLANATIONS/lime_report.txt
```

## Referências

- [Artigo Original LIME](https://arxiv.org/abs/1602.04938)
- [Documentação lime Python](https://lime-ml.readthedocs.io/)
- [Tutorial LIME](https://github.com/marcotcr/lime/tree/master/doc/notebooks)

## FAQ

**P: Por que não usar apenas feature importance dos modelos base?**
R: Feature importance é global. LIME explica **por que esta instância específica** foi classificada assim.

**P: LIME funciona com qualquer modelo?**
R: Sim, é model-agnostic. Só precisa de uma função predict.

**P: Quanto tempo leva para explicar 1 contrato?**
R: ~1-2 segundos com 5000 samples. Para 10k contratos, pode levar horas.

**P: Posso confiar 100% nas explicações?**
R: Use como guia, não como verdade absoluta. Valide com conhecimento de negócio.
