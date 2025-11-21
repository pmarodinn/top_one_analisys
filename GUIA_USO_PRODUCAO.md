# GUIA DE USO - SISTEMA DE PREDIÇÃO PROFIT-LOSS HURDLE

## 📋 Visão Geral

Após treinar o modelo com `inadimplente_PROFIT_LOSS_HURDLE.py`, você terá:

**Modelos salvos em:** `../modelos/profit_loss_hurdle/`
- 9 modelos de ML (LGBM, XGBoost, CatBoost)
- 1 scaler
- 1 arquivo de metadados

## 🚀 Como Usar em Produção

### 1. Treinar o modelo (uma vez)

```bash
cd code
python inadimplente_PROFIT_LOSS_HURDLE.py
```

Isso vai:
- Treinar todos os modelos
- Salvar em `../modelos/profit_loss_hurdle/`
- Gerar relatórios e gráficos

### 2. Fazer predições para novos clientes

#### Opção A: Um único cliente

```bash
python predict_new_client.py --input novo_cliente.csv
```

Saída detalhada com:
- Valor esperado (EV)
- Probabilidade de lucro
- Potencial upside/downside
- **Decisão: ACEITAR ou REJEITAR**

#### Opção B: Múltiplos clientes (batch)

```bash
python predict_new_client.py --input lista_clientes.csv --output resultados.csv
```

Processa vários clientes de uma vez e salva resultados em CSV.

### 3. Integrar em uma aplicação Python

```python
from predict_new_client import ProfitLossPredictor
import pandas as pd

# Carregar o modelo (fazer uma vez, no início)
predictor = ProfitLossPredictor(model_dir='../modelos/profit_loss_hurdle')

# Dados do novo cliente (pode vir de formulário, API, banco, etc)
novo_cliente = pd.DataFrame({
    'valor_solicitado': [5000.0],
    'taxa_juros': [2.5],
    'prazo_meses': [12],
    'renda_mensal': [3500.0],
    'idade': [35],
    'score_credito': [650],
    # ... todas as outras features que seu modelo usa
})

# Preprocessar
cliente_prep = predictor.preprocess_client(novo_cliente)

# Fazer predição
resultado = predictor.predict(cliente_prep)

# Usar o resultado
if resultado['decision'] == 'ACEITAR':
    print(f"✅ Aprovar! Valor esperado: R$ {resultado['expected_value']:.2f}")
    print(f"   Probabilidade de lucro: {resultado['prob_profit']:.1%}")
else:
    print(f"❌ Rejeitar! Risco muito alto.")
    print(f"   Probabilidade de prejuízo: {resultado['prob_loss']:.1%}")
```

## 📊 Formato dos Dados de Entrada

Seu CSV ou DataFrame precisa ter as mesmas features usadas no treinamento.

**Exemplo de `novo_cliente.csv`:**

```csv
valor_solicitado,taxa_juros,prazo_meses,renda_mensal,idade,score_credito,...
5000.0,2.5,12,3500.0,35,650,...
```

**Importante:** 
- Se faltarem features, o script preenche com 0 automaticamente
- Se tiver features extras, são ignoradas
- A ordem das colunas não importa

## 🔄 Atualizar o Modelo

Quando quiser retreinar com novos dados:

1. Atualize `dataset_interno_top_one_atualizado.csv`
2. Execute novamente: `python inadimplente_PROFIT_LOSS_HURDLE.py`
3. Os modelos antigos em `../modelos/profit_loss_hurdle/` serão sobrescritos
4. Não precisa alterar nada no código de predição!

## 📈 Monitoramento em Produção

Recomendações:
- **Guardar todas as predições** (para análise posterior)
- **Comparar predições vs resultados reais** (quando disponível)
- **Retreinar periodicamente** (ex: mensalmente) com novos dados

## 🎯 Exemplo Completo: API Flask

```python
from flask import Flask, request, jsonify
from predict_new_client import ProfitLossPredictor
import pandas as pd

app = Flask(__name__)

# Carregar modelo na inicialização
predictor = ProfitLossPredictor(model_dir='../modelos/profit_loss_hurdle')

@app.route('/avaliar_cliente', methods=['POST'])
def avaliar_cliente():
    # Receber dados do cliente
    dados = request.json
    
    # Converter para DataFrame
    cliente_df = pd.DataFrame([dados])
    
    # Preprocessar e prever
    cliente_prep = predictor.preprocess_client(cliente_df)
    resultado = predictor.predict(cliente_prep)
    
    # Retornar resposta
    return jsonify({
        'decisao': resultado['decision'],
        'valor_esperado': round(resultado['expected_value'], 2),
        'probabilidade_lucro': round(resultado['prob_profit'] * 100, 1),
        'lucro_potencial': round(resultado['upside_potential'], 2),
        'risco_perda': round(resultado['downside_risk'], 2)
    })

if __name__ == '__main__':
    app.run(debug=False, port=5000)
```

**Uso da API:**

```bash
curl -X POST http://localhost:5000/avaliar_cliente \
  -H "Content-Type: application/json" \
  -d '{
    "valor_solicitado": 5000,
    "taxa_juros": 2.5,
    "prazo_meses": 12,
    "renda_mensal": 3500,
    "idade": 35,
    "score_credito": 650
  }'
```

## 📁 Estrutura de Arquivos

```
top_one_model_v2/
├── code/
│   ├── inadimplente_PROFIT_LOSS_HURDLE.py  # Treino
│   ├── predict_new_client.py               # Predição
│   └── load_and_preprocess_v3.py          # Preprocessamento
├── modelos/
│   └── profit_loss_hurdle/
│       ├── clf_lgb.txt                     # 9 modelos
│       ├── clf_xgb.json
│       ├── clf_cat.cbm
│       ├── reg_upside_lgb.txt
│       ├── reg_upside_xgb.json
│       ├── reg_upside_cat.cbm
│       ├── reg_downside_lgb.txt
│       ├── reg_downside_xgb.json
│       ├── reg_downside_cat.cbm
│       ├── scaler.pkl                      # Scaler
│       └── metadata.json                   # Metadados
└── data/
    └── dataset_interno_top_one_atualizado.csv
```

## ⚠️ Troubleshooting

**Erro: "Diretório de modelos não encontrado"**
- Certifique-se de ter executado o treinamento primeiro
- Verifique o caminho em `--model-dir`

**Erro: "Feature não encontrada"**
- O script preenche automaticamente com 0
- Verifique se os nomes das colunas estão corretos

**Predições estranhas**
- Verifique se os dados estão no mesmo formato do treino
- Retreine o modelo com dados mais recentes
