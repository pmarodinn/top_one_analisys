# 🚀 Scripts de Produção - Profit-Loss Hurdle Model

Esta pasta contém os scripts para **usar o modelo treinado em produção**.

## 📁 Arquivos

### `predict_new_client.py`
**Script principal de predição**

Carrega os modelos salvos e faz predições para novos clientes.

**Uso via linha de comando:**
```bash
# A partir desta pasta (code/producao/)
python predict_new_client.py --input novo_cliente.csv

# Especificar diretório de modelos diferente
python predict_new_client.py --input clientes.csv --model-dir /caminho/custom/
```

**Uso em código Python:**
```python
from predict_new_client import ProfitLossPredictor

# Carregar modelo (fazer UMA vez)
predictor = ProfitLossPredictor(model_dir='../../modelos/profit_loss_hurdle')

# Fazer predição
resultado = predictor.predict(cliente_data)
print(f"Decisão: {resultado['decision']}")
print(f"Valor Esperado: R$ {resultado['expected_value']:.2f}")
```

### `exemplos_uso.py`
**Exemplos práticos de uso**

Contém 4 exemplos completos:
1. Avaliar cliente único
2. Avaliar múltiplos clientes (batch)
3. Integração em sistema real
4. Comparação com regra simples

**Executar:**
```bash
python exemplos_uso.py
```

## 🔧 Configuração

### Pré-requisitos

1. **Treinar o modelo primeiro:**
   ```bash
   cd ../
   python inadimplente_PROFIT_LOSS_HURDLE.py
   ```
   Isso cria os modelos em `../../modelos/profit_loss_hurdle/`

2. **Instalar dependências:**
   ```bash
   pip install pandas numpy lightgbm xgboost catboost joblib
   ```

### Estrutura de Diretórios Esperada

```
top_one_model_v2/
├── code/
│   ├── inadimplente_PROFIT_LOSS_HURDLE.py  # Script de treino
│   ├── modelos/
│   │   └── load_and_preprocess_v3.py       # Preprocessamento
│   └── producao/                            # 📍 VOCÊ ESTÁ AQUI
│       ├── predict_new_client.py
│       ├── exemplos_uso.py
│       └── README.md
├── modelos/
│   └── profit_loss_hurdle/                  # Modelos treinados
│       ├── clf_lgb.txt
│       ├── clf_xgb.json
│       ├── clf_cat.cbm
│       ├── ... (9 modelos)
│       ├── scaler.pkl
│       └── metadata.json
└── data/
    └── dataset_interno_top_one_atualizado.csv
```

## 📊 Formato de Entrada

O CSV ou DataFrame deve conter as mesmas features usadas no treino.

**Exemplo de `novo_cliente.csv`:**
```csv
valor_solicitado,taxa_juros,prazo_meses,renda_mensal,idade,score_credito
5000.0,2.5,12,3500.0,35,650
```

**Nota:** Se faltarem features, o script preenche automaticamente com 0.

## 🎯 Saída da Predição

```python
{
    'prob_profit': 0.75,           # 75% chance de lucro
    'prob_loss': 0.25,             # 25% chance de prejuízo
    'expected_value': 700.00,      # R$ 700 valor esperado
    'upside_potential': 1200.00,   # Lucro potencial se der certo
    'downside_risk': 800.00,       # Prejuízo potencial se der errado
    'decision': 'ACEITAR',         # Recomendação final
    'threshold': 45.67,            # Threshold de decisão
    'confidence_margin': 654.33    # Margem de confiança
}
```

## 🔄 Workflow de Produção

### 1️⃣ Desenvolvimento/Treino (Periódico)
```bash
cd /path/to/top_one_model_v2/code
python inadimplente_PROFIT_LOSS_HURDLE.py
```
- Treina modelos
- Salva em `../modelos/profit_loss_hurdle/`
- Gera relatórios em `../graficos/analise_modelos/PROFIT_LOSS_MODEL/`

### 2️⃣ Produção (Tempo Real)
```bash
cd producao
python predict_new_client.py --input novos_clientes.csv --output resultados.csv
```
- Carrega modelos salvos
- Faz predições
- Salva resultados

### 3️⃣ Monitoramento
- Guardar todas as predições
- Comparar predições vs resultados reais (quando disponível)
- Retreinar periodicamente com novos dados

## 💡 Exemplo Rápido

```python
from predict_new_client import ProfitLossPredictor
import pandas as pd

# 1. Carregar preditor (UMA VEZ no início da aplicação)
predictor = ProfitLossPredictor()

# 2. Novo cliente chega
novo_cliente = pd.DataFrame({
    'valor_solicitado': [10000],
    'renda_mensal': [5000],
    'score_credito': [720],
    # ... outras features
})

# 3. Preprocessar
cliente_prep = predictor.preprocess_client(novo_cliente)

# 4. Prever
resultado = predictor.predict(cliente_prep)

# 5. Decidir
if resultado['decision'] == 'ACEITAR':
    print(f"✅ Aprovar! EV: R$ {resultado['expected_value']:.2f}")
    print(f"   Prob. Lucro: {resultado['prob_profit']:.1%}")
else:
    print(f"❌ Rejeitar! Risco alto.")
    print(f"   Prob. Prejuízo: {resultado['prob_loss']:.1%}")
```

## 🚨 Troubleshooting

### Erro: "Diretório de modelos não encontrado"
- Certifique-se de ter treinado o modelo primeiro
- Verifique o caminho em `model_dir` (padrão: `../../modelos/profit_loss_hurdle`)

### Erro: "Module 'load_and_preprocess_v3' not found"
- O script busca em `../modelos/load_and_preprocess_v3.py`
- Certifique-se que o arquivo existe

### Predições estranhas
- Verifique se os dados estão no mesmo formato do treino
- Retreine o modelo com dados mais recentes

## 📚 Documentação Adicional

- `../../GUIA_USO_PRODUCAO.md` - Guia completo de uso
- `../../ESTRUTURA_MODELOS.md` - Como os modelos funcionam
- `../inadimplente_PROFIT_LOSS_HURDLE.py` - Script de treino (com comentários)

## ⚡ Performance

- **Carregamento inicial:** ~1-2 segundos (9 modelos)
- **Predição individual:** ~50-100ms
- **Batch (100 clientes):** ~2-5 segundos
- **Memória:** ~200-300 MB

## 🔐 Boas Práticas

✅ **Fazer:**
- Carregar o preditor UMA vez no início da aplicação
- Usar batch processing quando possível (mais eficiente)
- Versionar os modelos (ex: v1.0, v2.0)
- Monitorar performance e retreinar periodicamente

❌ **Evitar:**
- Recarregar modelos a cada predição (muito lento!)
- Usar modelos muito antigos sem retreino
- Ignorar o drift de dados
