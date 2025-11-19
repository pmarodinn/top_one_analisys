Análise dos Embeddings e mapeamento para classes
===============================================

Este documento descreve, em linguagem prática, como o modelo extrai e analisa embeddings (representações latentes) da rede neural, como identificamos as "dimensões" mais importantes e como mapeamos cada dimensão de volta para as features originais para explicar relações com as classes (adimplente / inadimplente / inadimplente lucrativo).

1) Objetivo
-----------
- Entender quais componentes do espaço latente (dimensões do embedding) capturam padrões que discriminam os tipos de clientes.
- Traduzir essas dimensões para features originais para facilitar interpretação negócio.

2) Onde está implementado
-------------------------
Arquivo principal: `code/inadimplente_ENSEMBLE_PARETO_FINAL.py`
- Extração dos embeddings: camada intermediária da rede neural (variável `embedding_extractor`) e arrays `embeddings_train` / `embeddings_test`.
- Função que mapeia dimensões para features: `map_dimensions_to_features(...)`.
- Rotina que calcula as dimensões mais discriminativas por grupo e gera gráficos: seções "ANÁLISE DOS EMBEDDINGS" e "VISUALIZAÇÃO: MAPEAMENTO DIMENSÕES → FEATURES".
- Resumos e arquivos gerados: `graficos/ENSEMBLE_PARETO_FINAL/analise_embeddings.png` e `../graficos/ENSEMBLE_PARETO_FINAL/resumo.txt`.

3) O que são estas "dimensões"?
-------------------------------
- A rede neural inclui uma camada intermediária (denominada "embeddings") com N neurônios. Cada neurônio define uma dimensão do espaço latente.
- Para cada contrato (observação), a rede produz um vetor de dimensão N (o embedding). Esses vetores são utilizados pelo meta-learner e também analisados para interpretação.

4) Passo-a-passo: identificar dimensões importantes
--------------------------------------------------
1. Calcular a média do valor do embedding para cada dimensão e para cada grupo (ex.: adimplentes, inadimplentes totais, inadimplentes lucrativos). Ex.:
   - embeddings_adimplentes = embeddings_test[mask_adimplentes].mean(axis=0)
2. Calcular a média geral (todas as observações) por dimensão: `embeddings_geral`.
3. Medir o desvio absoluto da média do grupo em relação à média geral:
   - desvio = abs(embeddings_grupo - embeddings_geral)
4. Ordenar as dimensões pelo desvio absoluto (maior → mais discriminativa) e selecionar TOP-K (por exemplo, K=5).

Racional: uma dimensão que tem média bastante diferente para um grupo em relação à média geral provavelmente captura um sinal característico desse grupo.

5) Mapear dimensão → features originais (interpretação)
------------------------------------------------------
Para entender o que significa, numericamente, uma dimensão do embedding, fazemos o seguinte:

1. Para cada dimensão d selecionada, pegamos o vetor 1D com o valor dessa dimensão para todas as observações: `embeddings[:, d]`.
2. Calculamos a correlação (Pearson) entre esse vetor e cada feature original (após o mesmo pré-processamento/escalonamento usado em treino). Ex.:
   - corr = np.corrcoef(embeddings[:, d], features[:, i])[0, 1]
3. Ordenamos as features por |corr| e selecionamos as top-N features mais correlacionadas.

Interpretação:
- O sinal da correlação indica direção (positivo → quando a feature aumenta, a dimensão tende a aumentar; negativo → quando a feature aumenta, a dimensão tende a diminuir).
- O valor absoluto indica força do relacionamento.

6) Como usamos isso para explicar classes
-----------------------------------------
- Primeiro identificamos as dimensões que melhor separam um grupo (passo 4).
- Em seguida, para cada dimensão, listamos as features top-correlacionadas (passo 5).
- Assim podemos construir explicações do tipo:
  "A dimensão 12 está alta para adimplentes; essa dimensão se correlaciona positivamente com 'idade', 'saldo', e negativamente com 'número de parcelamentos' — logo, clientes com maior idade e saldo tendem a ter embedding que favorece ser adimplente."

7) Saídas geradas pelo código
----------------------------
- `graficos/ENSEMBLE_PARETO_FINAL/analise_embeddings.png`: 4 subplots
  - Perfil completo dos embeddings (curvas médias por grupo)
  - Barplot Top-5 dimensões discriminativas por grupo
  - Heatmap com valores médios dos embeddings nas dimensões importantes
  - Top-15 dimensões com maior variabilidade

- `graficos/ENSEMBLE_PARETO_FINAL/analise_completa.png`: painéis de desempenho e decisão
- `graficos/ENSEMBLE_PARETO_FINAL/resumo.txt`: arquivo texto contendo os Top-K das dimensões e as features mapeadas para cada dimensão (valor médio, desvio e correlações)

8) Como reproduzir (comandos)
-----------------------------
No diretório `code/` execute:

```bash
python inadimplente_ENSEMBLE_PARETO_FINAL.py
```

Arquivos serão salvos em `graficos/ENSEMBLE_PARETO_FINAL/`.

9) Boas práticas e pontos de atenção
-----------------------------------
- Escalonamento: garanta que ao calcular correlações você passe features escaladas (mesmo StandardScaler do treino). Correlações entre vetores com escalas diferentes podem ser enganadoras.
- Tamanho da amostra: se um grupo (ex.: inadimplentes lucrativos) tiver poucas observações, as médias por dimensão e correlações podem ser instáveis.
- Correlação ≠ causalidade: uma feature correlacionada com uma dimensão indica associação, não prova que a feature "causa" o comportamento.
- Múltiplos testes: quando se testa muitas features, considerar controle de falso-positivos (ex.: p-value, Bonferroni) se for necessário relatório estatístico rigoroso.

10) Próximos passos recomendados
-------------------------------
- Calcular p-values para as correlações e reportar somente as features estatisticamente significantes.
- Usar SHAP no meta-learner (ou em cada base model) para explicações individuais por contrato.
- Treinar um pequeno classificador (ex.: logistic regression) usando apenas as dimensões embeddings como features e inspecionar coeficientes: isso dá um atalho interpretável sobre quais dimensões importam globalmente.
- Clustering dos embeddings para identificar perfis latentes (p.ex., KMeans + análise de features por cluster).

11) Perguntas frequentes
-----------------------
Q: "Uma dimensão do embedding é uma feature original?"
A: Não. É uma combinação não-linear das features que a rede aprendeu. A correlação dimensão→feature dá pistas de que features originais influenciam aquela dimensão.

Q: "Posso usar isso para justificar rejeições aceites?"
A: Sim. A pipeline produz listas de features correlacionadas com as dimensões que levaram à decisão. Para justificativas formais (regulamentação), complemente com SHAP/LIME sobre o meta-learner.

12) Referências rápidas no código
--------------------------------
- Extração de embeddings: `create_nn_regression(...)` e `embedding_extractor` em `inadimplente_ENSEMBLE_PARETO_FINAL.py`.
- Função de correlação: `map_dimensions_to_features(dimensions, embeddings, features, top_n=3)`.
- Plotagem: bloco `VISUALIZAÇÃO: MAPEAMENTO DIMENSÕES → FEATURES` (gera `analise_embeddings.png`).


Se quiser, eu posso:
- Gerar uma versão HTML interativa (Plotly) do mapa dimensão→features.
- Adicionar p-values e filtrar por significância antes de salvar no `resumo.txt`.
- Executar o script aqui e enviar os trechos do `resumo.txt` gerado (se você permitir que eu execute o código no seu workspace).

Quer que eu gere a versão HTML interativa agora, ou prefira que eu rode o script e traga o conteúdo do `resumo.txt`? 