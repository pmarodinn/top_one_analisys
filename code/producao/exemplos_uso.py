"""
Script de exemplo para demonstrar como usar o sistema de predição
em uma aplicação real.
"""

from predict_new_client import ProfitLossPredictor
import pandas as pd
import numpy as np

def exemplo_cliente_unico():
    """
    Exemplo 1: Avaliar um único cliente novo
    """
    print("\n" + "="*70)
    print("EXEMPLO 1: AVALIAÇÃO DE CLIENTE ÚNICO")
    print("="*70)
    
    # Carregar preditor
    predictor = ProfitLossPredictor(model_dir='../../modelos/profit_loss_hurdle')
    
    # Simular dados de um novo cliente
    # (Em produção, isso viria de um formulário, banco de dados, API, etc.)
    novo_cliente = pd.DataFrame({
        'valor_solicitado': [8000.0],
        'taxa_juros': [3.5],
        'prazo_meses': [18],
        'renda_mensal': [4500.0],
        'idade': [42],
        'estado_civil': ['casado'],
        'escolaridade': ['superior'],
        'tempo_emprego_meses': [60],
        'tipo_emprego': ['CLT'],
        'score_credito': [720],
        'divida_atual': [5000.0],
        'possui_imovel': [1],
        'possui_veiculo': [1],
        'numero_dependentes': [2],
        # ... adicione todas as outras features conforme necessário
    })
    
    print("\n📝 Dados do cliente:")
    print(novo_cliente.T)
    
    # Preprocessar
    cliente_prep = predictor.preprocess_client(novo_cliente)
    
    # Fazer predição
    resultado = predictor.predict(cliente_prep)
    
    # Exibir resultado
    predictor.print_prediction(resultado)
    
    return resultado


def exemplo_batch_clientes():
    """
    Exemplo 2: Avaliar múltiplos clientes de uma vez (batch)
    """
    print("\n" + "="*70)
    print("EXEMPLO 2: AVALIAÇÃO EM LOTE (BATCH)")
    print("="*70)
    
    # Carregar preditor
    predictor = ProfitLossPredictor(model_dir='../../modelos/profit_loss_hurdle')
    
    # Simular lista de clientes
    clientes = pd.DataFrame({
        'valor_solicitado': [5000, 15000, 3000, 20000, 10000],
        'taxa_juros': [2.5, 4.0, 2.0, 5.0, 3.0],
        'prazo_meses': [12, 24, 6, 36, 18],
        'renda_mensal': [3000, 8000, 2500, 12000, 5000],
        'idade': [28, 45, 35, 52, 38],
        'score_credito': [600, 780, 550, 820, 680],
        # ... outras features
    })
    
    print(f"\n📊 Avaliando {len(clientes)} clientes...")
    
    # Preprocessar
    clientes_prep = predictor.preprocess_client(clientes)
    
    # Fazer predições em lote
    resultados = predictor.predict_batch(clientes_prep)
    
    # Mostrar resumo
    print("\n✅ Resultados:")
    print(resultados[['client_index', 'decision', 'expected_value', 
                      'prob_profit', 'upside_potential', 'downside_risk']])
    
    # Estatísticas
    print(f"\n📈 Estatísticas:")
    print(f"   • Total avaliados: {len(resultados)}")
    print(f"   • Aprovados (ACEITAR): {(resultados['decision'] == 'ACEITAR').sum()}")
    print(f"   • Rejeitados (REJEITAR): {(resultados['decision'] == 'REJEITAR').sum()}")
    print(f"   • Taxa de aprovação: {(resultados['decision'] == 'ACEITAR').mean():.1%}")
    print(f"   • Valor esperado médio: R$ {resultados['expected_value'].mean():,.2f}")
    print(f"   • Prob. lucro média: {resultados['prob_profit'].mean():.1%}")
    
    return resultados


def exemplo_integracao_sistema():
    """
    Exemplo 3: Como integrar em um sistema existente
    """
    print("\n" + "="*70)
    print("EXEMPLO 3: INTEGRAÇÃO EM SISTEMA REAL")
    print("="*70)
    
    # Carregar preditor UMA VEZ (na inicialização da aplicação)
    predictor = ProfitLossPredictor(model_dir='../../modelos/profit_loss_hurdle')
    
    print("\n✅ Preditor carregado e pronto para uso!")
    
    # Função que seria chamada quando um novo cliente chega
    def avaliar_novo_cliente(dados_cliente):
        """
        Função que seria chamada pela sua aplicação/API
        quando um novo cliente solicita crédito.
        """
        # Converter para DataFrame
        cliente_df = pd.DataFrame([dados_cliente])
        
        # Preprocessar
        cliente_prep = predictor.preprocess_client(cliente_df)
        
        # Prever
        resultado = predictor.predict(cliente_prep)
        
        # Retornar decisão e informações relevantes
        return {
            'decisao_sistema': resultado['decision'],
            'valor_esperado': resultado['expected_value'],
            'probabilidade_lucro': resultado['prob_profit'],
            'explicacao': f"Chance de {resultado['prob_profit']:.1%} de lucro de R$ {resultado['upside_potential']:,.2f} vs {resultado['prob_loss']:.1%} de perda de R$ {resultado['downside_risk']:,.2f}"
        }
    
    # Simular chegada de novos clientes
    print("\n🔄 Simulando avaliação de clientes em tempo real...\n")
    
    clientes_exemplo = [
        {'nome': 'João Silva', 'valor_solicitado': 10000, 'renda_mensal': 5000, 'score_credito': 720},
        {'nome': 'Maria Santos', 'valor_solicitado': 3000, 'renda_mensal': 2500, 'score_credito': 580},
        {'nome': 'Pedro Costa', 'valor_solicitado': 50000, 'renda_mensal': 15000, 'score_credito': 850},
    ]
    
    for i, cliente in enumerate(clientes_exemplo, 1):
        print(f"Cliente {i}: {cliente['nome']}")
        print(f"   Solicitação: R$ {cliente['valor_solicitado']:,.2f}")
        
        decisao = avaliar_novo_cliente(cliente)
        
        emoji = "✅" if decisao['decisao_sistema'] == 'ACEITAR' else "❌"
        print(f"   {emoji} Decisão: {decisao['decisao_sistema']}")
        print(f"   💰 Valor Esperado: R$ {decisao['valor_esperado']:,.2f}")
        print(f"   📊 Prob. Lucro: {decisao['probabilidade_lucro']:.1%}")
        print(f"   💡 {decisao['explicacao']}")
        print()


def exemplo_comparacao_com_regra_simples():
    """
    Exemplo 4: Comparar modelo ML com regra de negócio simples
    """
    print("\n" + "="*70)
    print("EXEMPLO 4: COMPARAÇÃO COM REGRA SIMPLES")
    print("="*70)
    
    predictor = ProfitLossPredictor(model_dir='../../modelos/profit_loss_hurdle')
    
    # Cliente para teste
    cliente_teste = pd.DataFrame({
        'valor_solicitado': [12000],
        'renda_mensal': [4000],
        'score_credito': [650],
        'divida_atual': [8000],
        # ... outras features
    })
    
    print("\n📊 Cliente teste:")
    print(f"   Renda: R$ 4.000")
    print(f"   Valor solicitado: R$ 12.000")
    print(f"   Score: 650")
    print(f"   Dívida atual: R$ 8.000")
    
    # REGRA SIMPLES (exemplo de regra tradicional)
    def regra_simples(cliente):
        renda = cliente['renda_mensal'].iloc[0]
        valor = cliente['valor_solicitado'].iloc[0]
        score = cliente['score_credito'].iloc[0]
        divida = cliente['divida_atual'].iloc[0]
        
        # Regra: aprovar se score >= 700 E valor <= 3x renda E divida < 50% renda
        if score >= 700 and valor <= renda * 3 and divida < renda * 0.5:
            return "ACEITAR"
        else:
            return "REJEITAR"
    
    # MODELO ML
    cliente_prep = predictor.preprocess_client(cliente_teste)
    resultado_ml = predictor.predict(cliente_prep)
    
    # Comparar
    print("\n📋 COMPARAÇÃO:")
    print(f"\n1️⃣ REGRA SIMPLES:")
    decisao_simples = regra_simples(cliente_teste)
    print(f"   Decisão: {decisao_simples}")
    print(f"   Justificativa: Baseada em thresholds fixos de score, renda e dívida")
    
    print(f"\n2️⃣ MODELO ML (PROFIT-LOSS HURDLE):")
    print(f"   Decisão: {resultado_ml['decision']}")
    print(f"   Valor Esperado: R$ {resultado_ml['expected_value']:,.2f}")
    print(f"   Prob. Lucro: {resultado_ml['prob_profit']:.1%}")
    print(f"   Justificativa: Baseada em análise de risco/retorno com 10.000+ features")
    
    print("\n💡 VANTAGEM DO MODELO ML:")
    print("   • Considera TODAS as variáveis simultaneamente")
    print("   • Estima lucro/prejuízo esperado (não apenas binário sim/não)")
    print("   • Adapta-se a padrões complexos dos dados")
    print("   • Pode ser atualizado continuamente com novos dados")


if __name__ == '__main__':
    print("\n🚀 EXEMPLOS DE USO DO SISTEMA PROFIT-LOSS HURDLE")
    
    # Executar todos os exemplos
    exemplo_cliente_unico()
    input("\n⏸️  Pressione ENTER para continuar...")
    
    exemplo_batch_clientes()
    input("\n⏸️  Pressione ENTER para continuar...")
    
    exemplo_integracao_sistema()
    input("\n⏸️  Pressione ENTER para continuar...")
    
    exemplo_comparacao_com_regra_simples()
    
    print("\n" + "="*70)
    print("✅ EXEMPLOS CONCLUÍDOS!")
    print("="*70)
    print("\n📚 Consulte GUIA_USO_PRODUCAO.md para mais informações.")
