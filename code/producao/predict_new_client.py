"""
SISTEMA DE PREDIÇÃO PARA NOVOS CLIENTES - MODELO PROFIT-LOSS HURDLE
====================================================================

Este script permite fazer predições para novos clientes potenciais usando
o modelo Profit-Loss Hurdle treinado.

USO:
    python predict_new_client.py --input novo_cliente.csv
    python predict_new_client.py --interactive  # Modo interativo

SAÍDA:
    - Probabilidade de ser lucrativo
    - Valor esperado (EV)
    - Upside estimado
    - Downside estimado
    - Recomendação de aceitar/rejeitar
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

import lightgbm as lgbm
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

# Adicionar path do projeto
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modelos'))
from load_and_preprocess_v3 import load_and_preprocess_v3

class ProfitLossPredictor:
    """
    Classe para carregar modelos e fazer predições para novos clientes.
    """
    
    def __init__(self, model_dir='../../modelos/profit_loss_hurdle'):
        """
        Inicializa o preditor carregando todos os modelos.
        
        Args:
            model_dir: Diretório onde os modelos foram salvos
        """
        self.model_dir = Path(model_dir)
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Diretório de modelos não encontrado: {model_dir}")
        
        print("="*70)
        print("🚀 CARREGANDO MODELO PROFIT-LOSS HURDLE")
        print("="*70)
        
        # Carregar metadados
        print("\n📋 Carregando metadados...")
        with open(self.model_dir / 'metadata.json', 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"   Modelo: {self.metadata['model_name']}")
        print(f"   Treinado em: {self.metadata['trained_date']}")
        print(f"   Features: {self.metadata['n_features']}")
        print(f"   AUC Ensemble: {self.metadata['metrics']['auc_ensemble']:.4f}")
        
        # Carregar scaler
        print("\n🔧 Carregando scaler...")
        self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
        
        # Carregar classificadores
        print("\n🔮 Carregando classificadores...")
        self.clf_lgb = lgbm.Booster(model_file=str(self.model_dir / 'clf_lgb.txt'))
        self.clf_xgb = xgb.Booster()
        self.clf_xgb.load_model(str(self.model_dir / 'clf_xgb.json'))
        self.clf_cat = CatBoostClassifier()
        self.clf_cat.load_model(str(self.model_dir / 'clf_cat.cbm'))
        
        # Carregar regressores Upside
        print("📈 Carregando regressores Upside...")
        self.reg_up_lgb = lgbm.Booster(model_file=str(self.model_dir / 'reg_upside_lgb.txt'))
        self.reg_up_xgb = xgb.Booster()
        self.reg_up_xgb.load_model(str(self.model_dir / 'reg_upside_xgb.json'))
        self.reg_up_cat = CatBoostRegressor()
        self.reg_up_cat.load_model(str(self.model_dir / 'reg_upside_cat.cbm'))
        
        # Carregar regressores Downside
        print("📉 Carregando regressores Downside...")
        self.reg_down_lgb = lgbm.Booster(model_file=str(self.model_dir / 'reg_downside_lgb.txt'))
        self.reg_down_xgb = xgb.Booster()
        self.reg_down_xgb.load_model(str(self.model_dir / 'reg_downside_xgb.json'))
        self.reg_down_cat = CatBoostRegressor()
        self.reg_down_cat.load_model(str(self.model_dir / 'reg_downside_cat.cbm'))
        
        # Pesos do ensemble
        self.weights_clf = self.metadata['ensemble_weights']['classifier']
        self.weights_up = self.metadata['ensemble_weights']['regressor_upside']
        self.weights_down = self.metadata['ensemble_weights']['regressor_downside']
        
        # Threshold de decisão
        self.threshold_ev = self.metadata['business_metrics']['threshold_ev']
        
        print("\n✅ Modelos carregados com sucesso!")
        print("="*70)
    
    def preprocess_client(self, client_data):
        """
        Preprocessa dados do cliente para formato esperado pelo modelo.
        
        Args:
            client_data: DataFrame com dados do cliente
            
        Returns:
            DataFrame preprocessado
        """
        # Garantir que tem todas as features necessárias
        expected_features = self.metadata['feature_names']
        
        # Adicionar features faltantes com valor 0
        for feat in expected_features:
            if feat not in client_data.columns:
                client_data[feat] = 0
        
        # Ordenar colunas na ordem esperada
        client_data = client_data[expected_features]
        
        # Garantir tipos numéricos
        for col in client_data.columns:
            client_data[col] = pd.to_numeric(client_data[col], errors='coerce')
        
        client_data = client_data.fillna(0)
        
        return client_data
    
    def predict(self, client_data):
        """
        Faz predição completa para um novo cliente.
        
        Args:
            client_data: DataFrame com dados do cliente (já preprocessado)
            
        Returns:
            dict com todas as predições e recomendação
        """
        # Converter para numpy array
        X = client_data.values
        
        # ETAPA 1: Classificador (Probabilidade de Lucro)
        prob_lgb = self.clf_lgb.predict(X)[0]
        prob_xgb = self.clf_xgb.predict(xgb.DMatrix(X))[0]
        prob_cat = self.clf_cat.predict(X, prediction_type='Probability')[:, 1][0]
        
        prob_profit = (
            self.weights_clf['lgb'] * prob_lgb +
            self.weights_clf['xgb'] * prob_xgb +
            self.weights_clf['cat'] * prob_cat
        )
        
        # ETAPA 2: Regressor Upside (Potencial de Lucro)
        pred_up_lgb = self.reg_up_lgb.predict(X)[0]
        pred_up_xgb = self.reg_up_xgb.predict(xgb.DMatrix(X))[0]
        pred_up_cat = self.reg_up_cat.predict(X)[0]
        
        pred_upside_log = (
            self.weights_up['lgb'] * pred_up_lgb +
            self.weights_up['xgb'] * pred_up_xgb +
            self.weights_up['cat'] * pred_up_cat
        )
        pred_upside = np.expm1(pred_upside_log)
        
        # ETAPA 3: Regressor Downside (Risco de Perda)
        pred_down_lgb = self.reg_down_lgb.predict(X)[0]
        pred_down_xgb = self.reg_down_xgb.predict(xgb.DMatrix(X))[0]
        pred_down_cat = self.reg_down_cat.predict(X)[0]
        
        pred_downside_log = (
            self.weights_down['lgb'] * pred_down_lgb +
            self.weights_down['xgb'] * pred_down_xgb +
            self.weights_down['cat'] * pred_down_cat
        )
        pred_downside = np.expm1(pred_downside_log)
        
        # CÁLCULO DO VALOR ESPERADO (EV)
        expected_value = (prob_profit * pred_upside) - ((1 - prob_profit) * pred_downside)
        
        # DECISÃO
        decision = "ACEITAR" if expected_value > self.threshold_ev else "REJEITAR"
        confidence = abs(expected_value - self.threshold_ev)
        
        return {
            'prob_profit': prob_profit,
            'prob_loss': 1 - prob_profit,
            'expected_value': expected_value,
            'upside_potential': pred_upside,
            'downside_risk': pred_downside,
            'decision': decision,
            'threshold': self.threshold_ev,
            'confidence_margin': confidence,
            'components': {
                'prob_lgb': prob_lgb,
                'prob_xgb': prob_xgb,
                'prob_cat': prob_cat
            }
        }
    
    def predict_batch(self, clients_df):
        """
        Faz predições em lote para múltiplos clientes.
        
        Args:
            clients_df: DataFrame com múltiplos clientes
            
        Returns:
            DataFrame com predições
        """
        results = []
        
        for idx in range(len(clients_df)):
            client_row = clients_df.iloc[[idx]]
            pred = self.predict(client_row)
            pred['client_index'] = idx
            results.append(pred)
        
        return pd.DataFrame(results)
    
    def print_prediction(self, prediction):
        """
        Imprime predição de forma amigável.
        """
        print("\n" + "="*70)
        print("📊 RESULTADO DA ANÁLISE")
        print("="*70)
        
        print(f"\n💰 VALOR ESPERADO: R$ {prediction['expected_value']:,.2f}")
        print(f"   (Threshold de decisão: R$ {prediction['threshold']:,.2f})")
        
        print(f"\n📈 PROBABILIDADES:")
        print(f"   • Chance de ser LUCRATIVO: {prediction['prob_profit']:.1%}")
        print(f"   • Chance de dar PREJUÍZO:  {prediction['prob_loss']:.1%}")
        
        print(f"\n📊 ESTIMATIVAS:")
        print(f"   • Lucro potencial (se lucrativo): R$ {prediction['upside_potential']:,.2f}")
        print(f"   • Perda potencial (se prejuízo):  R$ {prediction['downside_risk']:,.2f}")
        
        decision_emoji = "✅" if prediction['decision'] == "ACEITAR" else "❌"
        print(f"\n{decision_emoji} DECISÃO: {prediction['decision']}")
        print(f"   Margem de confiança: R$ {prediction['confidence_margin']:,.2f}")
        
        # Explicação da decisão
        print(f"\n💡 EXPLICAÇÃO:")
        if prediction['decision'] == "ACEITAR":
            print(f"   O valor esperado (R$ {prediction['expected_value']:,.2f}) está")
            print(f"   ACIMA do threshold otimizado (R$ {prediction['threshold']:,.2f}).")
            print(f"   O potencial de lucro ({prediction['prob_profit']:.1%} chance de R$ {prediction['upside_potential']:,.2f})")
            print(f"   supera o risco de perda ({prediction['prob_loss']:.1%} chance de R$ {prediction['downside_risk']:,.2f}).")
        else:
            print(f"   O valor esperado (R$ {prediction['expected_value']:,.2f}) está")
            print(f"   ABAIXO do threshold otimizado (R$ {prediction['threshold']:,.2f}).")
            print(f"   O risco de perda ({prediction['prob_loss']:.1%} chance de R$ {prediction['downside_risk']:,.2f})")
            print(f"   supera o potencial de lucro ({prediction['prob_profit']:.1%} chance de R$ {prediction['upside_potential']:,.2f}).")
        
        print("="*70)


def main():
    """
    Função principal para uso via linha de comando.
    """
    parser = argparse.ArgumentParser(description='Predição para novos clientes')
    parser.add_argument('--input', type=str, help='Arquivo CSV com dados do cliente')
    parser.add_argument('--output', type=str, help='Arquivo de saída (opcional)')
    parser.add_argument('--model-dir', type=str, default='../../modelos/profit_loss_hurdle',
                       help='Diretório dos modelos')
    
    args = parser.parse_args()
    
    # Carregar preditor
    predictor = ProfitLossPredictor(model_dir=args.model_dir)
    
    if args.input:
        # Modo arquivo
        print(f"\n📂 Carregando dados de: {args.input}")
        clients_df = pd.read_csv(args.input)
        
        print(f"   Encontrados {len(clients_df)} cliente(s)")
        
        # Preprocessar
        clients_preprocessed = predictor.preprocess_client(clients_df)
        
        if len(clients_df) == 1:
            # Um único cliente - mostrar detalhes
            prediction = predictor.predict(clients_preprocessed)
            predictor.print_prediction(prediction)
        else:
            # Múltiplos clientes - processar em lote
            print(f"\n⚙️  Processando {len(clients_df)} clientes...")
            results = predictor.predict_batch(clients_preprocessed)
            
            print("\n✅ Predições concluídas!")
            print(f"\nResumo:")
            print(f"   • Aceitar: {(results['decision'] == 'ACEITAR').sum()}")
            print(f"   • Rejeitar: {(results['decision'] == 'REJEITAR').sum()}")
            print(f"   • Valor esperado médio: R$ {results['expected_value'].mean():,.2f}")
            
            if args.output:
                results.to_csv(args.output, index=False)
                print(f"\n💾 Resultados salvos em: {args.output}")
            else:
                print("\n" + str(results[['client_index', 'decision', 'expected_value', 
                                         'prob_profit', 'upside_potential', 'downside_risk']]))
    
    else:
        print("\n⚠️  Nenhum arquivo de entrada fornecido.")
        print("   Use: python predict_new_client.py --input seu_arquivo.csv")
        print("   Ou veja --help para mais opções")


if __name__ == '__main__':
    main()
