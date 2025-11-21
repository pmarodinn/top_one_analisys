"""
Análise Específica: Rio de Janeiro - Inadimplentes Parciais e Lucro Temporal
==============================================================================
RJ é o estado com MAIS inadimplentes, mas não o que mais dá prejuízo.
Este script analisa a evolução temporal do lucro para entender quais períodos
deram lucro ou prejuízo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

print("="*80)
print("🔍 ANÁLISE ESPECÍFICA: RIO DE JANEIRO - LUCRO TEMPORAL")
print("="*80)

# ========== CARREGAMENTO DOS DADOS ==========
print("\n📂 Carregando dados...")
df = pd.read_csv('/home/usuario/Downloads/Documentos/top_one_model_v2/data/dataset_interno_top_one_atualizado.csv', 
                 sep=';', encoding='utf-8')

print(f"   Total de registros: {len(df):,}")

# ========== CONVERSÃO DE DATAS E COLUNAS ==========
print("\n📅 Convertendo datas e colunas numéricas...")
df['Data_inicio'] = pd.to_datetime(df['Data_inicio'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df['Data_Lancamento'] = pd.to_datetime(df['Data_Lancamento'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df['pago_perc'] = df['pago_perc'].astype(str).str.replace(',', '.').astype(float)
df['plano_financiamento'] = pd.to_numeric(df['plano_financiamento'], errors='coerce')
df['lucro'] = df['lucro'].astype(str).str.replace(',', '.').astype(float)

df = df.dropna(subset=['Data_inicio'])
print(f"   ✅ Conversões concluídas")

# ========== FILTRAR RIO DE JANEIRO ==========
print("\n🎯 Filtrando dados do Rio de Janeiro...")
df_rj = df[df['uf'] == 'RJ'].copy()

print(f"   Total RJ: {len(df_rj):,} contratos")
print(f"   Lucro total RJ: R$ {df_rj['lucro'].sum():,.2f}")

# Mostrar range de datas
data_min = df_rj['Data_inicio'].min()
data_max = df_rj['Data_inicio'].max()
meses_unicos = df_rj['Data_inicio'].dt.to_period('M').nunique()
print(f"\n   📅 Período dos dados:")
print(f"      • Data início: {data_min.strftime('%Y-%m-%d')}")
print(f"      • Data fim: {data_max.strftime('%Y-%m-%d')}")
print(f"      • Total de meses: {meses_unicos} meses")

# Separar por tipo de cliente
df_rj_adim = df_rj[df_rj['pago_perc'] == 1.0]
df_rj_inadim_total = df_rj[df_rj['pago_perc'] == 0.0]
df_rj_inadim_parcial = df_rj[(df_rj['pago_perc'] > 0) & (df_rj['pago_perc'] < 1)]

print(f"\n   📊 Distribuição:")
print(f"      • Adimplentes (100%): {len(df_rj_adim):,} ({len(df_rj_adim)/len(df_rj)*100:.1f}%)")
print(f"      • Inadimplentes Totais (0%): {len(df_rj_inadim_total):,} ({len(df_rj_inadim_total)/len(df_rj)*100:.1f}%)")
print(f"      • Inadimplentes Parciais (0-100%): {len(df_rj_inadim_parcial):,} ({len(df_rj_inadim_parcial)/len(df_rj)*100:.1f}%)")

print(f"\n   💰 Lucro por tipo:")
print(f"      • Adimplentes: R$ {df_rj_adim['lucro'].sum():,.2f}")
print(f"      • Inadim. Totais: R$ {df_rj_inadim_total['lucro'].sum():,.2f}")
print(f"      • Inadim. Parciais: R$ {df_rj_inadim_parcial['lucro'].sum():,.2f}")

# ========== ANÁLISE TEMPORAL DO LUCRO ==========
print("\n" + "="*80)
print("📈 ANÁLISE TEMPORAL DO LUCRO - RIO DE JANEIRO")
print("="*80)

# Criar coluna de mês/ano
df_rj['ano_mes'] = df_rj['Data_inicio'].dt.to_period('M').astype(str)

# Agregar lucro por mês
lucro_mensal = df_rj.groupby('ano_mes').agg({
    'lucro': ['sum', 'mean', 'count'],
    'pago_perc': 'mean'
}).reset_index()

lucro_mensal.columns = ['ano_mes', 'lucro_total', 'lucro_medio', 'num_contratos', 'pago_perc_medio']

# Separar por tipo de cliente
lucro_adim = df_rj_adim.groupby(df_rj_adim['Data_inicio'].dt.to_period('M').astype(str))['lucro'].sum()
lucro_inadim_total = df_rj_inadim_total.groupby(df_rj_inadim_total['Data_inicio'].dt.to_period('M').astype(str))['lucro'].sum()
lucro_inadim_parcial = df_rj_inadim_parcial.groupby(df_rj_inadim_parcial['Data_inicio'].dt.to_period('M').astype(str))['lucro'].sum()

# Alinhar índices
todos_meses = sorted(df_rj['ano_mes'].unique())
lucro_adim = lucro_adim.reindex(todos_meses, fill_value=0)
lucro_inadim_total = lucro_inadim_total.reindex(todos_meses, fill_value=0)
lucro_inadim_parcial = lucro_inadim_parcial.reindex(todos_meses, fill_value=0)

print("\n🔥 Meses com MAIOR lucro:")
top_lucro = lucro_mensal.nlargest(5, 'lucro_total')
for _, row in top_lucro.iterrows():
    print(f"   {row['ano_mes']}: R$ {row['lucro_total']:,.2f} ({row['num_contratos']:.0f} contratos)")

print("\n❌ Meses com MAIOR prejuízo:")
top_prejuizo = lucro_mensal.nsmallest(5, 'lucro_total')
for _, row in top_prejuizo.iterrows():
    print(f"   {row['ano_mes']}: R$ {row['lucro_total']:,.2f} ({row['num_contratos']:.0f} contratos)")

# Calcular acumulado
lucro_mensal['lucro_acumulado'] = lucro_mensal['lucro_total'].cumsum()

print(f"\n💰 Lucro acumulado até hoje: R$ {lucro_mensal['lucro_acumulado'].iloc[-1]:,.2f}")

# ========== ANÁLISE POR CIDADE ==========
print("\n" + "="*80)
print("🏙️  ANÁLISE POR CIDADE - RIO DE JANEIRO")
print("="*80)

lucro_por_cidade = df_rj.groupby('municipio').agg({
    'lucro': ['sum', 'mean', 'count'],
    'pago_perc': 'mean'
}).reset_index()

lucro_por_cidade.columns = ['cidade', 'lucro_total', 'lucro_medio', 'num_contratos', 'pago_perc_medio']
lucro_por_cidade = lucro_por_cidade.sort_values('lucro_total', ascending=False)

print("\n🏆 Top 10 cidades com MAIOR lucro:")
for idx, row in lucro_por_cidade.head(10).iterrows():
    print(f"   {row['cidade']}: R$ {row['lucro_total']:,.2f} ({row['num_contratos']:.0f} contratos, pago médio: {row['pago_perc_medio']*100:.1f}%)")

print("\n❌ Top 10 cidades com MAIOR prejuízo:")
# Pegar as 10 PIORES (menor lucro = maior prejuízo)
for idx, row in lucro_por_cidade.nsmallest(10, 'lucro_total').iterrows():
    print(f"   {row['cidade']}: R$ {row['lucro_total']:,.2f} ({row['num_contratos']:.0f} contratos, pago médio: {row['pago_perc_medio']*100:.1f}%)")

# ========== VISUALIZAÇÕES ==========
print("\n📊 Gerando visualizações...")

import os
output_dir = '../../graficos/analises_dataset/ANALISE_RJ_LUCRO'
os.makedirs(output_dir, exist_ok=True)

# ========== FIGURA 1: EVOLUÇÃO TEMPORAL DO LUCRO ==========
fig1 = plt.figure(figsize=(24, 14))

# 1. Lucro Total Mensal com linha de zero
ax1 = plt.subplot(3, 2, 1)
colors = ['green' if x > 0 else 'red' for x in lucro_mensal['lucro_total']]
ax1.bar(range(len(lucro_mensal)), lucro_mensal['lucro_total'], color=colors, edgecolor='black', alpha=0.7)
ax1.axhline(0, color='black', linewidth=2, linestyle='-')
ax1.set_xticks(range(0, len(lucro_mensal), max(1, len(lucro_mensal)//12)))
ax1.set_xticklabels(lucro_mensal['ano_mes'].iloc[::max(1, len(lucro_mensal)//12)], rotation=45, ha='right')
ax1.set_xlabel('Mês/Ano')
ax1.set_ylabel('Lucro Total (R$)')
ax1.set_title('Evolução Temporal do Lucro Total Mensal - RJ\n(Verde = Lucro | Vermelho = Prejuízo)', 
              fontweight='bold', fontsize=13)
ax1.grid(True, alpha=0.3, axis='y')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x/1000:.0f}k'))

# 2. Lucro Acumulado
ax2 = plt.subplot(3, 2, 2)
ax2.plot(lucro_mensal['ano_mes'], lucro_mensal['lucro_acumulado'], 
         linewidth=3, marker='o', markersize=6, color='darkblue', label='Lucro Acumulado')
ax2.axhline(0, color='red', linewidth=2, linestyle='--', alpha=0.5, label='Zero')
ax2.fill_between(range(len(lucro_mensal)), lucro_mensal['lucro_acumulado'], 0, 
                  where=(lucro_mensal['lucro_acumulado'] > 0), alpha=0.3, color='green', label='Lucro Positivo')
ax2.fill_between(range(len(lucro_mensal)), lucro_mensal['lucro_acumulado'], 0, 
                  where=(lucro_mensal['lucro_acumulado'] <= 0), alpha=0.3, color='red', label='Prejuízo')
ax2.set_xlabel('Mês/Ano')
ax2.set_ylabel('Lucro Acumulado (R$)')
ax2.set_title('Evolução do Lucro Acumulado - RJ', fontweight='bold', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best')
ax2.set_xticks(range(0, len(lucro_mensal), max(1, len(lucro_mensal)//12)))
ax2.set_xticklabels(lucro_mensal['ano_mes'].iloc[::max(1, len(lucro_mensal)//12)], rotation=45, ha='right')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x/1000:.0f}k'))

# 3. Lucro por Tipo de Cliente (Empilhado)
ax3 = plt.subplot(3, 2, 3)
ax3.plot(todos_meses, lucro_adim.values, linewidth=2.5, marker='o', markersize=5, 
         color='green', label='Adimplentes (100%)', alpha=0.8)
ax3.plot(todos_meses, lucro_inadim_parcial.values, linewidth=2.5, marker='s', markersize=5, 
         color='orange', label='Inadim. Parciais (0-100%)', alpha=0.8)
ax3.plot(todos_meses, lucro_inadim_total.values, linewidth=2.5, marker='^', markersize=5, 
         color='red', label='Inadim. Totais (0%)', alpha=0.8)
ax3.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
ax3.set_xlabel('Mês/Ano')
ax3.set_ylabel('Lucro (R$)')
ax3.set_title('Evolução do Lucro por Tipo de Cliente - RJ', fontweight='bold', fontsize=13)
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(0, len(todos_meses), max(1, len(todos_meses)//12)))
ax3.set_xticklabels([todos_meses[i] for i in range(0, len(todos_meses), max(1, len(todos_meses)//12))], 
                     rotation=45, ha='right')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x/1000:.0f}k'))

# 4. Número de Contratos por Mês
ax4 = plt.subplot(3, 2, 4)
ax4.plot(lucro_mensal['ano_mes'], lucro_mensal['num_contratos'], 
         linewidth=2.5, marker='o', markersize=6, color='steelblue')
ax4.fill_between(range(len(lucro_mensal)), lucro_mensal['num_contratos'], alpha=0.3, color='steelblue')
ax4.set_xlabel('Mês/Ano')
ax4.set_ylabel('Número de Contratos')
ax4.set_title('Volume de Contratos por Mês - RJ', fontweight='bold', fontsize=13)
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(0, len(lucro_mensal), max(1, len(lucro_mensal)//12)))
ax4.set_xticklabels(lucro_mensal['ano_mes'].iloc[::max(1, len(lucro_mensal)//12)], rotation=45, ha='right')

# 5. Lucro Médio por Contrato
ax5 = plt.subplot(3, 2, 5)
colors_medio = ['green' if x > 0 else 'red' for x in lucro_mensal['lucro_medio']]
ax5.bar(range(len(lucro_mensal)), lucro_mensal['lucro_medio'], color=colors_medio, 
        edgecolor='black', alpha=0.7)
ax5.axhline(0, color='black', linewidth=2, linestyle='-')
ax5.set_xlabel('Mês/Ano')
ax5.set_ylabel('Lucro Médio por Contrato (R$)')
ax5.set_title('Lucro Médio por Contrato - RJ', fontweight='bold', fontsize=13)
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_xticks(range(0, len(lucro_mensal), max(1, len(lucro_mensal)//12)))
ax5.set_xticklabels(lucro_mensal['ano_mes'].iloc[::max(1, len(lucro_mensal)//12)], rotation=45, ha='right')

# 6. Percentual Médio Pago
ax6 = plt.subplot(3, 2, 6)
ax6.plot(lucro_mensal['ano_mes'], lucro_mensal['pago_perc_medio'] * 100, 
         linewidth=2.5, marker='o', markersize=6, color='purple')
ax6.axhline(100, color='green', linewidth=1, linestyle='--', alpha=0.5, label='100% (Adimplente)')
ax6.axhline(50, color='orange', linewidth=1, linestyle='--', alpha=0.5, label='50%')
ax6.axhline(0, color='red', linewidth=1, linestyle='--', alpha=0.5, label='0% (Inadimplente Total)')
ax6.fill_between(range(len(lucro_mensal)), lucro_mensal['pago_perc_medio'] * 100, 50,
                  where=(lucro_mensal['pago_perc_medio'] * 100 >= 50), alpha=0.2, color='green')
ax6.fill_between(range(len(lucro_mensal)), lucro_mensal['pago_perc_medio'] * 100, 50,
                  where=(lucro_mensal['pago_perc_medio'] * 100 < 50), alpha=0.2, color='red')
ax6.set_xlabel('Mês/Ano')
ax6.set_ylabel('Percentual Médio Pago (%)')
ax6.set_title('Percentual Médio Pago por Mês - RJ', fontweight='bold', fontsize=13)
ax6.legend(loc='best')
ax6.grid(True, alpha=0.3)
ax6.set_xticks(range(0, len(lucro_mensal), max(1, len(lucro_mensal)//12)))
ax6.set_xticklabels(lucro_mensal['ano_mes'].iloc[::max(1, len(lucro_mensal)//12)], rotation=45, ha='right')

plt.suptitle('Análise Temporal do Lucro - Rio de Janeiro\nInadimplentes Parciais e Totais', 
             fontsize=18, fontweight='bold', y=0.998)
plt.tight_layout(rect=[0, 0, 1, 0.995])
plt.savefig(f'{output_dir}/evolucao_temporal_lucro_rj.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Salvo: graficos/ANALISE_RJ_LUCRO/evolucao_temporal_lucro_rj.png")

# ========== FIGURA 2: ANÁLISE POR CIDADE ==========
fig2 = plt.figure(figsize=(20, 12))

# Top 15 cidades por lucro
top_15_cidades = lucro_por_cidade.head(15)

# 1. Top 15 Cidades - Lucro Total
ax1 = plt.subplot(2, 2, 1)
colors_cidade = ['green' if x > 0 else 'red' for x in top_15_cidades['lucro_total']]
ax1.barh(range(len(top_15_cidades)), top_15_cidades['lucro_total'], 
         color=colors_cidade, edgecolor='black')
ax1.set_yticks(range(len(top_15_cidades)))
ax1.set_yticklabels(top_15_cidades['cidade'])
ax1.axvline(0, color='black', linewidth=2)
ax1.set_xlabel('Lucro Total (R$)')
ax1.set_title('Top 15 Cidades - Lucro Total', fontweight='bold', fontsize=13)
ax1.grid(True, alpha=0.3, axis='x')
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x/1000:.0f}k'))

# 2. Top 15 Cidades - Lucro Médio
ax2 = plt.subplot(2, 2, 2)
colors_medio_cidade = ['green' if x > 0 else 'red' for x in top_15_cidades['lucro_medio']]
ax2.barh(range(len(top_15_cidades)), top_15_cidades['lucro_medio'], 
         color=colors_medio_cidade, edgecolor='black')
ax2.set_yticks(range(len(top_15_cidades)))
ax2.set_yticklabels(top_15_cidades['cidade'])
ax2.axvline(0, color='black', linewidth=2)
ax2.set_xlabel('Lucro Médio por Contrato (R$)')
ax2.set_title('Top 15 Cidades - Lucro Médio por Contrato', fontweight='bold', fontsize=13)
ax2.grid(True, alpha=0.3, axis='x')

# 3. Scatter: Num Contratos vs Lucro Total
ax3 = plt.subplot(2, 2, 3)
scatter = ax3.scatter(lucro_por_cidade['num_contratos'], lucro_por_cidade['lucro_total'],
                     c=lucro_por_cidade['pago_perc_medio'], cmap='RdYlGn', 
                     s=100, alpha=0.6, edgecolors='black')
ax3.axhline(0, color='red', linewidth=1, linestyle='--', alpha=0.5)
ax3.set_xlabel('Número de Contratos')
ax3.set_ylabel('Lucro Total (R$)')
ax3.set_title('Relação: Volume de Contratos vs Lucro Total por Cidade', fontweight='bold', fontsize=13)
plt.colorbar(scatter, ax=ax3, label='% Médio Pago')
ax3.grid(True, alpha=0.3)

# 4. Top 10 piores cidades (maior prejuízo)
ax4 = plt.subplot(2, 2, 4)
bottom_10 = lucro_por_cidade.nsmallest(10, 'lucro_total')
ax4.barh(range(len(bottom_10)), bottom_10['lucro_total'], 
         color='darkred', edgecolor='black')
ax4.set_yticks(range(len(bottom_10)))
ax4.set_yticklabels(bottom_10['cidade'])
ax4.axvline(0, color='black', linewidth=2)
ax4.set_xlabel('Lucro Total (R$)')
ax4.set_title('Top 10 Cidades com MAIOR Prejuízo', fontweight='bold', fontsize=13)
ax4.grid(True, alpha=0.3, axis='x')

plt.suptitle('Análise de Lucro por Cidade - Rio de Janeiro', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(f'{output_dir}/analise_cidades_lucro_rj.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Salvo: graficos/ANALISE_RJ_LUCRO/analise_cidades_lucro_rj.png")

# ========== SALVAR DADOS ==========
print("\n💾 Salvando dados da análise...")

lucro_mensal.to_csv(
    '/home/usuario/Downloads/Documentos/top_one_model_v2/data/rj_lucro_mensal.csv',
    sep=';', encoding='utf-8', index=False
)
print(f"   ✅ Salvo: data/rj_lucro_mensal.csv")

lucro_por_cidade.to_csv(
    '/home/usuario/Downloads/Documentos/top_one_model_v2/data/rj_lucro_por_cidade.csv',
    sep=';', encoding='utf-8', index=False
)
print(f"   ✅ Salvo: data/rj_lucro_por_cidade.csv")

# ========== RESUMO FINAL ==========
print("\n" + "="*80)
print("✅ ANÁLISE DO RIO DE JANEIRO CONCLUÍDA")
print("="*80)

print(f"\n📊 Resumo Geral:")
print(f"   • Total de contratos: {len(df_rj):,}")
print(f"   • Lucro total: R$ {df_rj['lucro'].sum():,.2f}")
print(f"   • Lucro médio por contrato: R$ {df_rj['lucro'].mean():.2f}")
print(f"   • Taxa de inadimplência parcial: {len(df_rj_inadim_parcial)/len(df_rj)*100:.1f}%")

# Identificar períodos críticos
lucro_mensal_sorted = lucro_mensal.sort_values('lucro_total')
print(f"\n🔥 Período de MAIOR lucro: {lucro_mensal_sorted.iloc[-1]['ano_mes']} (R$ {lucro_mensal_sorted.iloc[-1]['lucro_total']:,.2f})")
print(f"❌ Período de MAIOR prejuízo: {lucro_mensal_sorted.iloc[0]['ano_mes']} (R$ {lucro_mensal_sorted.iloc[0]['lucro_total']:,.2f})")

# Contar meses com lucro vs prejuízo
meses_lucro = len(lucro_mensal[lucro_mensal['lucro_total'] > 0])
meses_prejuizo = len(lucro_mensal[lucro_mensal['lucro_total'] <= 0])

print(f"\n📈 Meses com lucro: {meses_lucro} ({meses_lucro/len(lucro_mensal)*100:.1f}%)")
print(f"📉 Meses com prejuízo: {meses_prejuizo} ({meses_prejuizo/len(lucro_mensal)*100:.1f}%)")

print("\n📁 Arquivos gerados:")
print("   • graficos/ANALISE_RJ_LUCRO/evolucao_temporal_lucro_rj.png")
print("   • graficos/ANALISE_RJ_LUCRO/analise_cidades_lucro_rj.png")
print("   • data/rj_lucro_mensal.csv")
print("   • data/rj_lucro_por_cidade.csv")

print("\n" + "="*80)
print("🎯 Insights:")
print("   1. RJ tem alto volume de inadimplentes mas nem todos dão prejuízo")
print("   2. Inadimplentes parciais podem ser lucrativos se pagarem o suficiente")
print("   3. Identificar períodos de prejuízo para investigar causas externas")
print("   4. Focar em cidades/períodos mais rentáveis para estratégias futuras")
print("="*80)
