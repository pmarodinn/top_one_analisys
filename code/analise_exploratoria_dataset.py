"""
================================================================================
ANÁLISE EXPLORATÓRIA DO DATASET - TOP ONE MODEL V2
================================================================================
Análise estatística completa sem predição: distribuições, frequências,
inadimplência por região/profissão, visualizações financeiras.

Autor: Sistema de Análise
Data: 16 de novembro de 2025
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurações visuais
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

# ========== CONFIGURAÇÕES ==========
DATA_PATH = Path('../data/dataset_interno_top_one_atualizado.csv')
OUTPUT_DIR = Path('../graficos/ANALISE_EXPLORATORIA')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("📊 ANÁLISE EXPLORATÓRIA DO DATASET - TOP ONE MODEL V2")
print("="*80)

# ========== 1. CARREGAR DADOS ==========
print("\n🔄 Carregando dados...")
# Carregar com separador ; e decimal ,
df = pd.read_csv(DATA_PATH, 
                 sep=';', 
                 decimal=',',
                 encoding='utf-8',
                 on_bad_lines='skip',
                 engine='python')

print(f"✓ Dataset carregado: {len(df):,} registros")
print(f"✓ Colunas: {len(df.columns)}")
print(f"✓ Memória: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Converter colunas com vírgula como decimal
cols_comma_decimal = [
    'valor_inicial_da_prestacao', 'salario_perc', 'lucro', 'IPCA',
    'Score_MC', 'idhm_2010', 'idhm_renda_2010', 
    'idhm_longevidade_2010', 'idhm_educacao_2010',
    'populacao', 'area', 'densidade_pop', 'preco_combustivel',
    'valor_cestabasica', 'preco_cb_perc', 'pago_perc', 'renda_cliente'
]
for col in cols_comma_decimal:
    if col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Filtrar apenas registros com pago_perc preenchido
df_clean = df[df['pago_perc'].notna()].copy()
print(f"✓ Após filtrar pago_perc preenchido: {len(df_clean):,} registros")

# Remover infinitos e substituir por NaN
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

# Criar variáveis derivadas
df_clean['default'] = (df_clean['pago_perc'] < 1).astype(int)
df_clean['adimplente'] = (df_clean['pago_perc'] == 1).astype(int)

# Calcular valores financeiros
# valor_financiado já existe no dataset como coluna
# Se não existir, calcular como: valor_inicial_da_prestacao * plano_financiamento
if 'valor_financiado' not in df_clean.columns:
    df_clean['valor_financiado'] = df_clean['valor_inicial_da_prestacao'] * df_clean['plano_financiamento']

# Lucro esperado caso todos fossem 100% adimplentes
# Lucro esperado = Total a receber - Valor financiado
# Total a receber = valor_inicial_da_prestacao * plano_financiamento
df_clean['total_a_receber'] = df_clean['valor_inicial_da_prestacao'] * df_clean['plano_financiamento']
df_clean['lucro_esperado'] = df_clean['total_a_receber'] - df_clean['valor_financiado']

# Lucro real (já existe no dataset como coluna 'lucro')
# Se não existir, considerar que lucro = o que foi pago - valor financiado
if 'lucro' not in df_clean.columns:
    df_clean['valor_pago'] = df_clean['total_a_receber'] * df_clean['pago_perc']
    df_clean['lucro'] = df_clean['valor_pago'] - df_clean['valor_financiado']

# Limpar infinitos novamente após cálculos
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

# Limitar valores extremos (cap)
if 'salario_perc' in df_clean.columns:
    df_clean['salario_perc'] = df_clean['salario_perc'].clip(upper=100)
if 'valor_financiado' in df_clean.columns:
    df_clean['valor_financiado'] = df_clean['valor_financiado'].clip(upper=df_clean['valor_financiado'].quantile(0.99))
if 'lucro' in df_clean.columns:
    df_clean['lucro'] = df_clean['lucro'].clip(lower=df_clean['lucro'].quantile(0.01), 
                                                  upper=df_clean['lucro'].quantile(0.99))

# ========== 2. ESTATÍSTICAS GERAIS ==========
print("\n" + "="*80)
print("📈 ESTATÍSTICAS GERAIS DO DATASET")
print("="*80)

stats_summary = {
    'Total de Contratos': len(df_clean),
    'Adimplentes (100% pago)': df_clean['adimplente'].sum(),
    'Inadimplentes (<100% pago)': df_clean['default'].sum(),
    'Taxa de Inadimplência': f"{df_clean['default'].mean()*100:.2f}%",
    'Valor Total Financiado': f"R$ {df_clean['valor_financiado'].sum():,.2f}",
    'Total a Receber (100% adimplência)': f"R$ {df_clean['total_a_receber'].sum():,.2f}",
    'Lucro Esperado (100% adimplência)': f"R$ {df_clean['lucro_esperado'].sum():,.2f}",
    'Lucro Real Total': f"R$ {df_clean['lucro'].sum():,.2f}",
    'Lucro Médio por Contrato': f"R$ {df_clean['lucro'].mean():,.2f}",
}

for key, value in stats_summary.items():
    print(f"{key:.<40} {value}")

# ========== 3. DISTRIBUIÇÃO DA VARIÁVEL ALVO ==========
print("\n" + "="*80)
print("🎯 DISTRIBUIÇÃO DA VARIÁVEL ALVO (pago_perc)")
print("="*80)

pago_stats = df_clean['pago_perc'].describe()
print("\nEstatísticas descritivas:")
print(pago_stats)

percentiles = df_clean['pago_perc'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
print("\nPercentis:")
for p, v in percentiles.items():
    print(f"  {p*100:.0f}%: {v*100:.2f}%")

# Contar valores específicos
parcialmente_pagos = ((df_clean['pago_perc'] > 0) & (df_clean['pago_perc'] < 1)).sum()
lucro_positivo_nao_100 = ((df_clean['pago_perc'] < 1) & (df_clean['lucro'] > 0)).sum()

print("\nDistribuição de valores:")
print(f"  Pagaram 0%: {(df_clean['pago_perc'] == 0).sum():,} ({(df_clean['pago_perc'] == 0).mean()*100:.2f}%)")
print(f"  Pagaram 100%: {(df_clean['pago_perc'] == 1).sum():,} ({(df_clean['pago_perc'] == 1).mean()*100:.2f}%)")
print(f"  Pagaram entre 0-100%: {parcialmente_pagos:,} ({parcialmente_pagos/len(df_clean)*100:.2f}%)")
print(f"  Inadimplentes com Lucro Positivo: {lucro_positivo_nao_100:,} ({lucro_positivo_nao_100/len(df_clean)*100:.2f}%)")

# ========== 4. VISUALIZAÇÕES - DISTRIBUIÇÕES ==========
print("\n" + "="*80)
print("📊 GERANDO VISUALIZAÇÕES - DISTRIBUIÇÕES")
print("="*80)

fig = plt.figure(figsize=(20, 12))

# 4.1 Distribuição de pago_perc
ax1 = plt.subplot(3, 3, 1)
plt.hist(df_clean['pago_perc'], bins=50, edgecolor='black', alpha=0.7)
plt.axvline(df_clean['pago_perc'].mean(), color='red', linestyle='--', label=f'Média: {df_clean["pago_perc"].mean():.2%}')
plt.axvline(df_clean['pago_perc'].median(), color='green', linestyle='--', label=f'Mediana: {df_clean["pago_perc"].median():.2%}')
plt.xlabel('Percentual Pago')
plt.ylabel('Frequência')
plt.title('Distribuição de Percentual Pago', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 4.2 Distribuição de valor_financiado
ax2 = plt.subplot(3, 3, 2)
plt.hist(df_clean['valor_financiado'], bins=50, edgecolor='black', alpha=0.7, color='orange')
plt.xlabel('Valor Financiado (R$)')
plt.ylabel('Frequência')
plt.title('Distribuição de Valor Financiado', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# 4.3 Distribuição de lucro
ax3 = plt.subplot(3, 3, 3)
plt.hist(df_clean['lucro'], bins=50, edgecolor='black', alpha=0.7, color='green')
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Lucro Zero')
plt.xlabel('Lucro (R$)')
plt.ylabel('Frequência')
plt.title('Distribuição de Lucro por Contrato', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 4.4 Distribuição de plano_financiamento (número de parcelas)
ax4 = plt.subplot(3, 3, 4)
parcelas_counts = df_clean['plano_financiamento'].value_counts().head(15)
plt.bar(range(len(parcelas_counts)), parcelas_counts.values, color='skyblue', edgecolor='black')
plt.xticks(range(len(parcelas_counts)), parcelas_counts.index, rotation=45)
plt.xlabel('Número de Parcelas')
plt.ylabel('Quantidade de Contratos')
plt.title('Top 15 - Planos de Financiamento Mais Comuns', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 4.5 Distribuição de Score_SPC
ax5 = plt.subplot(3, 3, 5)
plt.hist(df_clean['score_SPC'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='purple')
plt.xlabel('Score SPC')
plt.ylabel('Frequência')
plt.title('Distribuição de Score SPC', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# 4.6 Inadimplência vs Adimplência (Pizza)
ax6 = plt.subplot(3, 3, 6)
labels = ['Adimplente (100%)', 'Inadimplente (<100%)']
sizes = [df_clean['adimplente'].sum(), df_clean['default'].sum()]
colors = ['#66b3ff', '#ff6666']
explode = (0.05, 0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.title('Proporção Adimplentes vs Inadimplentes', fontsize=12, fontweight='bold')

# 4.7 Distribuição de valor da prestação
ax7 = plt.subplot(3, 3, 7)
prestacao_clean = df_clean['valor_inicial_da_prestacao'].dropna()
prestacao_clean = prestacao_clean[prestacao_clean > 0]  # Apenas valores positivos
plt.hist(prestacao_clean, bins=50, edgecolor='black', alpha=0.7, color='coral')
plt.xlabel('Valor da Prestação (R$)')
plt.ylabel('Frequência')
plt.title('Distribuição do Valor da Prestação', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# 4.8 Comprometimento da Renda (Prestação/Renda)
ax8 = plt.subplot(3, 3, 8)
# Calcular comprometimento real: prestação / renda
df_temp = df_clean[(df_clean['valor_inicial_da_prestacao'].notna()) & 
                    (df_clean['renda_cliente'].notna()) &
                    (df_clean['renda_cliente'] > 0)].copy()
comprometimento = (df_temp['valor_inicial_da_prestacao'] / df_temp['renda_cliente']) * 100
comprometimento = comprometimento[comprometimento <= 100]  # Limitar a 100%
plt.hist(comprometimento, bins=50, edgecolor='black', alpha=0.7, color='teal')
plt.xlabel('% da Renda Comprometida')
plt.ylabel('Frequência')
plt.title('Distribuição de % da Renda Comprometida', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# 4.9 Distribuição de Score MC
ax9 = plt.subplot(3, 3, 9)
if 'Score_MC' in df_clean.columns:
    plt.hist(df_clean['Score_MC'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='mediumpurple')
    plt.xlabel('Score MC')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Score MC', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'Score MC\nnão disponível', ha='center', va='center', fontsize=14)
    plt.axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'distribuicoes_gerais.png', dpi=300, bbox_inches='tight')
print("✓ Salvo: distribuicoes_gerais.png")
plt.close()

# ========== 5. ANÁLISE FINANCEIRA ==========
print("\n" + "="*80)
print("💰 ANÁLISE FINANCEIRA")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 5.1 Lucro Máximo Possível vs Lucro Real
total_financiado = df_clean['valor_financiado'].sum()
total_a_receber = df_clean['total_a_receber'].sum()
lucro_esperado_total = df_clean['lucro_esperado'].sum()
lucro_real_total = df_clean['lucro'].sum()

# Separar contratos lucrativos de prejuízo
contratos_lucro = df_clean[df_clean['lucro'] > 0]
contratos_prejuizo = df_clean[df_clean['lucro'] < 0]
total_lucros = contratos_lucro['lucro'].sum()
total_prejuizos = abs(contratos_prejuizo['lucro'].sum())

ax = axes[0, 0]
categories = ['Valor\nFinanciado', 'Total a\nReceber', 'Lucro\nEsperado\n(100% adim.)', 'Lucro\nReal']
values = [total_financiado, total_a_receber, lucro_esperado_total, lucro_real_total]
colors_bar = ['#ff9999', '#99ccff', '#99ff99', '#ffcc66']
bars = ax.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Valor (R$)', fontsize=11, fontweight='bold')
ax.set_title('Visão Geral Financeira - Potencial vs Real', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'R$ {val/1e6:.1f}M', ha='center', va='bottom', fontweight='bold')

# 5.2 Lucros vs Prejuízos
ax = axes[0, 1]
status_labels = [f'Lucrativos\n({len(contratos_lucro):,})', f'Prejuízo\n({len(contratos_prejuizo):,})']
lucro_values = [total_lucros, -total_prejuizos]  # Negativo para mostrar abaixo do eixo
colors_status = ['#66ff66', '#ff6666']
bars = ax.bar(status_labels, lucro_values, color=colors_status, edgecolor='black', linewidth=1.5)
ax.axhline(0, color='black', linewidth=2)
ax.set_ylabel('Lucro/Prejuízo Total (R$)', fontsize=11, fontweight='bold')
ax.set_title('Contratos Lucrativos vs Prejuízo', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, [total_lucros, total_prejuizos]):
    height = bar.get_height()
    y_pos = height if height > 0 else -height
    va = 'bottom' if height > 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2., y_pos if height > 0 else -y_pos,
            f'R$ {val/1e6:.1f}M', ha='center', va=va, fontweight='bold')

# 5.3 Distribuição de Lucro por Faixa
ax = axes[1, 0]
lucro_bins = [-np.inf, -1000, 0, 500, 1000, 2000, np.inf]
lucro_labels = ['< -R$1k', '-R$1k a 0', 'R$0 a 500', 'R$500 a 1k', 'R$1k a 2k', '> R$2k']
df_clean['lucro_faixa'] = pd.cut(df_clean['lucro'], bins=lucro_bins, labels=lucro_labels)
lucro_dist = df_clean['lucro_faixa'].value_counts().sort_index()
ax.bar(range(len(lucro_dist)), lucro_dist.values, color='steelblue', edgecolor='black')
ax.set_xticks(range(len(lucro_dist)))
ax.set_xticklabels(lucro_dist.index, rotation=45, ha='right')
ax.set_ylabel('Quantidade de Contratos', fontsize=11, fontweight='bold')
ax.set_title('Distribuição de Contratos por Faixa de Lucro', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 5.4 Receita Acumulada
ax = axes[1, 1]
df_sorted = df_clean.sort_values('lucro', ascending=False).reset_index(drop=True)
lucro_acum = df_sorted['lucro'].cumsum()
percentual_contratos = np.arange(1, len(df_sorted)+1) / len(df_sorted) * 100
ax.plot(percentual_contratos, lucro_acum / 1e6, linewidth=2, color='darkgreen')
ax.axhline(lucro_real_total / 1e6, color='red', linestyle='--', label='Lucro Total')
ax.axvline(80, color='orange', linestyle='--', label='80% dos Contratos')
ax.set_xlabel('% Contratos (ordenados por lucro)', fontsize=11, fontweight='bold')
ax.set_ylabel('Lucro Acumulado (R$ Milhões)', fontsize=11, fontweight='bold')
ax.set_title('Curva de Lucro Acumulado (Pareto)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'analise_financeira.png', dpi=300, bbox_inches='tight')
print("✓ Salvo: analise_financeira.png")
plt.close()

print(f"\nResumo Financeiro:")
print(f"  Valor Total Financiado: R$ {total_financiado:,.2f}")
print(f"  Total a Receber (100% adimplência): R$ {total_a_receber:,.2f}")
print(f"  Lucro Esperado (100% adimplência): R$ {lucro_esperado_total:,.2f}")
print(f"  Lucro Real Total: R$ {lucro_real_total:,.2f}")
print(f"  Total Lucros (contratos lucrativos): R$ {total_lucros:,.2f} ({len(contratos_lucro):,} contratos)")
print(f"  Total Prejuízos (contratos negativos): R$ {total_prejuizos:,.2f} ({len(contratos_prejuizo):,} contratos)")
print(f"  Perda vs Potencial: {(1 - lucro_real_total/lucro_esperado_total)*100:.2f}%")

# ========== 6. ANÁLISE GEOGRÁFICA ==========
print("\n" + "="*80)
print("🗺️  ANÁLISE GEOGRÁFICA")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 6.1 Top 15 Cidades por Frequência
ax = axes[0, 0]
top_cidades = df_clean['Cidade_Loja'].value_counts().head(15)
ax.barh(range(len(top_cidades)), top_cidades.values, color='skyblue', edgecolor='black')
ax.set_yticks(range(len(top_cidades)))
ax.set_yticklabels(top_cidades.index)
ax.set_xlabel('Quantidade de Contratos', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Cidades com Mais Contratos', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# 6.2 Top 15 Cidades por Taxa de Inadimplência
ax = axes[0, 1]
cidade_stats = df_clean.groupby('Cidade_Loja').agg({
    'default': ['sum', 'count', 'mean']
}).reset_index()
cidade_stats.columns = ['cidade', 'inadimplentes', 'total', 'taxa_inadimplencia']
cidade_stats = cidade_stats[cidade_stats['total'] >= 50]  # Mínimo 50 contratos
top_inad_cidades = cidade_stats.nlargest(15, 'taxa_inadimplencia')
ax.barh(range(len(top_inad_cidades)), top_inad_cidades['taxa_inadimplencia'].values * 100, 
        color='salmon', edgecolor='black')
ax.set_yticks(range(len(top_inad_cidades)))
ax.set_yticklabels(top_inad_cidades['cidade'].values)
ax.set_xlabel('Taxa de Inadimplência (%)', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Cidades com Maior Inadimplência (mín 50 contratos)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# 6.3 Distribuição por UF
ax = axes[1, 0]
uf_counts = df_clean['uf'].value_counts().head(10)
ax.bar(range(len(uf_counts)), uf_counts.values, color='lightgreen', edgecolor='black')
ax.set_xticks(range(len(uf_counts)))
ax.set_xticklabels(uf_counts.index, rotation=0)
ax.set_ylabel('Quantidade de Contratos', fontsize=11, fontweight='bold')
ax.set_xlabel('UF', fontsize=11, fontweight='bold')
ax.set_title('Top 10 Estados por Quantidade de Contratos', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 6.4 Taxa de Inadimplência por UF
ax = axes[1, 1]
uf_stats = df_clean.groupby('uf').agg({
    'default': ['sum', 'count', 'mean']
}).reset_index()
uf_stats.columns = ['uf', 'inadimplentes', 'total', 'taxa_inadimplencia']
uf_stats = uf_stats[uf_stats['total'] >= 100]  # Mínimo 100 contratos
uf_stats = uf_stats.sort_values('taxa_inadimplencia', ascending=False).head(10)
ax.bar(range(len(uf_stats)), uf_stats['taxa_inadimplencia'].values * 100, 
       color='coral', edgecolor='black')
ax.set_xticks(range(len(uf_stats)))
ax.set_xticklabels(uf_stats['uf'].values, rotation=0)
ax.set_ylabel('Taxa de Inadimplência (%)', fontsize=11, fontweight='bold')
ax.set_xlabel('UF', fontsize=11, fontweight='bold')
ax.set_title('Top 10 Estados com Maior Inadimplência (mín 100 contratos)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'analise_geografica.png', dpi=300, bbox_inches='tight')
print("✓ Salvo: analise_geografica.png")
plt.close()

# Estatísticas geográficas
print("\nTop 5 Cidades por Contratos:")
for i, (cidade, count) in enumerate(top_cidades.head().items(), 1):
    print(f"  {i}. {cidade}: {count:,} contratos")

print("\nTop 5 Cidades por Inadimplência:")
for i, row in top_inad_cidades.head().iterrows():
    print(f"  {row['cidade']}: {row['taxa_inadimplencia']*100:.2f}% ({row['inadimplentes']:.0f}/{row['total']:.0f})")

# ========== 7. ANÁLISE PROFISSIONAL ==========
print("\n" + "="*80)
print("👔 ANÁLISE POR PROFISSÃO")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 7.1 Top 15 Profissões por Frequência
ax = axes[0, 0]
top_prof = df_clean['descricao_da_Profissao'].value_counts().head(15)
ax.barh(range(len(top_prof)), top_prof.values, color='plum', edgecolor='black')
ax.set_yticks(range(len(top_prof)))
ax.set_yticklabels([p[:30] + '...' if len(p) > 30 else p for p in top_prof.index])
ax.set_xlabel('Quantidade de Contratos', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Profissões com Mais Contratos', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# 7.2 Top 15 Profissões por Taxa de Inadimplência
ax = axes[0, 1]
prof_stats = df_clean.groupby('descricao_da_Profissao').agg({
    'default': ['sum', 'count', 'mean']
}).reset_index()
prof_stats.columns = ['profissao', 'inadimplentes', 'total', 'taxa_inadimplencia']
prof_stats = prof_stats[prof_stats['total'] >= 50]  # Mínimo 50 contratos
top_inad_prof = prof_stats.nlargest(15, 'taxa_inadimplencia')
ax.barh(range(len(top_inad_prof)), top_inad_prof['taxa_inadimplencia'].values * 100, 
        color='indianred', edgecolor='black')
ax.set_yticks(range(len(top_inad_prof)))
ax.set_yticklabels([p[:30] + '...' if len(p) > 30 else p for p in top_inad_prof['profissao'].values])
ax.set_xlabel('Taxa de Inadimplência (%)', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Profissões com Maior Inadimplência (mín 50 contratos)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# 7.3 Top 15 Profissões com Menor Inadimplência
ax = axes[1, 0]
top_adim_prof = prof_stats.nsmallest(15, 'taxa_inadimplencia')
ax.barh(range(len(top_adim_prof)), top_adim_prof['taxa_inadimplencia'].values * 100, 
        color='lightgreen', edgecolor='black')
ax.set_yticks(range(len(top_adim_prof)))
ax.set_yticklabels([p[:30] + '...' if len(p) > 30 else p for p in top_adim_prof['profissao'].values])
ax.set_xlabel('Taxa de Inadimplência (%)', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Profissões com Menor Inadimplência (mín 50 contratos)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# 7.4 Lucro Médio por Profissão (Top 15)
ax = axes[1, 1]
prof_lucro = df_clean.groupby('descricao_da_Profissao').agg({
    'lucro': ['mean', 'count']
}).reset_index()
prof_lucro.columns = ['profissao', 'lucro_medio', 'total']
prof_lucro = prof_lucro[prof_lucro['total'] >= 50]
top_lucro_prof = prof_lucro.nlargest(15, 'lucro_medio')
ax.barh(range(len(top_lucro_prof)), top_lucro_prof['lucro_medio'].values, 
        color='gold', edgecolor='black')
ax.set_yticks(range(len(top_lucro_prof)))
ax.set_yticklabels([p[:30] + '...' if len(p) > 30 else p for p in top_lucro_prof['profissao'].values])
ax.set_xlabel('Lucro Médio (R$)', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Profissões por Lucro Médio (mín 50 contratos)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'analise_profissao.png', dpi=300, bbox_inches='tight')
print("✓ Salvo: analise_profissao.png")
plt.close()

print("\nTop 5 Profissões por Contratos:")
for i, (prof, count) in enumerate(top_prof.head().items(), 1):
    print(f"  {i}. {prof[:50]}: {count:,} contratos")

# ========== 7.5. ANÁLISE INADIMPLENTES POR TIPO (% NORMALIZADO) ==========
print("\n" + "="*80)
print("🔍 ANÁLISE DE INADIMPLENTES POR TIPO - % NORMALIZADO (Profissões, Cidades e UF)")
print("="*80)

fig, axes = plt.subplots(3, 2, figsize=(20, 20))

# Calcular totais por profissão, cidade e UF
total_por_prof = df_clean.groupby('descricao_da_Profissao').size()
total_por_cidade = df_clean.groupby('Cidade_Loja').size()
total_por_uf = df_clean.groupby('uf').size()

# 7.5.1 Top 15 Profissões com Maior % de Inadimplentes Totais (0% pago)
ax = axes[0, 0]
prof_inad_zero_count = df_clean[df_clean['pago_perc'] == 0].groupby('descricao_da_Profissao').size()
prof_inad_zero_pct = (prof_inad_zero_count / total_por_prof * 100).reset_index(name='percentual')
# Filtrar profissões com pelo menos 50 contratos para ter significância estatística
prof_com_volume = total_por_prof[total_por_prof >= 50].index
prof_inad_zero_pct = prof_inad_zero_pct[prof_inad_zero_pct['descricao_da_Profissao'].isin(prof_com_volume)]
prof_inad_zero_pct = prof_inad_zero_pct.nlargest(15, 'percentual')
ax.barh(range(len(prof_inad_zero_pct)), prof_inad_zero_pct['percentual'].values, color='darkred', edgecolor='black')
ax.set_yticks(range(len(prof_inad_zero_pct)))
ax.set_yticklabels([p[:30] + '...' if len(p) > 30 else p for p in prof_inad_zero_pct['descricao_da_Profissao'].values])
ax.set_xlabel('% de Inadimplentes (0% pago)', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Profissões - Maior % Inadimplentes Totais (≥50 contratos)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
# Adicionar valores percentuais
for i, (idx, row) in enumerate(prof_inad_zero_pct.iterrows()):
    prof = row['descricao_da_Profissao']
    pct = row['percentual']
    total = total_por_prof[prof]
    ax.text(pct, i, f' {pct:.1f}% ({int(prof_inad_zero_count[prof])}/{int(total)})', va='center', fontsize=8, fontweight='bold')

# 7.5.2 Top 15 Profissões com Maior % de Inadimplentes Lucrativos (<100% mas lucro>0)
ax = axes[0, 1]
prof_inad_lucro_count = df_clean[(df_clean['pago_perc'] < 1) & (df_clean['lucro'] > 0)].groupby('descricao_da_Profissao').size()
prof_inad_lucro_pct = (prof_inad_lucro_count / total_por_prof * 100).reset_index(name='percentual')
prof_inad_lucro_pct = prof_inad_lucro_pct[prof_inad_lucro_pct['descricao_da_Profissao'].isin(prof_com_volume)]
prof_inad_lucro_pct = prof_inad_lucro_pct.nlargest(15, 'percentual')
ax.barh(range(len(prof_inad_lucro_pct)), prof_inad_lucro_pct['percentual'].values, color='darkorange', edgecolor='black')
ax.set_yticks(range(len(prof_inad_lucro_pct)))
ax.set_yticklabels([p[:30] + '...' if len(p) > 30 else p for p in prof_inad_lucro_pct['descricao_da_Profissao'].values])
ax.set_xlabel('% de Inadimplentes Lucrativos', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Profissões - Maior % Inadimplentes Lucrativos (≥50 contratos)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
# Adicionar valores percentuais
for i, (idx, row) in enumerate(prof_inad_lucro_pct.iterrows()):
    prof = row['descricao_da_Profissao']
    pct = row['percentual']
    total = total_por_prof[prof]
    ax.text(pct, i, f' {pct:.1f}% ({int(prof_inad_lucro_count[prof])}/{int(total)})', va='center', fontsize=8, fontweight='bold')

# 7.5.3 Top 15 Cidades com Maior % de Inadimplentes Totais (0% pago)
ax = axes[1, 0]
cidade_inad_zero_count = df_clean[df_clean['pago_perc'] == 0].groupby('Cidade_Loja').size()
cidade_inad_zero_pct = (cidade_inad_zero_count / total_por_cidade * 100).reset_index(name='percentual')
# Filtrar cidades com pelo menos 100 contratos para ter significância estatística
cidade_com_volume = total_por_cidade[total_por_cidade >= 100].index
cidade_inad_zero_pct = cidade_inad_zero_pct[cidade_inad_zero_pct['Cidade_Loja'].isin(cidade_com_volume)]
cidade_inad_zero_pct = cidade_inad_zero_pct.nlargest(15, 'percentual')
ax.barh(range(len(cidade_inad_zero_pct)), cidade_inad_zero_pct['percentual'].values, color='crimson', edgecolor='black')
ax.set_yticks(range(len(cidade_inad_zero_pct)))
ax.set_yticklabels(cidade_inad_zero_pct['Cidade_Loja'].values)
ax.set_xlabel('% de Inadimplentes (0% pago)', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Cidades - Maior % Inadimplentes Totais (≥100 contratos)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
# Adicionar valores percentuais
for i, (idx, row) in enumerate(cidade_inad_zero_pct.iterrows()):
    cidade = row['Cidade_Loja']
    pct = row['percentual']
    total = total_por_cidade[cidade]
    ax.text(pct, i, f' {pct:.1f}% ({int(cidade_inad_zero_count[cidade])}/{int(total)})', va='center', fontsize=8, fontweight='bold')

# 7.5.4 Top 15 Cidades com Maior % de Inadimplentes Lucrativos (<100% mas lucro>0)
ax = axes[1, 1]
cidade_inad_lucro_count = df_clean[(df_clean['pago_perc'] < 1) & (df_clean['lucro'] > 0)].groupby('Cidade_Loja').size()
cidade_inad_lucro_pct = (cidade_inad_lucro_count / total_por_cidade * 100).reset_index(name='percentual')
cidade_inad_lucro_pct = cidade_inad_lucro_pct[cidade_inad_lucro_pct['Cidade_Loja'].isin(cidade_com_volume)]
cidade_inad_lucro_pct = cidade_inad_lucro_pct.nlargest(15, 'percentual')
ax.barh(range(len(cidade_inad_lucro_pct)), cidade_inad_lucro_pct['percentual'].values, color='gold', edgecolor='black')
ax.set_yticks(range(len(cidade_inad_lucro_pct)))
ax.set_yticklabels(cidade_inad_lucro_pct['Cidade_Loja'].values)
ax.set_xlabel('% de Inadimplentes Lucrativos', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Cidades - Maior % Inadimplentes Lucrativos (≥100 contratos)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
# Adicionar valores percentuais
for i, (idx, row) in enumerate(cidade_inad_lucro_pct.iterrows()):
    cidade = row['Cidade_Loja']
    pct = row['percentual']
    total = total_por_cidade[cidade]
    ax.text(pct, i, f' {pct:.1f}% ({int(cidade_inad_lucro_count[cidade])}/{int(total)})', va='center', fontsize=8, fontweight='bold')

# 7.5.5 UFs com Maior % de Inadimplentes Totais (0% pago)
ax = axes[2, 0]
uf_inad_zero_count = df_clean[df_clean['pago_perc'] == 0].groupby('uf').size()
uf_inad_zero_pct = (uf_inad_zero_count / total_por_uf * 100).reset_index(name='percentual')
# Filtrar UFs com pelo menos 200 contratos para ter significância estatística
uf_com_volume = total_por_uf[total_por_uf >= 200].index
uf_inad_zero_pct = uf_inad_zero_pct[uf_inad_zero_pct['uf'].isin(uf_com_volume)]
uf_inad_zero_pct = uf_inad_zero_pct.sort_values('percentual', ascending=False)
ax.barh(range(len(uf_inad_zero_pct)), uf_inad_zero_pct['percentual'].values, color='darkred', edgecolor='black', alpha=0.8)
ax.set_yticks(range(len(uf_inad_zero_pct)))
ax.set_yticklabels(uf_inad_zero_pct['uf'].values)
ax.set_xlabel('% de Inadimplentes (0% pago)', fontsize=11, fontweight='bold')
ax.set_title('UFs - Maior % Inadimplentes Totais (≥200 contratos)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
# Adicionar valores percentuais
for i, (idx, row) in enumerate(uf_inad_zero_pct.iterrows()):
    uf = row['uf']
    pct = row['percentual']
    total = total_por_uf[uf]
    ax.text(pct, i, f' {pct:.1f}% ({int(uf_inad_zero_count[uf])}/{int(total)})', va='center', fontsize=9, fontweight='bold')

# 7.5.6 UFs com Maior % de Inadimplentes Lucrativos (<100% mas lucro>0)
ax = axes[2, 1]
uf_inad_lucro_count = df_clean[(df_clean['pago_perc'] < 1) & (df_clean['lucro'] > 0)].groupby('uf').size()
uf_inad_lucro_pct = (uf_inad_lucro_count / total_por_uf * 100).reset_index(name='percentual')
uf_inad_lucro_pct = uf_inad_lucro_pct[uf_inad_lucro_pct['uf'].isin(uf_com_volume)]
uf_inad_lucro_pct = uf_inad_lucro_pct.sort_values('percentual', ascending=False)
ax.barh(range(len(uf_inad_lucro_pct)), uf_inad_lucro_pct['percentual'].values, color='darkorange', edgecolor='black', alpha=0.8)
ax.set_yticks(range(len(uf_inad_lucro_pct)))
ax.set_yticklabels(uf_inad_lucro_pct['uf'].values)
ax.set_xlabel('% de Inadimplentes Lucrativos', fontsize=11, fontweight='bold')
ax.set_title('UFs - Maior % Inadimplentes Lucrativos (≥200 contratos)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
# Adicionar valores percentuais
for i, (idx, row) in enumerate(uf_inad_lucro_pct.iterrows()):
    uf = row['uf']
    pct = row['percentual']
    total = total_por_uf[uf]
    ax.text(pct, i, f' {pct:.1f}% ({int(uf_inad_lucro_count[uf])}/{int(total)})', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'analise_inadimplentes_2.png', dpi=300, bbox_inches='tight')
print("✓ Salvo: analise_inadimplentes_2.png")
plt.close()

# Estatísticas sobre esses grupos
print("\n📊 Estatísticas - Inadimplentes por Tipo (% Normalizado):")
print(f"\nProfissões (com ≥50 contratos):")
if len(prof_inad_zero_pct) > 0:
    top_prof_zero = prof_inad_zero_pct.iloc[0]
    prof_name = top_prof_zero['descricao_da_Profissao']
    print(f"  Top profissão com Inadimpl. Totais (0%): {prof_name[:50]}")
    print(f"    - {top_prof_zero['percentual']:.1f}% ({int(prof_inad_zero_count[prof_name])}/{int(total_por_prof[prof_name])} contratos)")
if len(prof_inad_lucro_pct) > 0:
    top_prof_lucro = prof_inad_lucro_pct.iloc[0]
    prof_name = top_prof_lucro['descricao_da_Profissao']
    print(f"  Top profissão com Inadimpl. Lucrativos: {prof_name[:50]}")
    print(f"    - {top_prof_lucro['percentual']:.1f}% ({int(prof_inad_lucro_count[prof_name])}/{int(total_por_prof[prof_name])} contratos)")
print(f"\nCidades (com ≥100 contratos):")
if len(cidade_inad_zero_pct) > 0:
    top_cidade_zero = cidade_inad_zero_pct.iloc[0]
    cidade_name = top_cidade_zero['Cidade_Loja']
    print(f"  Top cidade com Inadimpl. Totais (0%): {cidade_name}")
    print(f"    - {top_cidade_zero['percentual']:.1f}% ({int(cidade_inad_zero_count[cidade_name])}/{int(total_por_cidade[cidade_name])} contratos)")
if len(cidade_inad_lucro_pct) > 0:
    top_cidade_lucro = cidade_inad_lucro_pct.iloc[0]
    cidade_name = top_cidade_lucro['Cidade_Loja']
    print(f"  Top cidade com Inadimpl. Lucrativos: {cidade_name}")
    print(f"    - {top_cidade_lucro['percentual']:.1f}% ({int(cidade_inad_lucro_count[cidade_name])}/{int(total_por_cidade[cidade_name])} contratos)")
print(f"\nUFs (com ≥200 contratos):")
if len(uf_inad_zero_pct) > 0:
    top_uf_zero = uf_inad_zero_pct.iloc[0]
    uf_name = top_uf_zero['uf']
    print(f"  Top UF com Inadimpl. Totais (0%): {uf_name}")
    print(f"    - {top_uf_zero['percentual']:.1f}% ({int(uf_inad_zero_count[uf_name])}/{int(total_por_uf[uf_name])} contratos)")
if len(uf_inad_lucro_pct) > 0:
    top_uf_lucro = uf_inad_lucro_pct.iloc[0]
    uf_name = top_uf_lucro['uf']
    print(f"  Top UF com Inadimpl. Lucrativos: {uf_name}")
    print(f"    - {top_uf_lucro['percentual']:.1f}% ({int(uf_inad_lucro_count[uf_name])}/{int(total_por_uf[uf_name])} contratos)")

# ========== 8. ANÁLISE PARCELAS vs INADIMPLÊNCIA ==========
print("\n" + "="*80)
print("📅 ANÁLISE NÚMERO DE PARCELAS vs INADIMPLÊNCIA")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(22, 12))

# 8.1 Distribuição de Parcelas - Adimplentes
ax = axes[0, 0]
parcelas_adim = df_clean[df_clean['adimplente'] == 1]['plano_financiamento'].value_counts().head(20)
ax.bar(range(len(parcelas_adim)), parcelas_adim.values, color='mediumseagreen', edgecolor='black')
ax.set_xticks(range(len(parcelas_adim)))
ax.set_xticklabels(parcelas_adim.index, rotation=45)
ax.set_ylabel('Quantidade de Contratos', fontsize=11, fontweight='bold')
ax.set_xlabel('Número de Parcelas', fontsize=11, fontweight='bold')
ax.set_title('Distribuição de Parcelas - ADIMPLENTES (Top 20)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 8.2 Distribuição de Parcelas - Inadimplentes
ax = axes[0, 1]
parcelas_inad = df_clean[df_clean['default'] == 1]['plano_financiamento'].value_counts().head(20)
ax.bar(range(len(parcelas_inad)), parcelas_inad.values, color='tomato', edgecolor='black')
ax.set_xticks(range(len(parcelas_inad)))
ax.set_xticklabels(parcelas_inad.index, rotation=45)
ax.set_ylabel('Quantidade de Contratos', fontsize=11, fontweight='bold')
ax.set_xlabel('Número de Parcelas', fontsize=11, fontweight='bold')
ax.set_title('Distribuição de Parcelas - INADIMPLENTES (Top 20)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 8.3 Curvas de Inadimplência vs Adimplência por Faixa de Parcelas
ax = axes[0, 2]
parcelas_bins = [0, 12, 24, 36, 48, 60, 1000]
parcelas_labels = ['1-12', '13-24', '25-36', '37-48', '49-60', '60+']
df_clean['parcelas_faixa'] = pd.cut(df_clean['plano_financiamento'], bins=parcelas_bins, labels=parcelas_labels)

# Calcular estatísticas por faixa - AGORA inadimplência total = apenas quem pagou 0%
inad_total_por_faixa = []
adim_por_faixa = []
inad_lucrativos_por_faixa = []
inad_nao_lucrativos_por_faixa = []

for faixa in parcelas_labels:
    df_faixa = df_clean[df_clean['parcelas_faixa'] == faixa]
    total_faixa = len(df_faixa)
    
    # Inadimplência Total = APENAS quem pagou 0% (nunca pagou nada)
    inad_total_0perc = len(df_faixa[df_faixa['pago_perc'] == 0])
    
    # Adimplentes = 100% pago
    adimplentes = len(df_faixa[df_faixa['pago_perc'] == 1])
    
    # Inadimplentes com lucro positivo (pago entre 0-100% mas lucro > 0)
    inad_lucrativos = len(df_faixa[(df_faixa['pago_perc'] > 0) & (df_faixa['pago_perc'] < 1) & (df_faixa['lucro'] > 0)])
    
    # Inadimplentes com lucro negativo (pago entre 0-100% e lucro <= 0)
    inad_nao_lucrativos = len(df_faixa[(df_faixa['pago_perc'] > 0) & (df_faixa['pago_perc'] < 1) & (df_faixa['lucro'] <= 0)])
    
    taxa_inad_total = (inad_total_0perc / total_faixa * 100) if total_faixa > 0 else 0
    taxa_adim = (adimplentes / total_faixa * 100) if total_faixa > 0 else 0
    taxa_luc = (inad_lucrativos / total_faixa * 100) if total_faixa > 0 else 0
    taxa_nao_luc = (inad_nao_lucrativos / total_faixa * 100) if total_faixa > 0 else 0
    
    inad_total_por_faixa.append(taxa_inad_total)
    adim_por_faixa.append(taxa_adim)
    inad_lucrativos_por_faixa.append(taxa_luc)
    inad_nao_lucrativos_por_faixa.append(taxa_nao_luc)

x_pos = range(len(parcelas_labels))
# Curva de inadimplência total (pagou 0% - nunca pagou)
ax.plot(x_pos, inad_total_por_faixa, marker='o', linewidth=2.5, 
        markersize=10, color='#ff0000', label='Inadimpl. Total (0% pago)', linestyle='-', zorder=3)
# Curva de adimplência (100%)
ax.plot(x_pos, adim_por_faixa, marker='s', linewidth=2.5, 
        markersize=10, color='#00cc00', label='Adimplência (100%)', linestyle='-', zorder=3)
# Curva de inadimplentes lucrativos (pagaram parcialmente com lucro positivo)
ax.plot(x_pos, inad_lucrativos_por_faixa, marker='D', linewidth=2.5, 
        markersize=9, color='#ffaa00', label='Inad. Parcial Lucrativos', linestyle='--', zorder=4)
# Curva de inadimplentes NÃO lucrativos (pagaram parcialmente com prejuízo)
ax.plot(x_pos, inad_nao_lucrativos_por_faixa, marker='v', linewidth=2.5, 
        markersize=9, color='#cc0066', label='Inad. Parcial Prejuízo', linestyle='--', zorder=4)

ax.set_xticks(x_pos)
ax.set_xticklabels(parcelas_labels, rotation=0)
ax.set_ylabel('Taxa (%)', fontsize=11, fontweight='bold')
ax.set_xlabel('Faixa de Parcelas', fontsize=11, fontweight='bold')
ax.set_title('Inadimplência: Total (0%) vs Adimplência (100%) vs Parciais', fontsize=11, fontweight='bold')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 105])

# Adicionar valores nos pontos principais
for i, (inad_total, adim) in enumerate(zip(inad_total_por_faixa, adim_por_faixa)):
    ax.text(i, inad_total + 2, f'{inad_total:.1f}%', ha='center', va='bottom', fontsize=7, color='#cc0000', fontweight='bold')
    ax.text(i, adim - 2, f'{adim:.1f}%', ha='center', va='top', fontsize=7, color='#00cc00', fontweight='bold')

# 8.4 Taxa de Inadimplência por Faixa de Parcelas (Barras) - 0% pagamento apenas
ax = axes[1, 0]
ax.bar(range(len(parcelas_labels)), inad_total_por_faixa, color='darkred', edgecolor='black')
ax.set_xticks(range(len(parcelas_labels)))
ax.set_xticklabels(parcelas_labels, rotation=0)
ax.set_ylabel('Taxa de Inadimplência Total (%)', fontsize=11, fontweight='bold')
ax.set_xlabel('Faixa de Parcelas', fontsize=11, fontweight='bold')
ax.set_title('Taxa de Inadimplência Total (0% pago) por Faixa', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 8.5 Lucro Médio por Faixa de Parcelas
ax = axes[1, 1]
faixa_lucro = df_clean.groupby('parcelas_faixa')['lucro'].mean()
ax.bar(range(len(faixa_lucro)), faixa_lucro.values, color='orange', edgecolor='black')
ax.set_xticks(range(len(faixa_lucro)))
ax.set_xticklabels(faixa_lucro.index, rotation=0)
ax.set_ylabel('Lucro Médio (R$)', fontsize=11, fontweight='bold')
ax.set_xlabel('Faixa de Parcelas', fontsize=11, fontweight='bold')
ax.set_title('Lucro Médio por Faixa de Parcelas', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 8.6 Quantidade de Contratos por Faixa
ax = axes[1, 2]
# Calcular totais por faixa
faixa_totals = []
for faixa in parcelas_labels:
    total_faixa = len(df_clean[df_clean['parcelas_faixa'] == faixa])
    faixa_totals.append(total_faixa)

ax.bar(range(len(parcelas_labels)), faixa_totals, color='mediumpurple', edgecolor='black')
ax.set_xticks(range(len(parcelas_labels)))
ax.set_xticklabels(parcelas_labels, rotation=0)
ax.set_ylabel('Quantidade de Contratos', fontsize=11, fontweight='bold')
ax.set_xlabel('Faixa de Parcelas', fontsize=11, fontweight='bold')
ax.set_title('Volume de Contratos por Faixa de Parcelas', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
# Adicionar valores nas barras
for i, val in enumerate(faixa_totals):
    ax.text(i, val, f'{val:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'analise_parcelas.png', dpi=300, bbox_inches='tight')
print("✓ Salvo: analise_parcelas.png")
plt.close()

print("\nTaxa de Inadimplência por Faixa de Parcelas:")
for idx, faixa in enumerate(parcelas_labels):
    total_faixa = faixa_totals[idx]
    taxa_inad = inad_total_por_faixa[idx]
    taxa_adim = adim_por_faixa[idx]
    taxa_luc = inad_lucrativos_por_faixa[idx]
    taxa_nao_luc = inad_nao_lucrativos_por_faixa[idx]
    
    # Calcular números absolutos
    num_inad_total = int(total_faixa * taxa_inad / 100)
    num_adim = int(total_faixa * taxa_adim / 100)
    
    print(f"\n  Faixa {faixa}:")
    print(f"    Inadimplência Total (0% pago): {taxa_inad:.2f}% ({num_inad_total}/{total_faixa})")
    print(f"    Adimplência (100%): {taxa_adim:.2f}% ({num_adim}/{total_faixa})")
    print(f"    Inadimpl. Parcial Lucrativos (0-100%, lucro>0): {taxa_luc:.2f}%")
    print(f"    Inadimpl. Parcial Prejuízo (0-100%, lucro≤0): {taxa_nao_luc:.2f}%")

# Estatísticas gerais sobre inadimplência
print("\n" + "="*80)
print("📊 ANÁLISE DETALHADA - INADIMPLENTES")

print("="*80)
total_contratos = len(df_clean)
inad_total = (df_clean['pago_perc'] < 1).sum()
inad_zero = (df_clean['pago_perc'] == 0).sum()
inad_parcial = ((df_clean['pago_perc'] > 0) & (df_clean['pago_perc'] < 1)).sum()
inad_lucrativos = ((df_clean['pago_perc'] < 1) & (df_clean['lucro'] > 0)).sum()
inad_prejuizo = ((df_clean['pago_perc'] < 1) & (df_clean['lucro'] <= 0)).sum()

print(f"\nInadimplentes (< 100% pago): {inad_total:,} ({inad_total/total_contratos*100:.2f}%)")
print(f"  • Pagaram 0% (nunca pagaram): {inad_zero:,} ({inad_zero/total_contratos*100:.2f}%)")
print(f"  • Pagaram parcialmente (0-100%): {inad_parcial:,} ({inad_parcial/total_contratos*100:.2f}%)")
print(f"\nPor Resultado Financeiro:")
print(f"  • Inadimpl. com Lucro POSITIVO: {inad_lucrativos:,} ({inad_lucrativos/total_contratos*100:.2f}%)")
print(f"  • Inadimpl. com Lucro NEGATIVO/ZERO: {inad_prejuizo:,} ({inad_prejuizo/total_contratos*100:.2f}%)")
print(f"\nVerificação: Inadimpl. Total = {inad_total:,} | Lucrativos + Prejuízo = {inad_lucrativos + inad_prejuizo:,}")

# ========== 9. CORRELAÇÕES E HEATMAP ==========
print("\n" + "="*80)
print("🔗 ANÁLISE DE CORRELAÇÕES")
print("="*80)

# Selecionar colunas numéricas relevantes
numeric_cols = ['pago_perc', 'valor_inicial_da_prestacao', 'plano_financiamento', 
                'valor_financiado', 'lucro', 'renda_cliente', 'salario_perc', 
                'score_SPC', 'Score_MC', 'default']
df_corr = df_clean[numeric_cols].dropna()

fig, ax = plt.subplots(figsize=(14, 10))
correlation_matrix = df_corr.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlação - Variáveis Numéricas', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlacoes.png', dpi=300, bbox_inches='tight')
print("✓ Salvo: correlacoes.png")
plt.close()

# Correlações mais fortes com default
print("\nCorrelações mais fortes com inadimplência (default):")
corr_default = correlation_matrix['default'].sort_values(ascending=False)
for var, corr in corr_default.items():
    if var != 'default' and abs(corr) > 0.05:
        print(f"  {var}: {corr:.3f}")

# ========== 10. RELATÓRIO RESUMO ==========
print("\n" + "="*80)
print("📄 GERANDO RELATÓRIO RESUMO")
print("="*80)

with open(OUTPUT_DIR / 'relatorio_analise_exploratoria.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("RELATÓRIO DE ANÁLISE EXPLORATÓRIA - TOP ONE MODEL V2\n")
    f.write("="*80 + "\n\n")
    
    f.write("Data: 16 de novembro de 2025\n")
    f.write(f"Total de Registros Analisados: {len(df_clean):,}\n\n")
    
    f.write("--- ESTATÍSTICAS GERAIS ---\n")
    for key, value in stats_summary.items():
        f.write(f"{key}: {value}\n")
    
    f.write("\n--- VARIÁVEL ALVO (pago_perc) ---\n")
    f.write(f"Média: {df_clean['pago_perc'].mean()*100:.2f}%\n")
    f.write(f"Mediana: {df_clean['pago_perc'].median()*100:.2f}%\n")
    f.write(f"Desvio Padrão: {df_clean['pago_perc'].std()*100:.2f}%\n")
    f.write(f"Contratos com 100% pago: {(df_clean['pago_perc'] == 1).sum():,} ({(df_clean['pago_perc'] == 1).mean()*100:.2f}%)\n")
    f.write(f"Contratos com 0% pago: {(df_clean['pago_perc'] == 0).sum():,} ({(df_clean['pago_perc'] == 0).mean()*100:.2f}%)\n")
    
    f.write("\n--- TOP 5 CIDADES ---\n")
    for i, (cidade, count) in enumerate(top_cidades.head().items(), 1):
        f.write(f"{i}. {cidade}: {count:,} contratos\n")
    
    f.write("\n--- TOP 5 PROFISSÕES ---\n")
    for i, (prof, count) in enumerate(top_prof.head().items(), 1):
        f.write(f"{i}. {prof}: {count:,} contratos\n")
    
    f.write("\n--- INADIMPLÊNCIA POR FAIXA DE PARCELAS (APENAS 0% PAGO) ---\n")
    for idx, faixa in enumerate(parcelas_labels):
        total_faixa = faixa_totals[idx]
        taxa_inad = inad_total_por_faixa[idx]
        num_inad = int(total_faixa * taxa_inad / 100)
        f.write(f"{faixa}: {taxa_inad:.2f}% ({num_inad}/{total_faixa})\n")
    
    f.write("\n--- CORRELAÇÕES COM INADIMPLÊNCIA ---\n")
    for var, corr in corr_default.items():
        if var != 'default' and abs(corr) > 0.05:
            f.write(f"{var}: {corr:.3f}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("Arquivos Gerados:\n")
    f.write("  - distribuicoes_gerais.png\n")
    f.write("  - analise_financeira.png\n")
    f.write("  - analise_geografica.png\n")
    f.write("  - analise_profissao.png\n")
    f.write("  - analise_parcelas.png\n")
    f.write("  - correlacoes.png\n")
    f.write("  - relatorio_analise_exploratoria.txt\n")

print("✓ Salvo: relatorio_analise_exploratoria.txt")

# ========== FIM ==========
print("\n" + "="*80)
print("✅ ANÁLISE EXPLORATÓRIA COMPLETA!")
print("="*80)
print(f"\n📁 Arquivos salvos em: {OUTPUT_DIR}")
print("\nArquivos gerados:")
print("  1. distribuicoes_gerais.png - 9 gráficos de distribuição")
print("  2. analise_financeira.png - Visão geral financeira")
print("  3. analise_geografica.png - Análise por cidade e UF")
print("  4. analise_profissao.png - Análise por profissão")
print("  5. analise_parcelas.png - Parcelas vs inadimplência")
print("  6. correlacoes.png - Heatmap de correlações")
print("  7. relatorio_analise_exploratoria.txt - Resumo textual")

print("\n🎯 Principais Descobertas:")
print(f"  • {df_clean['adimplente'].sum():,} contratos adimplentes ({df_clean['adimplente'].mean()*100:.1f}%)")
print(f"  • {df_clean['default'].sum():,} contratos inadimplentes ({df_clean['default'].mean()*100:.1f}%)")
print(f"  • Lucro real total: R$ {lucro_real_total:,.2f}")
print(f"  • Lucro esperado (100% adimplência): R$ {lucro_esperado_total:,.2f}")
print(f"  • Cidade com mais contratos: {top_cidades.index[0]} ({top_cidades.values[0]:,})")
print(f"  • Profissão mais comum: {top_prof.index[0][:50]} ({top_prof.values[0]:,})")

print("\n" + "="*80)
