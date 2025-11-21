"""
Análise Temporal de Inadimplência Parcial
==========================================
Este script analisa quando clientes que pagaram parcialmente pararam de pagar,
identificando padrões temporais e regionais de inadimplência.

Foco: Clientes com 0 < pago_perc < 1 (pagaram algo, mas não tudo)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

print("="*80)
print("🔍 ANÁLISE TEMPORAL DE INADIMPLÊNCIA PARCIAL")
print("="*80)

# ========== CARREGAMENTO DOS DADOS ==========
print("\n📂 Carregando dados...")
df = pd.read_csv('/home/usuario/Downloads/Documentos/top_one_model_v2/data/dataset_interno_top_one_atualizado.csv', 
                 sep=';', encoding='utf-8')

print(f"   Total de registros: {len(df):,}")

# ========== CONVERSÃO DE DATAS ==========
print("\n📅 Convertendo datas...")
df['Data_inicio'] = pd.to_datetime(df['Data_inicio'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df['Data_Lancamento'] = pd.to_datetime(df['Data_Lancamento'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Remover registros sem data
df = df.dropna(subset=['Data_inicio'])
print(f"   Registros com data válida: {len(df):,}")

# ========== CONVERSÃO DE COLUNAS NUMÉRICAS ==========
print("\n🔢 Convertendo colunas numéricas...")
# Converter pago_perc para float (pode estar como string com vírgula)
df['pago_perc'] = df['pago_perc'].astype(str).str.replace(',', '.').astype(float)
df['plano_financiamento'] = pd.to_numeric(df['plano_financiamento'], errors='coerce')
df['lucro'] = df['lucro'].astype(str).str.replace(',', '.').astype(float)

print(f"   ✅ Conversões concluídas")

# ========== FILTRAR APENAS INADIMPLENTES PARCIAIS ==========
print("\n🎯 Filtrando inadimplentes parciais (0 < pago_perc < 1)...")
df_parcial = df[(df['pago_perc'] > 0) & (df['pago_perc'] < 1)].copy()

print(f"   Inadimplentes parciais: {len(df_parcial):,} ({len(df_parcial)/len(df)*100:.1f}% do total)")
print(f"   Pagamento médio: {df_parcial['pago_perc'].mean()*100:.1f}%")
print(f"   Pagamento mediano: {df_parcial['pago_perc'].median()*100:.1f}%")

# ========== CALCULAR QUANDO PARARAM DE PAGAR ==========
print("\n⏱️  Calculando quando pararam de pagar...")

def estimar_mes_parada(row):
    """
    Estima em qual mês o cliente parou de pagar
    Baseado no percentual pago e no plano de financiamento
    """
    try:
        pago_perc = row['pago_perc']
        plano = int(row['plano_financiamento'])
        data_inicio = row['Data_inicio']
        
        # Calcular quantas parcelas foram pagas (aproximadamente)
        parcelas_pagas = int(pago_perc * plano)
        
        # Se pagou 0 parcelas, parou no mês 0 (não pagou nem a primeira)
        if parcelas_pagas == 0:
            mes_parada = 0
        else:
            # Parou após pagar X parcelas
            mes_parada = parcelas_pagas
        
        # Calcular data aproximada da parada (mês após a última parcela paga)
        if pd.notna(data_inicio):
            data_parada = data_inicio + timedelta(days=30 * mes_parada)
        else:
            data_parada = None
            
        return {
            'parcelas_pagas_estimadas': parcelas_pagas,
            'mes_parada': mes_parada,
            'data_parada_estimada': data_parada,
            'perc_pago_categorizado': categorizar_pagamento(pago_perc)
        }
    except:
        return {
            'parcelas_pagas_estimadas': 0,
            'mes_parada': 0,
            'data_parada_estimada': None,
            'perc_pago_categorizado': 'Desconhecido'
        }

def categorizar_pagamento(pago_perc):
    """Categoriza o percentual de pagamento"""
    if pago_perc <= 0:
        return 'Não pagou'
    elif pago_perc < 0.25:
        return 'Pagou pouco (0-25%)'
    elif pago_perc < 0.5:
        return 'Pagou menos da metade (25-50%)'
    elif pago_perc < 0.75:
        return 'Pagou mais da metade (50-75%)'
    elif pago_perc < 1.0:
        return 'Pagou quase tudo (75-99%)'
    else:
        return 'Adimplente (100%)'

# Aplicar cálculo
info_parada = df_parcial.apply(estimar_mes_parada, axis=1, result_type='expand')
df_parcial = pd.concat([df_parcial, info_parada], axis=1)

print(f"   ✅ Cálculo concluído!")

# ========== ESTATÍSTICAS GERAIS ==========
print("\n" + "="*80)
print("📊 ESTATÍSTICAS GERAIS - INADIMPLENTES PARCIAIS")
print("="*80)

print("\n1️⃣ Distribuição por Categoria de Pagamento:")
cat_dist = df_parcial['perc_pago_categorizado'].value_counts().sort_index()
for cat, count in cat_dist.items():
    pct = count / len(df_parcial) * 100
    print(f"   {cat}: {count:,} ({pct:.1f}%)")

print("\n2️⃣ Distribuição por Mês de Parada:")
mes_dist = df_parcial['mes_parada'].value_counts().sort_index().head(12)
for mes, count in mes_dist.items():
    pct = count / len(df_parcial) * 100
    print(f"   Parou no mês {mes}: {count:,} ({pct:.1f}%)")

print(f"\n3️⃣ Média de Parcelas Pagas: {df_parcial['parcelas_pagas_estimadas'].mean():.1f}")
print(f"   Mediana de Parcelas Pagas: {df_parcial['parcelas_pagas_estimadas'].median():.0f}")

# ========== ANÁLISE TEMPORAL: CLUSTERS DE INADIMPLÊNCIA ==========
print("\n" + "="*80)
print("📈 ANÁLISE TEMPORAL: CLUSTERS DE INADIMPLÊNCIA")
print("="*80)

# Agregar por mês/ano da parada
df_parcial['ano_mes_parada'] = df_parcial['data_parada_estimada'].dt.to_period('M')
df_parcial['mes_ano_parada_str'] = df_parcial['data_parada_estimada'].dt.strftime('%Y-%m')

# Contar inadimplências por mês
inadimplencias_por_mes = df_parcial.groupby('mes_ano_parada_str').size().sort_index()

print("\n🔥 Meses com MAIOR concentração de paradas de pagamento:")
top_meses = inadimplencias_por_mes.nlargest(10)
for mes, count in top_meses.items():
    print(f"   {mes}: {count:,} clientes pararam de pagar")

# ========== ANÁLISE REGIONAL COM PERCENTUAIS RELATIVOS ==========
print("\n" + "="*80)
print("🗺️  ANÁLISE REGIONAL: TAXA DE INADIMPLÊNCIA PARCIAL POR REGIÃO")
print("="*80)
print("(Comparando com o total de contratos de cada região - não apenas números absolutos)\n")

# Calcular totais por UF no dataset completo
total_contratos_por_uf = df['uf'].value_counts()

# Calcular inadimplentes parciais por UF
inadim_parcial_por_uf = df_parcial['uf'].value_counts()

# Calcular taxa de inadimplência parcial por UF
taxa_inadim_uf = pd.DataFrame({
    'total_contratos': total_contratos_por_uf,
    'inadim_parciais': inadim_parcial_por_uf,
    'media_pago_perc': df_parcial.groupby('uf')['pago_perc'].mean()
})
taxa_inadim_uf['taxa_inadim_parcial'] = (taxa_inadim_uf['inadim_parciais'] / taxa_inadim_uf['total_contratos'] * 100)
taxa_inadim_uf = taxa_inadim_uf.sort_values('taxa_inadim_parcial', ascending=False)

print("1️⃣ Estados com MAIOR taxa de inadimplência parcial (% do total de contratos):")
for uf, row in taxa_inadim_uf.head(10).iterrows():
    print(f"   {uf}:")
    print(f"      📊 Taxa de inadim. parcial: {row['taxa_inadim_parcial']:.2f}%")
    print(f"      👥 Total contratos: {row['total_contratos']:,.0f} | Inadim parciais: {row['inadim_parciais']:,.0f}")
    print(f"      💰 Média paga pelos inadim: {row['media_pago_perc']*100:.1f}%")

# Calcular totais por cidade no dataset completo
total_contratos_por_cidade = df['municipio'].value_counts()

# Calcular inadimplentes parciais por cidade
inadim_parcial_por_cidade = df_parcial['municipio'].value_counts()

# Calcular taxa de inadimplência parcial por cidade
taxa_inadim_cidade = pd.DataFrame({
    'total_contratos': total_contratos_por_cidade,
    'inadim_parciais': inadim_parcial_por_cidade,
    'media_pago_perc': df_parcial.groupby('municipio')['pago_perc'].mean(),
    'uf': df_parcial.groupby('municipio')['uf'].first()
})
taxa_inadim_cidade['taxa_inadim_parcial'] = (taxa_inadim_cidade['inadim_parciais'] / taxa_inadim_cidade['total_contratos'] * 100)

# Filtrar cidades com pelo menos 50 contratos para ter relevância estatística
taxa_inadim_cidade_relevante = taxa_inadim_cidade[taxa_inadim_cidade['total_contratos'] >= 50].sort_values('taxa_inadim_parcial', ascending=False)

print("\n2️⃣ Cidades com MAIOR taxa de inadimplência parcial (min. 50 contratos):")
for cidade, row in taxa_inadim_cidade_relevante.head(10).iterrows():
    print(f"   {cidade}/{row['uf']}:")
    print(f"      📊 Taxa de inadim. parcial: {row['taxa_inadim_parcial']:.2f}%")
    print(f"      👥 Total contratos: {row['total_contratos']:,.0f} | Inadim parciais: {row['inadim_parciais']:,.0f}")
    print(f"      💰 Média paga pelos inadim: {row['media_pago_perc']*100:.1f}%")

# ========== ANÁLISE DE CLUSTERS REGIONAIS TEMPORAIS (COM TAXA RELATIVA) ==========
print("\n" + "="*80)
print("🎯 CLUSTERS REGIONAIS E TEMPORAIS (Normalizado por Volume de Negócios)")
print("="*80)
print("Identificando quando ALTA PROPORÇÃO de clientes da região pararam ao mesmo tempo\n")

# Calcular total de contratos por UF e mês no dataset completo
df['ano_mes_inicio'] = df['Data_inicio'].dt.strftime('%Y-%m')
total_por_uf_mes = df.groupby(['uf', 'ano_mes_inicio']).size()

# Agrupar inadimplentes parciais por UF + Mês de parada
clusters_regionais = df_parcial.groupby(['uf', 'mes_ano_parada_str']).agg({
    'contrato_id': 'count',
    'pago_perc': 'mean',
    'lucro': 'sum',
    'municipio': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'
}).rename(columns={
    'contrato_id': 'num_inadimplentes',
    'pago_perc': 'media_pago_perc',
    'lucro': 'lucro_total',
    'municipio': 'cidade_principal'
})

# Calcular taxa relativa corretamente:
# (num inadimplentes que pararam neste mês e região / total de contratos da região) * 100
clusters_regionais['taxa_inadim_relativa'] = 0.0
for (uf, mes), row in clusters_regionais.iterrows():
    total_uf = total_contratos_por_uf.get(uf, 1)
    # Taxa = quantos % do total de contratos da região pararam de pagar neste mês
    clusters_regionais.loc[(uf, mes), 'taxa_inadim_relativa'] = (row['num_inadimplentes'] / total_uf * 100) if total_uf > 0 else 0

# Filtrar clusters significativos (pelo menos 5 inadimplentes E taxa > 1%)
clusters_significativos = clusters_regionais[
    (clusters_regionais['num_inadimplentes'] >= 5) & 
    (clusters_regionais['taxa_inadim_relativa'] > 0)
].sort_values('taxa_inadim_relativa', ascending=False)

print(f"🚨 Encontrados {len(clusters_significativos)} clusters significativos")
print("\nTop 20 Clusters com MAIOR taxa relativa de inadimplência:")
print("-" * 100)
for idx, ((uf, mes), row) in enumerate(clusters_significativos.head(20).iterrows(), 1):
    print(f"{idx:2d}. {uf} - {mes}:")
    print(f"    📊 Taxa: {row['taxa_inadim_relativa']:.2f}% do total de contratos do estado pararam neste mês")
    print(f"    👥 {row['num_inadimplentes']} inadimplentes parciais")
    print(f"    💰 Média paga: {row['media_pago_perc']*100:.1f}%")
    print(f"    📍 Cidade principal: {row['cidade_principal']}")
    print(f"    💸 Lucro total: R$ {row['lucro_total']:,.2f}")
    print()

# ========== ANÁLISE POR CIDADE + MÊS (COM TAXA RELATIVA) ==========
print("\n" + "="*80)
print("🏙️  CLUSTERS POR CIDADE (Normalizado por Volume)")
print("="*80)

# Calcular total de contratos por cidade e mês
total_por_cidade_mes = df.groupby(['municipio', 'ano_mes_inicio']).size()

clusters_cidade = df_parcial.groupby(['municipio', 'mes_ano_parada_str']).agg({
    'contrato_id': 'count',
    'pago_perc': 'mean',
    'lucro': 'sum',
    'uf': 'first'
}).rename(columns={
    'contrato_id': 'num_inadimplentes',
    'pago_perc': 'media_pago_perc',
    'lucro': 'lucro_total'
})

# Calcular taxa relativa por cidade corretamente:
# (num inadimplentes que pararam neste mês na cidade / total de contratos da cidade) * 100
clusters_cidade['taxa_inadim_relativa'] = 0.0
for (cidade, mes), row in clusters_cidade.iterrows():
    total_cidade = total_contratos_por_cidade.get(cidade, 1)
    # Taxa = quantos % do total de contratos da cidade pararam de pagar neste mês
    clusters_cidade.loc[(cidade, mes), 'taxa_inadim_relativa'] = (row['num_inadimplentes'] / total_cidade * 100) if total_cidade > 0 else 0

clusters_cidade_sig = clusters_cidade[
    (clusters_cidade['num_inadimplentes'] >= 3) &
    (clusters_cidade['taxa_inadim_relativa'] > 0)
].sort_values('taxa_inadim_relativa', ascending=False)

print(f"\n🚨 Encontrados {len(clusters_cidade_sig)} clusters de cidades")
print("\nTop 15 Clusters por cidade (maior taxa relativa):")
print("-" * 100)
for idx, ((cidade, mes), row) in enumerate(clusters_cidade_sig.head(15).iterrows(), 1):
    print(f"{idx:2d}. {cidade}/{row['uf']} - {mes}:")
    print(f"    📊 Taxa: {row['taxa_inadim_relativa']:.2f}% do total de contratos da cidade pararam neste mês")
    print(f"    👥 {row['num_inadimplentes']} inadimplentes parciais")
    print(f"    💰 Média paga: {row['media_pago_perc']*100:.1f}%")
    print(f"    💸 Lucro total: R$ {row['lucro_total']:,.2f}")
    print()

# ========== VISUALIZAÇÕES ==========
print("\n📊 Gerando visualizações...")

# Criar pasta para salvar os gráficos
import os
output_dir = '../../graficos/analises_dataset/ANALISE_EXPLORATORIA_IN_PARCIAL'
os.makedirs(output_dir, exist_ok=True)

fig = plt.figure(figsize=(20, 12))

# 1. Distribuição de pagamento percentual
ax1 = plt.subplot(3, 3, 1)
df_parcial['pago_perc'].hist(bins=50, edgecolor='black', alpha=0.7)
plt.axvline(df_parcial['pago_perc'].mean(), color='red', linestyle='--', label=f'Média: {df_parcial["pago_perc"].mean()*100:.1f}%')
plt.axvline(df_parcial['pago_perc'].median(), color='green', linestyle='--', label=f'Mediana: {df_parcial["pago_perc"].median()*100:.1f}%')
plt.xlabel('Percentual Pago')
plt.ylabel('Frequência')
plt.title('Distribuição do Percentual Pago\n(Inadimplentes Parciais)', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Distribuição por categoria de pagamento
ax2 = plt.subplot(3, 3, 2)
cat_counts = df_parcial['perc_pago_categorizado'].value_counts()
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(cat_counts)))
cat_counts.plot(kind='barh', color=colors)
plt.xlabel('Número de Contratos')
plt.title('Distribuição por Categoria de Pagamento', fontweight='bold')
plt.tight_layout()

# 3. Distribuição por mês de parada
ax3 = plt.subplot(3, 3, 3)
mes_counts = df_parcial['mes_parada'].value_counts().sort_index().head(12)
mes_counts.plot(kind='bar', color='coral', edgecolor='black')
plt.xlabel('Mês em que Parou de Pagar')
plt.ylabel('Número de Contratos')
plt.title('Distribuição por Mês de Parada', fontweight='bold')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3, axis='y')

# 4. Linha temporal de inadimplências
ax4 = plt.subplot(3, 3, 4)
inadimplencias_por_mes.plot(kind='line', marker='o', linewidth=2, markersize=6, color='darkred')
plt.xlabel('Mês/Ano da Parada')
plt.ylabel('Número de Inadimplentes')
plt.title('Evolução Temporal das Paradas de Pagamento', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# 5. Top 10 estados (TAXA RELATIVA)
ax5 = plt.subplot(3, 3, 5)
top_taxa_uf = taxa_inadim_uf.head(10)['taxa_inadim_parcial'].sort_values()
colors_uf = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_taxa_uf)))
top_taxa_uf.plot(kind='barh', color=colors_uf, edgecolor='black')
plt.xlabel('Taxa de Inadimplência Parcial (%)')
plt.title('Top 10 Estados com MAIOR Taxa de Inadim. Parcial\n(% do total de contratos)', fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

# 6. Top 10 cidades (TAXA RELATIVA)
ax6 = plt.subplot(3, 3, 6)
top_taxa_cidade = taxa_inadim_cidade_relevante.head(10)['taxa_inadim_parcial'].sort_values()
colors_cidade = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_taxa_cidade)))
top_taxa_cidade.plot(kind='barh', color=colors_cidade, edgecolor='black')
plt.xlabel('Taxa de Inadimplência Parcial (%)')
plt.title('Top 10 Cidades com MAIOR Taxa de Inadim. Parcial\n(min. 50 contratos)', fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

# 7. Heatmap: UF x Categoria de Pagamento
ax7 = plt.subplot(3, 3, 7)
pivot_uf_cat = pd.crosstab(df_parcial['uf'], df_parcial['perc_pago_categorizado'])
# Pegar top 10 UFs
top_10_ufs = df_parcial['uf'].value_counts().head(10).index
pivot_uf_cat_top = pivot_uf_cat.loc[top_10_ufs]
sns.heatmap(pivot_uf_cat_top, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Número de Contratos'})
plt.title('Heatmap: Estado x Categoria de Pagamento', fontweight='bold')
plt.xlabel('Categoria de Pagamento')
plt.ylabel('Estado')

# 8. Boxplot: Percentual pago por estado (top 10)
ax8 = plt.subplot(3, 3, 8)
df_top_ufs = df_parcial[df_parcial['uf'].isin(top_10_ufs)]
df_top_ufs.boxplot(column='pago_perc', by='uf', ax=ax8)
plt.title('Distribuição do Percentual Pago por Estado\n(Top 10 Estados)', fontweight='bold')
plt.suptitle('')  # Remove o título automático
plt.xlabel('Estado')
plt.ylabel('Percentual Pago')
plt.xticks(rotation=45)

# 9. Scatter: Parcelas pagas vs Lucro
ax9 = plt.subplot(3, 3, 9)
sample = df_parcial.sample(min(1000, len(df_parcial)))  # Amostra para não sobrecarregar
scatter = ax9.scatter(sample['parcelas_pagas_estimadas'], sample['lucro'], 
                     c=sample['pago_perc'], cmap='RdYlGn', alpha=0.6, s=50)
plt.colorbar(scatter, label='Percentual Pago')
plt.xlabel('Parcelas Pagas (Estimadas)')
plt.ylabel('Lucro (R$)')
plt.title('Relação: Parcelas Pagas vs Lucro', fontweight='bold')
plt.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/analise_inadimplente_parcial.png', 
            dpi=300, bbox_inches='tight')
print(f"   ✅ Salvo: graficos/ANALISE_EXPLORATORIA_IN_PARCIAL/analise_inadimplente_parcial.png")

# ========== EVOLUÇÃO TEMPORAL POR ESTADO ==========
print("\n📊 Gerando evolução temporal por estado...")

# Pegar os top estados por volume de inadimplentes parciais
top_estados = df_parcial['uf'].value_counts().head(12).index.tolist()

# Criar figura com subplots para cada estado
fig_temporal = plt.figure(figsize=(20, 15))

for idx, uf in enumerate(top_estados, 1):
    ax = plt.subplot(4, 3, idx)
    
    # Filtrar dados do estado
    df_uf = df_parcial[df_parcial['uf'] == uf]
    
    # Contar inadimplências por mês
    inadim_por_mes_uf = df_uf.groupby('mes_ano_parada_str').size().sort_index()
    
    # Plotar
    inadim_por_mes_uf.plot(kind='line', marker='o', linewidth=2.5, markersize=7, 
                            color='darkred', ax=ax)
    
    # Calcular taxa relativa (se possível)
    total_uf = total_contratos_por_uf.get(uf, 1)
    taxa_media = (len(df_uf) / total_uf * 100) if total_uf > 0 else 0
    
    ax.set_xlabel('Mês/Ano da Parada', fontsize=9)
    ax.set_ylabel('Número de Inadimplentes', fontsize=9)
    ax.set_title(f'{uf} - {len(df_uf):,} inadim. parciais ({taxa_media:.1f}% do total)', 
                 fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Destacar picos
    if len(inadim_por_mes_uf) > 0:
        max_mes = inadim_por_mes_uf.idxmax()
        max_val = inadim_por_mes_uf.max()
        ax.axhline(max_val, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax.text(0.02, 0.98, f'Pico: {max_mes}\n({max_val} inadim.)', 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Evolução Temporal das Paradas de Pagamento por Estado\n(Top 12 Estados)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(f'{output_dir}/evolucao_temporal_por_estado.png', 
            dpi=300, bbox_inches='tight')
print(f"   ✅ Salvo: graficos/ANALISE_EXPLORATORIA_IN_PARCIAL/evolucao_temporal_por_estado.png")

# ========== HEATMAP TEMPORAL POR ESTADO ==========
print("\n📊 Gerando heatmap temporal por estado...")

# Criar matriz: Estados x Meses
pivot_temporal = df_parcial.groupby(['uf', 'mes_ano_parada_str']).size().unstack(fill_value=0)

# Pegar top 15 estados
top_15_estados = df_parcial['uf'].value_counts().head(15).index
pivot_temporal_top = pivot_temporal.loc[top_15_estados]

# Ordenar colunas (meses) cronologicamente
pivot_temporal_top = pivot_temporal_top[sorted(pivot_temporal_top.columns)]

# Limitar aos últimos 24 meses para visualização
if len(pivot_temporal_top.columns) > 24:
    pivot_temporal_top = pivot_temporal_top[pivot_temporal_top.columns[-24:]]

fig_heatmap = plt.figure(figsize=(20, 10))

# Heatmap com escala logarítmica para evidenciar células discretas porém excepcionais
from matplotlib.colors import LogNorm
ax1 = plt.subplot(1, 1, 1)

# Somar +1 para evitar zeros (LogNorm não aceita 0). Mantemos os valores originais em export se necessário.
heatmap_data = pivot_temporal_top.fillna(0).astype(int)
heatmap_display = heatmap_data + 1

vmin = heatmap_display.values.min() if heatmap_display.values.size > 0 else 1
vmax = heatmap_display.values.max() if heatmap_display.values.size > 0 else 1

sns.heatmap(heatmap_display, cmap='YlOrRd', annot=False, fmt='d', 
            norm=LogNorm(vmin=max(1, vmin), vmax=max(1, vmax)),
            cbar_kws={'label': 'Número de Inadimplentes Parciais (escala log +1)'},
            linewidths=0.5, linecolor='gray')

plt.title('Heatmap Temporal (Log) : Inadimplentes Parciais por Estado e Mês\n(Top 15 Estados - Últimos 24 Meses)', 
          fontweight='bold', fontsize=14)
plt.xlabel('Mês/Ano da Parada', fontsize=12)
plt.ylabel('Estado (UF)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=11)

plt.tight_layout()
plt.savefig(f'{output_dir}/heatmap_temporal_estados_log.png', 
            dpi=300, bbox_inches='tight')
print(f"   ✅ Salvo: graficos/ANALISE_EXPLORATORIA_IN_PARCIAL/heatmap_temporal_estados_log.png")

# ========== HEATMAPS INDIVIDUAIS POR ESTADO (Cidades x Meses) ==========
print("\n📊 Gerando heatmaps individuais por estado (Cidades x Meses)...")

# Criar subpasta para heatmaps por estado
heatmaps_dir = f'{output_dir}/heatmap_por_uf'
os.makedirs(heatmaps_dir, exist_ok=True)

from matplotlib.colors import LogNorm

# Pegar estados com pelo menos 20 inadimplentes parciais para ter dados significativos
estados_relevantes = df_parcial['uf'].value_counts()
estados_relevantes = estados_relevantes[estados_relevantes >= 20].index.tolist()

# Definir intervalo COMPLETO de meses do dataset (para padronizar todos os heatmaps)
todos_meses = sorted(df_parcial['mes_ano_parada_str'].dropna().unique())
print(f"   Intervalo de meses no dataset: {todos_meses[0]} a {todos_meses[-1]}")
print(f"   Gerando heatmaps para {len(estados_relevantes)} estados...")

for uf in estados_relevantes:
    # Filtrar dados do estado
    df_uf = df_parcial[df_parcial['uf'] == uf]
    
    # Criar matriz: Cidades x Meses
    pivot_cidade_mes = df_uf.groupby(['municipio', 'mes_ano_parada_str']).size().unstack(fill_value=0)
    
    # USAR TODAS AS CIDADES DO ESTADO (não limitar a top N)
    pivot_cidade_mes_top = pivot_cidade_mes.copy()
    
    # PADRONIZAR: adicionar colunas de meses faltantes com valor 0
    # Assim todos os heatmaps mostram o mesmo intervalo temporal
    for mes in todos_meses:
        if mes not in pivot_cidade_mes_top.columns:
            pivot_cidade_mes_top[mes] = 0
    
    # Ordenar colunas (meses) cronologicamente - TODOS os meses agora
    pivot_cidade_mes_top = pivot_cidade_mes_top[sorted(pivot_cidade_mes_top.columns)]
    
    # Ordenar linhas (cidades) por total de inadimplentes (maior primeiro)
    totais = pivot_cidade_mes_top.sum(axis=1).sort_values(ascending=False)
    pivot_cidade_mes_top = pivot_cidade_mes_top.loc[totais.index]
    
    # NÃO LIMITAR - mostrar TODOS os meses do dataset
    # Cada heatmap terá a largura necessária para mostrar todo o intervalo temporal
    
    # Skip se não tiver dados suficientes
    if pivot_cidade_mes_top.empty or pivot_cidade_mes_top.sum().sum() < 5:
        continue
    
    # Criar figura - ajustar largura baseado no número de meses
    num_meses = len(pivot_cidade_mes_top.columns)
    largura = max(20, num_meses * 0.5)  # Mínimo 20, ou 0.5 por mês
    altura = max(10, len(pivot_cidade_mes_top) * 0.5)  # 0.5 por cidade
    fig_uf = plt.figure(figsize=(largura, altura))
    
    # Preparar dados para log scale
    heatmap_data_uf = pivot_cidade_mes_top.fillna(0).astype(int)
    heatmap_display_uf = heatmap_data_uf + 1
    
    vmin_uf = heatmap_display_uf.values.min() if heatmap_display_uf.values.size > 0 else 1
    vmax_uf = heatmap_display_uf.values.max() if heatmap_display_uf.values.size > 0 else 1
    
    # Plotar heatmap
    if vmax_uf > vmin_uf and vmax_uf > 1:
        # Usar escala logarítmica se houver variação
        sns.heatmap(heatmap_display_uf, cmap='YlOrRd', annot=True, fmt='d', 
                    norm=LogNorm(vmin=max(1, vmin_uf), vmax=max(1, vmax_uf)),
                    cbar_kws={'label': 'Inadimplentes (log +1)'},
                    linewidths=0.5, linecolor='gray', 
                    annot_kws={'fontsize': 8})
    else:
        # Escala linear se valores forem muito similares
        sns.heatmap(heatmap_data_uf, cmap='YlOrRd', annot=True, fmt='d',
                    cbar_kws={'label': 'Inadimplentes'},
                    linewidths=0.5, linecolor='gray',
                    annot_kws={'fontsize': 8})
    
    # Calcular estatísticas do estado
    total_inadim_uf = len(df_uf)
    total_contratos_uf = total_contratos_por_uf.get(uf, 1)
    taxa_uf = (total_inadim_uf / total_contratos_uf * 100) if total_contratos_uf > 0 else 0
    
    plt.title(f'Heatmap Temporal: {uf} - Inadimplentes Parciais por Cidade\n' + 
              f'{total_inadim_uf:,} inadim. parciais ({taxa_uf:.1f}% do total de contratos) | ' +
              f'Total: {len(pivot_cidade_mes_top)} cidades',
              fontweight='bold', fontsize=13)
    plt.xlabel('Mês/Ano da Parada', fontsize=11)
    plt.ylabel('Cidade', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{heatmaps_dir}/heatmap_{uf}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"      ✅ {uf}: {len(pivot_cidade_mes_top)} cidades, {len(pivot_cidade_mes_top.columns)} meses")

print(f"   ✅ {len(estados_relevantes)} heatmaps salvos em: graficos/ANALISE_EXPLORATORIA_IN_PARCIAL/heatmap_por_uf/")

# ========== VISUALIZAÇÃO DOS CLUSTERS ==========
print("\n📊 Gerando visualização dos clusters críticos...")

fig2 = plt.figure(figsize=(18, 10))

# 1. Top 15 clusters regionais (UF + Mês) - TAXA RELATIVA
ax1 = plt.subplot(2, 2, 1)
top_clusters = clusters_significativos.head(15).copy()
top_clusters['label'] = [f"{uf}\n{mes[-5:]}" for uf, mes in top_clusters.index]  # Mostrar só MM-AA
colors_clusters = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(top_clusters)))
ax1.bar(range(len(top_clusters)), top_clusters['taxa_inadim_relativa'], color=colors_clusters, edgecolor='black')
ax1.set_xticks(range(len(top_clusters)))
ax1.set_xticklabels(top_clusters['label'], rotation=45, ha='right')
plt.xlabel('Estado - Mês')
plt.ylabel('Taxa de Inadimplência (%)')
plt.title('Top 15 Clusters: MAIOR Taxa de Inadim. Relativa\n(Estado + Mês)', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 2. Top 15 clusters por cidade - TAXA RELATIVA
ax2 = plt.subplot(2, 2, 2)
top_clusters_cidade = clusters_cidade_sig.head(15).copy()
top_clusters_cidade['label'] = [f"{cidade[:15]}\n{mes[-5:]}" for cidade, mes in top_clusters_cidade.index]
colors_cidade_cluster = plt.cm.OrRd(np.linspace(0.3, 0.9, len(top_clusters_cidade)))
ax2.bar(range(len(top_clusters_cidade)), top_clusters_cidade['taxa_inadim_relativa'], 
        color=colors_cidade_cluster, edgecolor='black')
ax2.set_xticks(range(len(top_clusters_cidade)))
ax2.set_xticklabels(top_clusters_cidade['label'], rotation=45, ha='right', fontsize=8)
plt.xlabel('Cidade - Mês')
plt.ylabel('Taxa de Inadimplência (%)')
plt.title('Top 15 Clusters Cidade: MAIOR Taxa Relativa', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 3. Média de pagamento nos clusters
ax3 = plt.subplot(2, 2, 3)
top_clusters_sorted = clusters_significativos.head(15).sort_values('media_pago_perc')
top_clusters_sorted['label'] = [f"{uf}-{mes}" for uf, mes in top_clusters_sorted.index]
colors_perc = plt.cm.RdYlGn(top_clusters_sorted['media_pago_perc'])
top_clusters_sorted.plot(x='label', y='media_pago_perc', kind='barh', ax=ax3, color=colors_perc, legend=False)
plt.xlabel('Percentual Médio Pago')
plt.title('Média de Pagamento nos Top 15 Clusters', fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

# 4. Impacto financeiro dos clusters
ax4 = plt.subplot(2, 2, 4)
top_clusters_lucro = clusters_significativos.head(15).sort_values('lucro_total')
top_clusters_lucro['label'] = [f"{uf}-{mes}" for uf, mes in top_clusters_lucro.index]
colors_lucro = ['red' if x < 0 else 'green' for x in top_clusters_lucro['lucro_total']]
top_clusters_lucro.plot(x='label', y='lucro_total', kind='barh', ax=ax4, color=colors_lucro, legend=False)
plt.xlabel('Lucro Total (R$)')
plt.title('Impacto Financeiro dos Top 15 Clusters', fontweight='bold')
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{output_dir}/clusters_inadimplencia_parcial.png', 
            dpi=300, bbox_inches='tight')
print(f"   ✅ Salvo: graficos/ANALISE_EXPLORATORIA_IN_PARCIAL/clusters_inadimplencia_parcial.png")

# ========== SALVAR DADOS DOS CLUSTERS ==========
print("\n💾 Salvando dados dos clusters...")

# Salvar clusters regionais
clusters_significativos.to_csv(
    '/home/usuario/Downloads/Documentos/top_one_model_v2/data/clusters_regionais_inadimplencia.csv',
    sep=';', encoding='utf-8'
)
print(f"   ✅ Salvo: data/clusters_regionais_inadimplencia.csv")

# Salvar clusters de cidade
clusters_cidade_sig.to_csv(
    '/home/usuario/Downloads/Documentos/top_one_model_v2/data/clusters_cidade_inadimplencia.csv',
    sep=';', encoding='utf-8'
)
print(f"   ✅ Salvo: data/clusters_cidade_inadimplencia.csv")

# Salvar dataset enriquecido
df_parcial.to_csv(
    '/home/usuario/Downloads/Documentos/top_one_model_v2/data/inadimplentes_parciais_enriquecido.csv',
    sep=';', encoding='utf-8', index=False
)
print(f"   ✅ Salvo: data/inadimplentes_parciais_enriquecido.csv")

# ========== RESUMO FINAL ==========
print("\n" + "="*80)
print("✅ ANÁLISE CONCLUÍDA")
print("="*80)
print(f"\n📊 Resumo:")
print(f"   • Total de inadimplentes parciais analisados: {len(df_parcial):,}")
print(f"   • Clusters regionais significativos encontrados: {len(clusters_significativos)}")
print(f"   • Clusters de cidade significativos encontrados: {len(clusters_cidade_sig)}")
print(f"   • Média de parcelas pagas: {df_parcial['parcelas_pagas_estimadas'].mean():.1f}")
print(f"   • Percentual médio pago: {df_parcial['pago_perc'].mean()*100:.1f}%")

print("\n📁 Arquivos gerados:")
print("   • graficos/ANALISE_EXPLORATORIA_IN_PARCIAL/analise_inadimplente_parcial.png")
print("   • graficos/ANALISE_EXPLORATORIA_IN_PARCIAL/evolucao_temporal_por_estado.png")
print("   • graficos/ANALISE_EXPLORATORIA_IN_PARCIAL/heatmap_temporal_estados_log.png (escala logarítmica)")
print("   • graficos/ANALISE_EXPLORATORIA_IN_PARCIAL/clusters_inadimplencia_parcial.png")
print(f"   • graficos/ANALISE_EXPLORATORIA_IN_PARCIAL/heatmap_por_uf/ ({len(estados_relevantes)} heatmaps individuais)")
print("   • data/clusters_regionais_inadimplencia.csv")
print("   • data/clusters_cidade_inadimplencia.csv")
print("   • data/inadimplentes_parciais_enriquecido.csv")

print("\n" + "="*80)
print("🎯 Próximos passos sugeridos:")
print("   1. Investigar os clusters mais críticos identificados")
print("   2. Analisar causas comuns nos meses com picos de inadimplência")
print("   3. Verificar eventos externos (econômicos, regionais) nos períodos críticos")
print("   4. Criar estratégias de cobrança específicas para cada cluster")
print("="*80)
