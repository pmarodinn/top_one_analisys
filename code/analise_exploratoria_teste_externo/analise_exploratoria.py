import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import geopandas as gpd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Caminhos
FILE_PATH = os.path.abspath(__file__)
FILE_FOLDER = os.path.dirname(FILE_PATH)
ROOT_FOLDER = os.path.dirname(os.path.dirname(FILE_FOLDER))

DATA_PATH = os.path.join(ROOT_FOLDER, "data", "teste_externo", "datasets_tratados", "dataset_teste_externo_final.csv")
SHAPEFILE_PATH = os.path.join(ROOT_FOLDER, "data", "geolocalizacao_vetorial", "BR_Municipios_2024.shp")
OUTPUT_DIR = os.path.join(ROOT_FOLDER, "graficos", "analise_teste_externo")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Carrega o dataset de teste externo"""
    print("Carregando dados...")
    if not os.path.exists(DATA_PATH):
        print(f"ERRO: Arquivo não encontrado em {DATA_PATH}")
        return None
        
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',')
    print(f"✓ Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # Converter coluna de data
    if 'Data_inicio' in df.columns:
        df['Data_inicio'] = pd.to_datetime(df['Data_inicio'], errors='coerce')
    
    return df


def basic_info(df):
    """Exibe informações básicas do dataset"""
    print("\n" + "="*80)
    print("INFORMAÇÕES BÁSICAS DO DATASET")
    print("="*80)
    print(f"\nDimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"\nColunas: {list(df.columns)}")
    
    # Missing values
    missing = df.isnull().sum()
    missing_perc = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Valores Ausentes': missing,
        'Percentual (%)': missing_perc
    })
    missing_df = missing_df[missing_df['Valores Ausentes'] > 0].sort_values(
        'Valores Ausentes', ascending=False
    )
    
    if not missing_df.empty:
        print("\n--- Valores Ausentes ---")
        print(missing_df)
        
        # Gráfico de valores ausentes
        if len(missing_df) > 0:
            fig, ax = plt.subplots(figsize=(12, max(6, len(missing_df) * 0.4)))
            missing_df['Valores Ausentes'].plot(kind='barh', ax=ax, color='steelblue')
            ax.set_title("Valores Ausentes por Coluna", fontsize=14, fontweight='bold')
            ax.set_xlabel("Quantidade de Valores Ausentes")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "01_missing_values.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Gráfico salvo: 01_missing_values.png")


def plot_contratos_por_estado_heatmap(df):
    """Cria heatmap de contratos por estado usando shapefile"""
    print("\n" + "="*80)
    print("ANÁLISE GEOGRÁFICA - HEATMAP DE CONTRATOS POR ESTADO")
    print("="*80)
    
    if not os.path.exists(SHAPEFILE_PATH):
        print(f"⚠ AVISO: Shapefile não encontrado em {SHAPEFILE_PATH}")
        return
    
    if 'uf' not in df.columns:
        print("⚠ AVISO: Coluna 'uf' não encontrada no dataset")
        return
    
    try:
        # Carregar shapefile de municípios
        print("Carregando shapefile...")
        gdf = gpd.read_file(SHAPEFILE_PATH)
        
        # Identificar coluna de UF no shapefile
        uf_col = None
        for col in ['SIGLA_UF', 'UF', 'SIGLA', 'CD_GEOCUF']:
            if col in gdf.columns:
                uf_col = col
                break
        
        if uf_col is None:
            print(f"⚠ Colunas disponíveis no shapefile: {list(gdf.columns)}")
            print("⚠ Não foi possível identificar coluna de UF")
            return
        
        print(f"✓ Usando coluna '{uf_col}' para UF")
        
        # Agrupar contratos por UF
        contratos_por_uf = df['uf'].value_counts().reset_index()
        contratos_por_uf.columns = ['UF', 'Contratos']
        
        print(f"\n--- Top 10 Estados com Mais Contratos ---")
        print(contratos_por_uf.head(10))
        
        # Dissolver geometrias por UF
        print("Processando geometrias...")
        gdf_uf = gdf.dissolve(by=uf_col, as_index=False)
        gdf_uf = gdf_uf[[uf_col, 'geometry']]
        gdf_uf.columns = ['UF', 'geometry']
        
        # Merge com dados de contratos
        gdf_merged = gdf_uf.merge(contratos_por_uf, on='UF', how='left')
        gdf_merged['Contratos'] = gdf_merged['Contratos'].fillna(0)
        
        # Criar heatmap
        fig, ax = plt.subplots(figsize=(16, 12))
        gdf_merged.plot(
            column='Contratos',
            cmap='YlOrRd',
            legend=True,
            ax=ax,
            edgecolor='black',
            linewidth=0.5,
            legend_kwds={
                'label': "Número de Contratos",
                'orientation': "horizontal",
                'shrink': 0.7,
                'pad': 0.05
            }
        )
        
        # Adicionar labels dos estados
        for idx, row in gdf_merged.iterrows():
            if row['Contratos'] > 0:
                centroid = row['geometry'].centroid
                ax.annotate(
                    text=f"{row['UF']}\n{int(row['Contratos'])}",
                    xy=(centroid.x, centroid.y),
                    ha='center',
                    fontsize=8,
                    fontweight='bold'
                )
        
        ax.set_title("Heatmap de Contratos por Estado (UF)", fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "02_heatmap_contratos_por_estado.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Heatmap salvo: 02_heatmap_contratos_por_estado.png")
        
    except Exception as e:
        print(f"✗ Erro ao gerar heatmap: {str(e)}")


def plot_contratos_por_mes(df):
    """Gráfico de número de contratos por mês inicial"""
    print("\n" + "="*80)
    print("ANÁLISE TEMPORAL - CONTRATOS POR MÊS")
    print("="*80)
    
    if 'Data_inicio' not in df.columns:
        print("⚠ AVISO: Coluna 'Data_inicio' não encontrada")
        return
    
    # Extrair ano e mês
    df['Ano_Mes'] = df['Data_inicio'].dt.to_period('M')
    contratos_por_mes = df['Ano_Mes'].value_counts().sort_index()
    
    print(f"\n--- Distribuição de Contratos por Mês ---")
    print(f"Total de meses: {len(contratos_por_mes)}")
    print(f"Período: {contratos_por_mes.index.min()} até {contratos_por_mes.index.max()}")
    print(f"\nEstatísticas:")
    print(f"  Média mensal: {contratos_por_mes.mean():.0f} contratos")
    print(f"  Mediana: {contratos_por_mes.median():.0f} contratos")
    print(f"  Máximo: {contratos_por_mes.max():.0f} contratos ({contratos_por_mes.idxmax()})")
    print(f"  Mínimo: {contratos_por_mes.min():.0f} contratos ({contratos_por_mes.idxmin()})")
    
    # Gráfico
    fig, ax = plt.subplots(figsize=(16, 6))
    x_values = range(len(contratos_por_mes))
    ax.plot(x_values, contratos_por_mes.values, marker='o', linewidth=2, markersize=5, color='steelblue')
    ax.fill_between(x_values, contratos_por_mes.values, alpha=0.3, color='steelblue')
    
    ax.set_xlabel("Mês", fontsize=12, fontweight='bold')
    ax.set_ylabel("Número de Contratos", fontsize=12, fontweight='bold')
    ax.set_title("Número de Contratos por Mês Inicial", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Configurar xticks para mostrar todos os meses
    step = max(1, len(contratos_por_mes) // 20)  # Mostrar no máximo 20 labels
    ax.set_xticks(x_values[::step])
    ax.set_xticklabels([str(m) for m in contratos_por_mes.index[::step]], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_contratos_por_mes.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico salvo: 03_contratos_por_mes.png")


def plot_distribuicao_idade(df):
    """Gráfico da distribuição de idade por contrato"""
    print("\n" + "="*80)
    print("ANÁLISE DEMOGRÁFICA - DISTRIBUIÇÃO DE IDADE")
    print("="*80)
    
    if 'idade' not in df.columns:
        print("⚠ AVISO: Coluna 'idade' não encontrada")
        return
    
    idades = df['idade'].dropna()
    
    print(f"\n--- Estatísticas de Idade ---")
    print(f"Total de registros: {len(idades)}")
    print(f"Idade média: {idades.mean():.1f} anos")
    print(f"Mediana: {idades.median():.1f} anos")
    print(f"Desvio padrão: {idades.std():.1f} anos")
    print(f"Idade mínima: {idades.min():.0f} anos")
    print(f"Idade máxima: {idades.max():.0f} anos")
    print(f"\nQuartis:")
    print(f"  Q1 (25%): {idades.quantile(0.25):.1f} anos")
    print(f"  Q2 (50%): {idades.quantile(0.50):.1f} anos")
    print(f"  Q3 (75%): {idades.quantile(0.75):.1f} anos")
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Histograma
    axes[0, 0].hist(idades, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(idades.mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {idades.mean():.1f}')
    axes[0, 0].axvline(idades.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {idades.median():.1f}')
    axes[0, 0].set_xlabel("Idade (anos)", fontweight='bold')
    axes[0, 0].set_ylabel("Frequência", fontweight='bold')
    axes[0, 0].set_title("Histograma de Distribuição de Idade", fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Boxplot
    axes[0, 1].boxplot(idades, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', color='steelblue'),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(color='steelblue'),
                       capprops=dict(color='steelblue'))
    axes[0, 1].set_ylabel("Idade (anos)", fontweight='bold')
    axes[0, 1].set_title("Boxplot de Idade", fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Distribuição por faixas etárias
    bins = [0, 20, 30, 40, 50, 60, 70, 100]
    labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    df['faixa_etaria'] = pd.cut(df['idade'], bins=bins, labels=labels, right=False)
    faixa_counts = df['faixa_etaria'].value_counts().sort_index()
    
    axes[1, 0].bar(range(len(faixa_counts)), faixa_counts.values, color='steelblue', edgecolor='black')
    axes[1, 0].set_xticks(range(len(faixa_counts)))
    axes[1, 0].set_xticklabels(faixa_counts.index, rotation=0)
    axes[1, 0].set_xlabel("Faixa Etária", fontweight='bold')
    axes[1, 0].set_ylabel("Número de Contratos", fontweight='bold')
    axes[1, 0].set_title("Contratos por Faixa Etária", fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for i, v in enumerate(faixa_counts.values):
        axes[1, 0].text(i, v, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    # 4. Densidade (KDE)
    axes[1, 1].hist(idades, bins=50, density=True, alpha=0.5, color='steelblue', edgecolor='black', label='Histograma')
    idades.plot(kind='kde', ax=axes[1, 1], color='red', linewidth=2, label='KDE')
    axes[1, 1].set_xlabel("Idade (anos)", fontweight='bold')
    axes[1, 1].set_ylabel("Densidade", fontweight='bold')
    axes[1, 1].set_title("Densidade de Distribuição de Idade", fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_distribuicao_idade.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico salvo: 04_distribuicao_idade.png")


def plot_profissoes_comuns(df):
    """Gráfico das profissões mais comuns"""
    print("\n" + "="*80)
    print("ANÁLISE DE PROFISSÕES")
    print("="*80)
    
    if 'descricao_da_Profissao' not in df.columns:
        print("⚠ AVISO: Coluna 'descricao_da_Profissao' não encontrada")
        return
    
    profissoes = df['descricao_da_Profissao'].dropna()
    top_profissoes = profissoes.value_counts().head(20)
    
    print(f"\n--- Top 20 Profissões Mais Comuns ---")
    for i, (prof, count) in enumerate(top_profissoes.items(), 1):
        perc = count / len(df) * 100
        print(f"{i:2d}. {prof:40s} - {count:5d} contratos ({perc:5.2f}%)")
    
    # Gráfico
    fig, ax = plt.subplots(figsize=(14, 10))
    y_pos = range(len(top_profissoes))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_profissoes)))
    
    ax.barh(y_pos, top_profissoes.values, color=colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_profissoes.index)
    ax.set_xlabel("Número de Contratos", fontsize=12, fontweight='bold')
    ax.set_title("Top 20 Profissões Mais Comuns", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Adicionar valores nas barras
    for i, v in enumerate(top_profissoes.values):
        perc = v / len(df) * 100
        ax.text(v, i, f'  {v} ({perc:.1f}%)', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_profissoes_comuns.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico salvo: 05_profissoes_comuns.png")


def plot_itens_comprados(df):
    """Gráfico dos itens/mercadorias mais comprados"""
    print("\n" + "="*80)
    print("ANÁLISE DE PRODUTOS/MERCADORIAS")
    print("="*80)
    
    if 'Mercadoria' not in df.columns:
        print("⚠ AVISO: Coluna 'Mercadoria' não encontrada")
        return
    
    mercadorias = df['Mercadoria'].dropna()
    top_mercadorias = mercadorias.value_counts().head(25)
    
    print(f"\n--- Top 25 Itens/Mercadorias Mais Comprados ---")
    for i, (item, count) in enumerate(top_mercadorias.items(), 1):
        perc = count / len(df) * 100
        print(f"{i:2d}. {item:50s} - {count:5d} contratos ({perc:5.2f}%)")
    
    # Gráfico
    fig, ax = plt.subplots(figsize=(14, 12))
    y_pos = range(len(top_mercadorias))
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(top_mercadorias)))
    
    ax.barh(y_pos, top_mercadorias.values, color=colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_mercadorias.index, fontsize=9)
    ax.set_xlabel("Número de Contratos", fontsize=12, fontweight='bold')
    ax.set_title("Top 25 Itens/Mercadorias Mais Comprados", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Adicionar valores nas barras
    for i, v in enumerate(top_mercadorias.values):
        perc = v / len(df) * 100
        ax.text(v, i, f'  {v} ({perc:.1f}%)', va='center', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_itens_comprados.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico salvo: 06_itens_comprados.png")


def plot_distribuicao_valor_financiado(df):
    """Gráfico da distribuição do valor financiado"""
    print("\n" + "="*80)
    print("ANÁLISE FINANCEIRA - VALOR FINANCIADO")
    print("="*80)
    
    if 'valor_financiado' not in df.columns:
        print("⚠ AVISO: Coluna 'valor_financiado' não encontrada")
        return
    
    valores = df['valor_financiado'].dropna()
    
    print(f"\n--- Estatísticas de Valor Financiado ---")
    print(f"Total de registros: {len(valores)}")
    print(f"Valor médio: R$ {valores.mean():,.2f}")
    print(f"Mediana: R$ {valores.median():,.2f}")
    print(f"Desvio padrão: R$ {valores.std():,.2f}")
    print(f"Valor mínimo: R$ {valores.min():,.2f}")
    print(f"Valor máximo: R$ {valores.max():,.2f}")
    print(f"Valor total: R$ {valores.sum():,.2f}")
    print(f"\nQuartis:")
    print(f"  Q1 (25%): R$ {valores.quantile(0.25):,.2f}")
    print(f"  Q2 (50%): R$ {valores.quantile(0.50):,.2f}")
    print(f"  Q3 (75%): R$ {valores.quantile(0.75):,.2f}")
    print(f"  Q9 (90%): R$ {valores.quantile(0.90):,.2f}")
    print(f"  Q95 (95%): R$ {valores.quantile(0.95):,.2f}")
    print(f"  Q99 (99%): R$ {valores.quantile(0.99):,.2f}")
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Histograma completo
    axes[0, 0].hist(valores, bins=100, color='green', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(valores.mean(), color='red', linestyle='--', linewidth=2, label=f'Média: R$ {valores.mean():,.0f}')
    axes[0, 0].axvline(valores.median(), color='orange', linestyle='--', linewidth=2, label=f'Mediana: R$ {valores.median():,.0f}')
    axes[0, 0].set_xlabel("Valor Financiado (R$)", fontweight='bold')
    axes[0, 0].set_ylabel("Frequência", fontweight='bold')
    axes[0, 0].set_title("Distribuição Completa do Valor Financiado", fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Histograma até 95º percentil (remover outliers)
    q95 = valores.quantile(0.95)
    valores_sem_outliers = valores[valores <= q95]
    axes[0, 1].hist(valores_sem_outliers, bins=80, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(valores_sem_outliers.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Média: R$ {valores_sem_outliers.mean():,.0f}')
    axes[0, 1].axvline(valores_sem_outliers.median(), color='orange', linestyle='--', linewidth=2, 
                       label=f'Mediana: R$ {valores_sem_outliers.median():,.0f}')
    axes[0, 1].set_xlabel("Valor Financiado (R$)", fontweight='bold')
    axes[0, 1].set_ylabel("Frequência", fontweight='bold')
    axes[0, 1].set_title(f"Distribuição até 95º Percentil (≤ R$ {q95:,.0f})", fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Boxplot
    axes[1, 0].boxplot(valores, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', color='darkgreen'),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(color='darkgreen'),
                       capprops=dict(color='darkgreen'))
    axes[1, 0].set_ylabel("Valor Financiado (R$)", fontweight='bold')
    axes[1, 0].set_title("Boxplot do Valor Financiado", fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Distribuição por faixas de valor
    bins = [0, 500, 1000, 2000, 3000, 5000, 10000, valores.max()+1]
    labels = ['<500', '500-1k', '1k-2k', '2k-3k', '3k-5k', '5k-10k', '>10k']
    df['faixa_valor'] = pd.cut(df['valor_financiado'], bins=bins, labels=labels, right=False)
    faixa_counts = df['faixa_valor'].value_counts().sort_index()
    
    axes[1, 1].bar(range(len(faixa_counts)), faixa_counts.values, color='green', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xticks(range(len(faixa_counts)))
    axes[1, 1].set_xticklabels(faixa_counts.index, rotation=45, ha='right')
    axes[1, 1].set_xlabel("Faixa de Valor (R$)", fontweight='bold')
    axes[1, 1].set_ylabel("Número de Contratos", fontweight='bold')
    axes[1, 1].set_title("Contratos por Faixa de Valor Financiado", fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores e percentuais nas barras
    for i, v in enumerate(faixa_counts.values):
        perc = v / len(df) * 100
        axes[1, 1].text(i, v, f'{v}\n({perc:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_distribuicao_valor_financiado.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico salvo: 07_distribuicao_valor_financiado.png")


def generate_summary_report(df):
    """Gera relatório resumo em texto"""
    print("\n" + "="*80)
    print("GERANDO RELATÓRIO RESUMO")
    print("="*80)
    
    report_path = os.path.join(OUTPUT_DIR, "00_RELATORIO_RESUMO.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RELATÓRIO DE ANÁLISE EXPLORATÓRIA - DATASET TESTE EXTERNO\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Data de geração: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {DATA_PATH}\n\n")
        
        f.write("--- INFORMAÇÕES GERAIS ---\n")
        f.write(f"Total de registros: {len(df):,}\n")
        f.write(f"Total de colunas: {len(df.columns)}\n\n")
        
        if 'Data_inicio' in df.columns:
            f.write("--- PERÍODO ---\n")
            f.write(f"Data início: {df['Data_inicio'].min()}\n")
            f.write(f"Data fim: {df['Data_inicio'].max()}\n\n")
        
        if 'uf' in df.columns:
            f.write("--- DISTRIBUIÇÃO GEOGRÁFICA ---\n")
            uf_counts = df['uf'].value_counts()
            f.write(f"Estados únicos: {len(uf_counts)}\n")
            f.write(f"Estado com mais contratos: {uf_counts.index[0]} ({uf_counts.iloc[0]:,} contratos)\n\n")
        
        if 'idade' in df.columns:
            f.write("--- PERFIL ETÁRIO ---\n")
            f.write(f"Idade média: {df['idade'].mean():.1f} anos\n")
            f.write(f"Idade mediana: {df['idade'].median():.1f} anos\n")
            f.write(f"Faixa etária: {df['idade'].min():.0f} - {df['idade'].max():.0f} anos\n\n")
        
        if 'valor_financiado' in df.columns:
            f.write("--- VALORES FINANCIADOS ---\n")
            f.write(f"Valor médio: R$ {df['valor_financiado'].mean():,.2f}\n")
            f.write(f"Valor mediano: R$ {df['valor_financiado'].median():,.2f}\n")
            f.write(f"Valor total: R$ {df['valor_financiado'].sum():,.2f}\n")
            f.write(f"Faixa de valores: R$ {df['valor_financiado'].min():,.2f} - R$ {df['valor_financiado'].max():,.2f}\n\n")
        
        if 'descricao_da_Profissao' in df.columns:
            f.write("--- TOP 5 PROFISSÕES ---\n")
            top_prof = df['descricao_da_Profissao'].value_counts().head(5)
            for i, (prof, count) in enumerate(top_prof.items(), 1):
                f.write(f"{i}. {prof}: {count:,} ({count/len(df)*100:.2f}%)\n")
            f.write("\n")
        
        if 'Mercadoria' in df.columns:
            f.write("--- TOP 5 PRODUTOS ---\n")
            top_prod = df['Mercadoria'].value_counts().head(5)
            for i, (prod, count) in enumerate(top_prod.items(), 1):
                f.write(f"{i}. {prod}: {count:,} ({count/len(df)*100:.2f}%)\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("Análise concluída com sucesso!\n")
        f.write(f"Gráficos salvos em: {OUTPUT_DIR}\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Relatório salvo: 00_RELATORIO_RESUMO.txt")


def main():
    """Função principal"""
    print("\n" + "="*80)
    print("ANÁLISE EXPLORATÓRIA - DATASET TESTE EXTERNO")
    print("="*80)
    
    # Carregar dados
    df = load_data()
    if df is None:
        return
    
    # Executar análises
    basic_info(df)
    plot_contratos_por_estado_heatmap(df)
    plot_contratos_por_mes(df)
    plot_distribuicao_idade(df)
    plot_profissoes_comuns(df)
    plot_itens_comprados(df)
    plot_distribuicao_valor_financiado(df)
    generate_summary_report(df)
    
    print("\n" + "="*80)
    print("✓ ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("="*80)
    print(f"\nTodos os gráficos foram salvos em: {OUTPUT_DIR}")
    print("\nArquivos gerados:")
    for file in sorted(os.listdir(OUTPUT_DIR)):
        if file.endswith(('.png', '.txt')):
            print(f"  - {file}")
    print("\n")


if __name__ == "__main__":
    main()
