"""
Mapas de calor geográficos (Brasil)
===================================

Gera quatro mapas temáticos com base na base interna:
1. Concentração de inadimplentes totais
2. Concentração de adimplentes totais
3. Concentração de inadimplentes parciais lucrativos
4. Taxa relativa de inadimplentes parciais lucrativos (% sobre contratos)

O shapefile municipal deve estar em ../data/BR_Municipios_2024.shp
"""

import os
import unicodedata
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# Alguns shapefiles distribuídos sem o arquivo .shx precisam desta flag para serem reconstruídos pelo GDAL
os.environ.setdefault('SHAPE_RESTORE_SHX', 'YES')

warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")

DATA_PATH = '../../data/dataset_interno_top_one_atualizado.csv'
SHAPE_PATH = '../../data/BR_Municipios_2024.shp'
OUTPUT_DIR = '../../graficos/analises_dataset/ANALISE_GEOGRAFICA_MAPAS'
SUMMARY_CSV = os.path.join(OUTPUT_DIR, 'resumo_municipios.csv')

sns.set_style('white')
plt.rcParams['figure.figsize'] = (10, 12)
plt.rcParams['axes.facecolor'] = 'white'


def normalize_text(value: str) -> str:
    if pd.isna(value):
        return 'DESCONHECIDO'
    text = str(value).strip().upper()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace('-', ' ')
    text = ' '.join(text.split())
    return text


def detect_column(columns, keywords):
    keywords = [kw.lower() for kw in keywords]
    for col in columns:
        lower = col.lower()
        if any(kw in lower for kw in keywords):
            return col
    return None


def load_base_dataset(path: str) -> pd.DataFrame:
    print(f"📂 Carregando base principal: {path}")
    df = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8', low_memory=False)
    df.columns = [col.strip() for col in df.columns]

    df['pago_perc'] = pd.to_numeric(df['pago_perc'].astype(str).str.replace(',', '.'), errors='coerce')
    df['lucro'] = pd.to_numeric(df['lucro'].astype(str).str.replace(',', '.'), errors='coerce')
    df['pago_perc'] = df['pago_perc'].fillna(0).clip(lower=0)
    df['lucro'] = df['lucro'].fillna(0)

    df['municipio_norm'] = df['municipio'].apply(normalize_text)
    df['uf_norm'] = df['uf'].astype(str).str.strip().str.upper()
    df['geo_key'] = df['municipio_norm'] + '_' + df['uf_norm']

    df['inadimplente_total'] = (df['pago_perc'] < 1).astype(int)
    df['adimplente_total'] = (df['pago_perc'] >= 1).astype(int)
    df['inadimplente_parcial'] = ((df['pago_perc'] > 0) & (df['pago_perc'] < 1)).astype(int)
    df['inad_parcial_lucrativo'] = (
        (df['pago_perc'] > 0) & (df['pago_perc'] < 1) & (df['lucro'] > 0)
    ).astype(int)
    df['inad_parcial_prejuizo'] = (
        (df['pago_perc'] > 0) & (df['pago_perc'] < 1) & (df['lucro'] <= 0)
    ).astype(int)

    print(f"   → Registros válidos: {len(df):,}")
    return df


def load_municipal_geometries(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Shapefile não encontrado em '{path}'. Adicione BR_Municipios_2024.shp ao diretório data/."
        )

    print(f"🗺️  Carregando shapefile municipal: {path}")
    # Força encoding UTF-8 para corrigir caracteres como 'Ã£' (mojibake de UTF-8 lido como Latin-1)
    try:
        gdf = gpd.read_file(path, encoding='utf-8')
    except Exception as e:
        print(f"   ⚠️  Erro ao ler com UTF-8, tentando padrão: {e}")
        gdf = gpd.read_file(path)
    print(f"   ℹ️  Colunas disponíveis no Shapefile: {gdf.columns.tolist()}")

    # Prioriza colunas de NOME e SIGLA (padrão IBGE: NM_MUN, SIGLA_UF)
    muni_col = detect_column(gdf.columns, ['nm_mun', 'nome_municipio', 'municipio_nome'])
    if not muni_col:
        muni_col = detect_column(gdf.columns, ['mun', 'nome'])
        
    uf_col = detect_column(gdf.columns, ['sigla', 'sigla_uf', 'uf_sigla'])
    if not uf_col:
        uf_col = detect_column(gdf.columns, ['uf', 'estado'])

    print(f"   → Coluna Município detectada: {muni_col}")
    print(f"   → Coluna UF detectada: {uf_col}")

    if muni_col is None or uf_col is None:
        raise ValueError(
            "Não foi possível identificar as colunas de município/UF no shapefile. "
            "Certifique-se de que exista um campo com o nome do município e outro com a UF."
        )

    gdf['municipio_norm'] = gdf[muni_col].apply(normalize_text)
    gdf['uf_norm'] = gdf[uf_col].astype(str).str.upper().str.strip()
    gdf['geo_key'] = gdf['municipio_norm'] + '_' + gdf['uf_norm']

    print(f"   → Geometrias carregadas: {len(gdf):,}")
    return gdf[['geo_key', 'municipio_norm', 'uf_norm', 'geometry']]


def build_city_metrics(df: pd.DataFrame) -> pd.DataFrame:
    print("📊 Agregando métricas por município...")
    grouped = df.groupby('geo_key').agg(
        municipio=('municipio', 'first'),
        uf=('uf', 'first'),
        total_contratos=('contrato_id', 'count'),
        inadimplentes_totais=('inadimplente_total', 'sum'),
        adimplentes_totais=('adimplente_total', 'sum'),
        inad_parciais_lucrativos=('inad_parcial_lucrativo', 'sum'),
        inad_parciais_prejuizo=('inad_parcial_prejuizo', 'sum')
    ).reset_index()

    # Evita divisão por zero usando np.where
    grouped['inad_parciais_lucrativos_pct'] = np.where(
        grouped['total_contratos'] > 0,
        grouped['inad_parciais_lucrativos'] / grouped['total_contratos'],
        0.0
    )

    # Percentual em relação ao total do país (ajuda na visualização geográfica)
    total_inad = float(grouped['inadimplentes_totais'].sum())
    total_adimplentes = float(grouped['adimplentes_totais'].sum())
    total_inad_parcial_lucr = float(grouped['inad_parciais_lucrativos'].sum())
    total_inad_parcial_prejuizo = float(grouped['inad_parciais_prejuizo'].sum())

    denom_inad = total_inad if total_inad > 0 else np.nan
    denom_adimplentes = total_adimplentes if total_adimplentes > 0 else np.nan
    denom_inad_parcial = total_inad_parcial_lucr if total_inad_parcial_lucr > 0 else np.nan
    denom_inad_parcial_prejuizo = total_inad_parcial_prejuizo if total_inad_parcial_prejuizo > 0 else np.nan

    grouped['inadimplentes_totais_pct_country'] = (
        grouped['inadimplentes_totais'] / denom_inad
    ).fillna(0)
    grouped['adimplentes_totais_pct_country'] = (
        grouped['adimplentes_totais'] / denom_adimplentes
    ).fillna(0)
    grouped['inad_parciais_lucrativos_pct_country'] = (
        grouped['inad_parciais_lucrativos'] / denom_inad_parcial
    ).fillna(0)
    grouped['inad_parciais_prejuizo_pct_country'] = (
        grouped['inad_parciais_prejuizo'] / denom_inad_parcial_prejuizo
    ).fillna(0)

    print("   → Municípios agregados:", len(grouped))
    return grouped


def plot_metric_map(gdf: gpd.GeoDataFrame, column: str, title: str,
                    legend_label: str, output_name: str, cmap: str = 'Reds',
                    is_percentage: bool = False, normalization: dict | None = None):
    if column not in gdf.columns:
        print(f"[AVISO] Coluna '{column}' não encontrada para o mapa '{title}'.")
        return

    values = gdf[column].fillna(0)
    
    # Debug: Imprimir estatísticas da coluna para verificar escala
    v_min, v_max, v_mean = values.min(), values.max(), values.mean()
    print(f"   ℹ️  Stats para '{column}': Min={v_min:.6f}, Max={v_max:.6f}, Mean={v_mean:.6f}")
    
    norm = build_color_norm(values, normalization)

    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 1. Camada base: Desenha todos os municípios em cinza claro para garantir a delimitação
    gdf.plot(
        ax=ax,
        color='#F5F5F5',
        edgecolor='#999999',  # Bordas mais escuras para melhor definição
        linewidth=0.3         # Linhas mais grossas
    )

    # 2. Camada de dados: Plota apenas onde há valor (ou tudo se fillna(0))
    # Aqui optamos por plotar tudo preenchendo NaNs com 0 para manter a continuidade do colormap
    gdf_plot = gdf.copy()
    gdf_plot[column] = gdf_plot[column].fillna(0)
    
    gdf_plot.plot(
        column=column,
        cmap=cmap,
        linewidth=0.3,        # Linhas mais grossas também na camada de dados
        ax=ax,
        edgecolor='#999999',  # Bordas consistentes
        legend=True,
        legend_kwds={'label': legend_label, 'shrink': 0.6},
        norm=norm
    )
    ax.set_axis_off()

    if is_percentage:
        ax.set_title(f"{title}\n(escala em % do total de contratos)", fontsize=14, fontweight='bold')
    else:
        ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Mapa salvo em {output_path}")


def save_summary_table(metrics: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics.sort_values('total_contratos', ascending=False).to_csv(
        SUMMARY_CSV, sep=';', decimal=',', index=False
    )
    print(f"📁 Resumo salvo em {SUMMARY_CSV}")


def build_color_norm(values: pd.Series, normalization: dict | None):
    values = values.fillna(0).astype(float)
    max_default = normalization.get('min_vmax', 1.0) if normalization else 1.0

    if normalization is None:
        vmax = max(values.max(), max_default)
        return mcolors.Normalize(vmin=0, vmax=vmax)

    strategy = normalization.get('strategy', 'percentile').lower()
    percentile = normalization.get('percentile', 95)
    vmax = np.nanpercentile(values, percentile)
    max_value = normalization.get('max_value')
    if max_value is not None:
        vmax = min(vmax, max_value)
    vmax = max(vmax, normalization.get('min_vmax', max_default))

    if strategy == 'power':
        gamma = normalization.get('gamma', 0.5)
        return mcolors.PowerNorm(gamma=gamma, vmin=0, vmax=vmax)
    if strategy == 'log':
        positive = values[values > 0]
        if len(positive) == 0:
            return mcolors.Normalize(vmin=0, vmax=vmax)
        vmin = max(positive.min(), normalization.get('min_positive', 1e-3))
        return mcolors.LogNorm(vmin=vmin, vmax=vmax)
    if strategy == 'minmax':
        vmin = float(np.nanmin(values)) if np.isfinite(np.nanmin(values)) else 0.0
        vmax = float(np.nanmax(values)) if np.isfinite(np.nanmax(values)) else max_default
        if vmin == vmax:
            vmax = vmin + 1e-6
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    return mcolors.Normalize(vmin=0, vmax=vmax)


def main():
    df = load_base_dataset(DATA_PATH)
    city_metrics = build_city_metrics(df)

    try:
        gdf = load_municipal_geometries(SHAPE_PATH)
    except (FileNotFoundError, ValueError) as exc:
        print(f"❌ {exc}")
        print("Interrompendo geração dos mapas até que o shapefile esteja disponível/correto.")
        save_summary_table(city_metrics)
        return

    # Debug: Verificar chaves de junção
    print("\n🔍 Verificando chaves de junção (geo_key):")
    print(f"   Exemplos CSV: {city_metrics['geo_key'].head().tolist()}")
    print(f"   Exemplos Shapefile: {gdf['geo_key'].head().tolist()}")
    
    common_keys = set(city_metrics['geo_key']).intersection(set(gdf['geo_key']))
    print(f"   → Chaves em comum: {len(common_keys)} de {len(city_metrics)} (CSV) e {len(gdf)} (Shapefile)")

    missing_keys = set(city_metrics['geo_key']) - set(gdf['geo_key'])
    if missing_keys:
        print(f"\n   ⚠️  {len(missing_keys)} municípios do CSV não foram encontrados no Shapefile.")
        print("   Exemplos de municípios não encontrados (CSV):")
        print(f"   {list(missing_keys)[:10]}")
        
        # Tentar identificar padrões nos não encontrados
        print("   Verificando se há problemas de acentuação ou formatação...")
        # Exemplo: comparar 'SAO PAULO_SP' com 'SÃO PAULO_SP' se possível, mas aqui apenas listamos.

    if len(common_keys) == 0:
        print("   ⚠️  AVISO: Nenhuma chave em comum encontrada! Verifique a normalização ou os nomes das colunas.")
        print("   Colunas do Shapefile:", gdf.columns.tolist())
        print("   Exemplo de registro do Shapefile:", gdf.iloc[0].to_dict())

    gdf = gdf.merge(city_metrics, on='geo_key', how='left')
    fill_cols = ['total_contratos', 'inadimplentes_totais', 'adimplentes_totais',
                 'inad_parciais_lucrativos', 'inad_parciais_lucrativos_pct',
                 'inad_parciais_prejuizo', 'inad_parciais_prejuizo_pct_country']
    for col in fill_cols:
        if col in gdf.columns:
            gdf[col] = gdf[col].fillna(0)

    # Percentuais relativos ao Brasil inteiro
    # Use percentage-of-country metrics for better contrast on national map
    metric_configs = [
        {
            'column': 'inadimplentes_totais_pct_country',
            'title': 'Concentração de Inadimplentes Totais (% do país)',
            'legend': 'Percentual do total de inadimplentes (país)',
            'filename': 'heatmap_inadimplentes_totais_pct_country.png',
            'cmap': 'Reds',
            'normalization': {'strategy': 'percentile', 'percentile': 98, 'min_vmax': 1e-4}
        },
        {
            'column': 'adimplentes_totais_pct_country',
            'title': 'Concentração de Adimplentes Totais (% do país)',
            'legend': 'Percentual do total de adimplentes (país)',
            'filename': 'heatmap_adimplentes_totais_pct_country.png',
            'cmap': 'Greens',
            'normalization': {'strategy': 'percentile', 'percentile': 98, 'min_vmax': 1e-4}
        },
        {
            'column': 'inad_parciais_lucrativos_pct_country',
            'title': 'Inadimplentes Parciais Lucrativos (% do país)',
            'legend': 'Percentual do total de inad. parciais lucrativos (país)',
            'filename': 'heatmap_inad_parciais_lucrativos_pct_country.png',
            'cmap': 'Blues',
            'normalization': {'strategy': 'percentile', 'percentile': 98, 'min_vmax': 1e-4}
        },
        {
            'column': 'inad_parciais_prejuizo_pct_country',
            'title': 'Inadimplentes Parciais Prejuízo (% do país)',
            'legend': 'Percentual do total de inad. parciais prejuízo (país)',
            'filename': 'heatmap_inad_parciais_prejuizo_pct_country.png',
            'cmap': 'Purples',
            'normalization': {'strategy': 'percentile', 'percentile': 98, 'min_vmax': 1e-4}
        },
        {
            'column': 'inad_parciais_lucrativos_pct',
            'title': 'Inadimplentes Parciais Lucrativos (% do município)',
            'legend': 'Percentual sobre contratos (município)',
            'filename': 'heatmap_inad_parciais_lucrativos_pct_mun.png',
            'cmap': 'Blues',
            'is_percentage': True,
            'normalization': {'strategy': 'percentile', 'percentile': 95, 'max_value': 1.0, 'min_vmax': 0.01}
        }
    ]
    print("\n🖨️  Gerando mapas...")
    for config in metric_configs:
        plot_metric_map(
            gdf,
            column=config['column'],
            title=config['title'],
            legend_label=config['legend'],
            output_name=config['filename'],
            cmap=config.get('cmap', 'viridis'),
            is_percentage=config.get('is_percentage', False),
            normalization=config.get('normalization')
        )

    save_summary_table(city_metrics)
    print("\n✅ Mapas concluídos!")


if __name__ == '__main__':
    main()
