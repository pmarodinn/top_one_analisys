import os
import re
import json
import unidecode
import numpy as np
import pandas as pd
from datetime import datetime, date
from collections import Counter
from difflib import get_close_matches
from rapidfuzz import process, fuzz

### Paths
FILE_PATH = os.path.abspath(__file__)
FILE_FOLDER = os.path.dirname(FILE_PATH)
ROOT_FOLDER = os.path.dirname(os.path.dirname(FILE_FOLDER))  # Volta 2 níveis (code -> top_one_model_v2)

data_path = os.path.join(ROOT_FOLDER, "data", "dataset_interno", "dataset_interno_top_one.xlsx")


### Functions
def capital_to_acronym(capital_name):
    # capital_name = unidecode.unidecode(str(capital_name)).title().strip()
    capital_mapping = {
        'Aracaju': 'SE',
        'Belém': 'PA',
        'Belo Horizonte': 'MG',
        'Boa Vista': 'RR',
        'Brasília': 'DF',
        'Campo Grande': 'MS',
        'Cuiabá': 'MT',
        'Curitiba': 'PR',
        'Florianópolis': 'SC',
        'Fortaleza': 'CE',
        'Goiânia': 'GO',
        'João Pessoa': 'PB',
        'Macapá': 'AP',
        'Maceió': 'AL',
        'Manaus': 'AM',
        'Natal': 'RN',
        'Palmas': 'TO',
        'Porto Alegre': 'RS',
        'Porto Velho': 'RO',
        'Recife': 'PE',
        'Rio Branco': 'AC',
        'Rio de Janeiro': 'RJ',
        'Salvador': 'BA',
        'São Luís': 'MA',
        'São Paulo': 'SP',
        'Teresina': 'PI',
        'Vitória': 'ES'
    }
    return capital_mapping.get(capital_name, None)

def state_to_acronym(state_name):
    state_name = unidecode.unidecode(str(state_name)).title().strip()
    state_mapping = {
        'Acre': 'AC',
        'Alagoas': 'AL',
        'Amapa': 'AP',
        'Amazonas': 'AM',
        'Bahia': 'BA',
        'Ceara': 'CE',
        'Distrito Federal': 'DF',
        'Espírito Santo': 'ES',
        'Goias': 'GO',
        'Maranhao': 'MA',
        'Mato Grosso': 'MT',
        'Mato Grosso do Sul': 'MS',
        'Minas Gerais': 'MG',
        'Para': 'PA',
        'Paraiba': 'PB',
        'Parana': 'PR',
        'Pernambuco': 'PE',
        'Piaui': 'PI',
        'Rio de Janeiro': 'RJ',
        'Rio Grande do Norte': 'RN',
        'Rio Grande do Sul': 'RS',
        'Rondonia': 'RO',
        'Roraima': 'RR',
        'Santa Catarina': 'SC',
        'Sao Paulo': 'SP',
        'Sergipe': 'SE',
        'Tocantins': 'TO'
    }
    return state_mapping.get(state_name, None)

def get_top_n_repeated_elements(arr, n):
    # Count the occurrences of each element
    element_counts = Counter(arr)

    # Get the n most common elements and their counts
    most_common_elements = element_counts.most_common(n)

    # Extract only the elements (discarding their counts)
    top_elements = [element for element, count in most_common_elements]

    return top_elements

def remove_consecutive_duplicates(input_string):
    if not input_string:
        return ""

    result = [input_string[0]]  # Start with the first character

    for i in range(1, len(input_string)):
        if input_string[i] != result[-1]:  # Check if current character is different from the last added
            result.append(input_string[i])

    return "".join(result)

def standirize_merc(merc):

    merc = unidecode.unidecode(str(merc)).lower().strip()

    merc = re.sub(r'^\d+\s*|\s*\d.*$', '', merc).strip()
    merc = re.sub(r'\d+(\s*[xX]\s*\d+)?', '', merc).strip()
    merc = re.sub(r'\b( novo | nova | usado | usada | seminovo | seminova | original | importado | nacional | atacado | varejo | barato | baratoo | baratooo | oferta | ofertas | ofertao | ofertaoo )\b', '', merc)
    merc = re.sub(r'\b( ml\d+| kg\d+| g\d+| l\d+| un\d+| cx\d+| pc\d+| par\d+| lt\d+| cm\d+| mm\d+| mts?\d+)\b', '', merc)  # unidades com números
    merc = re.sub(r'\b(kt| cs|cs |)\b', '', merc)  # unidades com números
    # merc = re.sub(r'\s+', ' ', merc).strip()

    std_mercs = re.split(r'/ | /|/|, | ,|,|\* | \*|\*|\+ | \+|\+| e | e. |- | -|-', merc)

    # remove repeated letters in each word
    std_mercs = [remove_consecutive_duplicates(item).strip() for item in std_mercs if item]

    return std_mercs

def classificar_produto(produto, dicionario):
    """Classifica um produto individual usando o dicionário"""
    if not isinstance(produto, str) or not produto.strip():
        return "invalido"
    p = produto.lower().strip()
    if len(p) < 3 or re.fullmatch(r'[\W\d_]+', p):
        return "invalido"
    if p in dicionario:
        return dicionario[p]
    similar = get_close_matches(p, dicionario.keys(), n=1, cutoff=0.8)
    if similar:
        return dicionario[similar[0]]
    return "invalido"

def classificar_mercadorias(merc, dicionario):
    """Retorna todas as categorias válidas encontradas na linha"""
    itens = standirize_merc(merc)
    categorias = [classificar_produto(i, dicionario) for i in itens if i]
    # mantém apenas categorias válidas (remove 'outros' e 'invalido')
    categorias_validas = [c for c in categorias if c not in ('outros', 'invalido')]
    return list(sorted(set(categorias_validas)))

def classificar_prof(prof, dicionario):
    """Classifica um produto individual usando o dicionário"""
    if not isinstance(prof, str) or not prof.strip():
        return "invalido"
    p = prof.lower().strip()
    if len(p) < 3 or re.fullmatch(r'[\W\d_]+', p):
        return "invalido"
    if p in dicionario:
        return dicionario[p]
    similar = get_close_matches(p, dicionario.keys(), n=1, cutoff=0.8)
    if similar:
        return dicionario[similar[0]]
    return "invalido"

def standirize_profession(profession):
    if type(profession) is not str:
        return "Desconhecido"
    profession = profession.title().rstrip()
    profession = profession.replace('.', '').strip()
    profession = profession.replace('Trab ', '').strip()
    profession = profession.replace('Ass ', 'Assistente ').strip()
    profession = profession.replace('Tec ', 'Tecnico ').strip()
    profession = profession.replace('Tecnica ', 'Tecnico ').strip()
    profession = profession.replace('Ax ', 'Auxiliar ').strip()
    profession = profession.replace('Aux ', 'Auxiliar ').strip()
    profession = profession.replace('Mec ', 'Mecanico ').strip()
    profession = profession.replace('Cpp', 'Carpinteiro').strip()
    profession = profession.replace('Serv ', 'Servicos ').strip()
    profession = profession.replace('Servico ', 'Servicos ').strip()
    profession = profession[:-1] + 'o' if profession.endswith('a') else profession
    profession = profession.replace('Prod', 'Producao').strip() if profession.endswith('Prod') else profession
    profession = profession.replace('Ven', 'Vendas').strip() if profession.endswith('Ven') else profession
    profession = profession.replace('Escr', 'Escrivao').strip() if profession.endswith('Prod') else profession
    profession = profession.replace('Aut', 'Autonomo').strip() if profession.endswith('Aut') else profession
    profession = profession.replace('Apos', 'Aposentado').strip() if profession.endswith('Apos') else profession
    profession = profession.replace('Ger', 'Geral').strip() if profession.endswith('Ger') else profession
    profession = profession.replace('Padaria', 'Padari').strip() if profession.endswith('Padari') else profession
    return profession

def add_ipca(df, df_ipca):
    # Converter as datas
    df["Data_Lancamento"] = pd.to_datetime(df["Data_Lancamento"])
    df_ipca["data"] = pd.to_datetime(df_ipca["data"])

    # Ordenar por data
    df = df.sort_values("Data_Lancamento")
    df_ipca = df_ipca.sort_values("data")

    # Merge asof (join pela data mais próxima anterior)
    df = pd.merge_asof(
        df,
        df_ipca[["data", "valor"]].rename(columns={"data": "data_ipca", "valor": "IPCA"}),
        left_on="Data_Lancamento",
        right_on="data_ipca",
        direction="backward"
    )

    # Remover coluna auxiliar
    df = df.drop(columns=["data_ipca"])

    return df

def normalize_city(name):
    """Remove acentos, deixa em minúsculo e elimina pontuação."""
    if pd.isna(name):
        return ""
    name = unidecode.unidecode(str(name)).lower()
    name = name.replace("-", " ").replace(".", " ")
    name = " ".join(name.split())  # remove espaços duplicados
    return name.strip()

def add_idh_info(df, df_idh, uf_col="uf", cidade_col="Cidade_Loja", municipio_col="municipio", similarity_threshold=85):
    """
    Faz merge entre df e df_idh usando nomes de cidade (com correspondência aproximada).
    Descarta linhas ambíguas (vários matches possíveis).
    """

    # Normaliza nomes
    df["Cidade_Loja_norm"] = df[cidade_col].apply(normalize_city)
    df_idh["municipio_norm"] = df_idh[municipio_col].apply(normalize_city)

    # Dicionário para armazenar matches únicos
    match_map = {}
    used_targets = set()

    for city in df["Cidade_Loja_norm"].unique():
        # Busca cidade mais similar
        result = process.extract(city, df_idh["municipio_norm"], scorer=fuzz.token_sort_ratio, limit=2)

        if not result:
            continue

        best_match, best_score, best_idx = result[0]

        # Se não atingir limiar de confiança → ignora
        if best_score < similarity_threshold:
            continue

        # Se houver 2 cidades com pontuação similar, marca como ambíguo
        if len(result) > 1 and abs(result[0][1] - result[1][1]) < 3:
            continue

        # Se o município já foi usado em outro match, evita duplicar
        if best_match in used_targets:
            continue

        match_map[city] = best_match
        used_targets.add(best_match)

    # Cria DataFrame de mapeamento único
    mapping_df = pd.DataFrame(list(match_map.items()), columns=["Cidade_Loja_norm", "municipio_norm"])

    # Junta com IDH
    merged = df.merge(mapping_df, on="Cidade_Loja_norm", how="left")
    merged = merged.merge(df_idh, on="municipio_norm", how="left")

    # Remove linhas que não tiveram correspondência válida
    merged = merged.dropna(subset=["idhm_2010"])

    # Remove colunas auxiliares
    merged = merged.drop(columns=["Cidade_Loja_norm", "municipio_norm", "fonte", "url_fonte", "metodo_coleta", "timestamp_coleta", "data_base", "ano_censo"], errors="ignore")

    return merged

def analisys(df):
    # Calcula lucros
    expected_profit = ((df["valor_inicial_da_prestacao"] * df["plano_financiamento"]) - df["valor_financiado"]).sum()
    total_profit = df["lucro"].sum()

    # Exibe resultados
    print("\n=== Avaliação Financeira ===")
    print(f"Total de contratos:       {len(df)}")
    print(f"Lucro esperado total:     R$ {expected_profit:,.2f}")
    print(f"Lucro real total:         R$ {total_profit:,.2f}")
    print(f"Perda total:              R$ {expected_profit - total_profit:,.2f}")
    print(f"Perda percentual:         {(expected_profit - total_profit) / expected_profit * 100:.2f}%")

####################################################################################################################
# create new dataset
def change_dataset(data_path):
    """Load dataset from Excel file and return as a pandas DataFrame."""

    # Define output directory
    output_dir = os.path.join(ROOT_FOLDER, "data", "datasets_tratados")
    os.makedirs(output_dir, exist_ok=True)
    
    # define output paths
    base_name = "dataset_interno_top_one"
    df_by_contract_path = os.path.join(output_dir, f"{base_name}_by_contract.xlsx")
    df_classified_products_path = os.path.join(output_dir, f"{base_name}_classified_products.xlsx")
    df_augmented_path = os.path.join(output_dir, f"{base_name}_augmented.xlsx")
    df_city_dens_path = os.path.join(output_dir, f"{base_name}_city_size.xlsx")
    df_fuel_path = os.path.join(output_dir, f"{base_name}_fuel.xlsx")
    df_cb_longbasica_path = os.path.join(output_dir, f"{base_name}_cestabasica.xlsx")
    df_final_csv_path = os.path.join(output_dir, f"{base_name}_final.csv")

    # if output by contract file does not exist, create it
    if not os.path.exists(df_by_contract_path):
        
        df = pd.read_excel(data_path)
        id_list = set(df["contrato_id"].tolist())
        
        df_new = pd.DataFrame()
        for idx, contrato_id in enumerate(id_list):
            df_subset = df[df["contrato_id"] == contrato_id]

            # if  df_subset[['Mercadoria', 'idade', 'genero', 'tipo_residencia', 'Cargo', 'descricao_da_Profissao', 'Cidade_Loja', 'Modalidade_da_Proposta']].isnull().any().any():
            #     continue

            paid_perc = (len(df_subset[df_subset['data_de_pagamento'] != 'SEM DATA DE PAGAMENTO']))/(df_subset['plano_financiamento'].values[0])
            salary_perc = df_subset['valor_inicial_da_prestacao'].values[0]/df_subset['renda_cliente'].values[0] if df_subset['renda_cliente'].values[0] != 0 else 1
            date_init = df_subset[df_subset['Parcela_Numero'] == 1]['Data_Vencimento_da_Prestacao'].values[0].astype('datetime64[D]')
            date_init = date_init.astype(datetime)

            recieved_money = df_subset['valor_inicial_da_prestacao'].values[0]*df_subset['plano_financiamento'].values[0]*paid_perc
            profit = recieved_money - df_subset['valor_financiado'].values[0]

            # correct city name
            city = df_subset['Cidade_Loja'].values[0].title().rstrip()
            city = city.replace('Do S', 'Do Sul') if city.lower().endswith('do s') else city
            city = city.replace('Do N', 'Do Norte') if city.lower().endswith('do n') else city
            city = city.replace('Do Oes', 'Do Oeste') if city.lower().endswith('do oes') else city
            city = city.replace('Do O', 'Do Oeste') if city.lower().endswith('do o') else city

            # correct profession


            # Get the first row by 'Parcela_Numero'
            first_row = df_subset.sort_values('Parcela_Numero').iloc[0].copy().drop(labels=['Parcela_Numero', 'Data_Vencimento_da_Prestacao', 'data_de_pagamento'])
            first_row['Data_inicio'] = date_init

            # Add paid percentage as a new column
            first_row['pago_perc'] = paid_perc
            first_row['salario_perc'] = salary_perc
            first_row['lucro'] = profit
            first_row['Cidade_Loja'] = city
            first_row['aceitar'] = first_row['lucro']>0

            # Append this single-row DataFrame to df_new
            df_new = pd.concat([df_new, first_row.to_frame().T], ignore_index=True)

            if idx % 1000 == 0:
                partial_path = os.path.join(ROOT_FOLDER, "data", "dataset_interno", "dataset_interno_top_one_partial.xlsx")
                df_new.to_excel(partial_path, index=False)
                print(f"Processed {idx} / {len(id_list)} contrato_ids.\r", end="")
        
        # save new dataset
        df_new.to_excel(df_by_contract_path, index=False)

    if not os.path.exists(df_classified_products_path):
        df = pd.read_excel(df_by_contract_path)

        analisys(df)

        with open(os.path.join(ROOT_FOLDER, 'data', 'dicionario', 'dict.json'), "r", encoding="utf-8") as f:
            categoria_dict = json.load(f)

        # 1️⃣ Gera lista de categorias válidas
        df["categorias"] = df["Mercadoria"].apply(lambda x: classificar_mercadorias(x, categoria_dict))
        df['descricao_da_Profissao'] = df["descricao_da_Profissao"].apply(lambda x: standirize_profession(x))

        # 2️⃣ Remove linhas sem nenhuma categoria válida
        df = df[df["categorias"].map(len) > 0].reset_index(drop=True)

        df.to_excel(df_classified_products_path)

    if not os.path.exists(df_augmented_path):
        df_ipca = pd.read_csv(os.path.join(ROOT_FOLDER, 'data', 'external_data', 'ipca', 'ipca_historico_20250920_201436.csv'))
        df_idh  = pd.read_csv(os.path.join(ROOT_FOLDER, 'data', 'external_data', 'idh_municipios', 'idh_completo_pnud_2010_20250804_014433.csv'))

        # print(df_ipca.head())
        # print(df_idh.head())
        # print(df_fuel.head())

        df = pd.read_excel(df_classified_products_path)
        id_list = set(df["contrato_id"].tolist())

        df = add_ipca(df, df_ipca)
        df = add_idh_info(df, df_idh)

        df.to_excel(df_augmented_path, index=False)

    if not os.path.exists(df_city_dens_path):

        df_city_pop_path = os.path.join(ROOT_FOLDER, 'data', 'external_data', 'pop_size', 'POP2024_20241101.xls')
        df_city_size_path = os.path.join(ROOT_FOLDER, 'data', 'external_data', 'pop_size', 'AR_BR_RG_UF_RGINT_RGI_MUN_2024.xls')

        df = pd.read_excel(df_augmented_path)
        df_city_pop = pd.read_excel(df_city_pop_path, sheet_name='MUNICÍPIOS')
        df_city_size = pd.read_excel(df_city_size_path, sheet_name='AR_BR_MUN_2024')

        df_city_pop = df_city_pop[['UF', 'NOME DO MUNICÍPIO', 'POPULAÇÃO ESTIMADA']]
        df_city_size = df_city_size[['NM_UF_SIGLA', 'NM_MUN', 'AR_MUN_2024']]

        # --- Normalize names for merging ---
        df_city_pop = df_city_pop.rename(columns={
            'UF': 'uf',
            'NOME DO MUNICÍPIO': 'municipio',
            'POPULAÇÃO ESTIMADA': 'populacao'
        })
        df_city_size = df_city_size.rename(columns={
            'NM_UF_SIGLA': 'uf',
            'NM_MUN': 'municipio',
            'AR_MUN_2024': 'area'
        })

        # --- Normalize string formatting to ensure matching ---
        for d in [df, df_city_pop, df_city_size]:
            d['uf'] = d['uf'].str.strip().str.upper()
            d['municipio'] = d['municipio'].str.strip().str.upper()

        # --- Merge population and area into df ---
        df = df.merge(df_city_pop, on=['uf', 'municipio'], how='left')
        df = df.merge(df_city_size, on=['uf', 'municipio'], how='left')

        # --- Compute population density ---
        df['densidade_pop'] = df['populacao'] / df['area']

        # Optional: handle missing or zero area values
        df.loc[df['area'] == 0, 'densidade_pop'] = None

        df.to_excel(df_city_dens_path, index=False)
    
    if not os.path.exists(df_fuel_path):
        df = pd.read_excel(df_city_dens_path)
        
        df_fuel = pd.read_csv(os.path.join(ROOT_FOLDER, 'data', 'external_data', 'fuel_prices', 'processed_data', 'dados_completos_de_combustivel.csv'))

        df_fuel = df_fuel[['ESTADO', 'UNIDADE DE MEDIDA', 'PRODUTO', 'ano', 'PREÇO MÉDIO REVENDA']]
        df_fuel = df_fuel[df_fuel['UNIDADE DE MEDIDA'] == 'R$/l']
        df_fuel['ESTADO'] = df_fuel['ESTADO'].apply(state_to_acronym)

        # divide fuel by year
        for year in df_fuel['ano'].unique():
            if year not in df['Data_inicio'].dt.year.values:
                continue
            temp_df = df_fuel[df_fuel['ano'] == year].copy()
            #calculate mean price of all fuels per state
            temp_df = (
                temp_df.groupby(["ESTADO", "ano"])["PREÇO MÉDIO REVENDA"]
                .mean()
                .reset_index()
                .rename(columns={"PREÇO MÉDIO REVENDA": "PREÇO MÉDIO"})
            )
            mean_value = temp_df['PREÇO MÉDIO'].mean()
            df.loc[df['Data_inicio'].dt.year == year, 'preco_combustivel'] = df.loc[df['Data_inicio'].dt.year == year].merge(
                temp_df,
                left_on='uf',
                right_on='ESTADO',
                how='left'
            )['PREÇO MÉDIO'].values

            std_value = temp_df['PREÇO MÉDIO'].std()
            print(f"Ano {year} - Preço médio combustível: {mean_value:.4f} | Desvio padrão: {std_value:.4f}")

            q1 = temp_df['PREÇO MÉDIO'].quantile(0.33)
            q2 = temp_df['PREÇO MÉDIO'].quantile(0.66)

            def classificar_preco(preco):
                if preco <= q1:
                    return "barato"
                elif preco <= q2:
                    return "medio"
                else:
                    return "caro"

            mask = df['Data_inicio'].dt.year == year

            df.loc[mask, 'class_preco_combustivel'] = (
                df.loc[mask, 'preco_combustivel']
                .apply(classificar_preco)
            )
            # count the number of occurrences of each class
            class_counts = df.loc[mask, 'class_preco_combustivel'].value_counts()
            print(f"Ano {year} - Contagem de classes de preço de combustível:")
            print(class_counts)
            print("Valores unicos da coluna preco_combustivel:", sorted(df['preco_combustivel'].unique()))
        
        df = df[~df['class_preco_combustivel'].isna()]
        print(df['Data_inicio'].dt.year.value_counts())

        df.to_excel(df_fuel_path, index=False)

    if not os.path.exists(df_cb_longbasica_path):
        df = pd.read_excel(df_fuel_path)
        df_cb_23  =  pd.read_excel(os.path.join(ROOT_FOLDER, 'data', 'external_data', 'cesta_basica_capitais', 'raw_data', 'cesta_basica_dieese_2023_012023_122023.xls'))
        df_cb_24  =  pd.read_excel(os.path.join(ROOT_FOLDER, 'data', 'external_data', 'cesta_basica_capitais', 'raw_data', 'cesta_basica_dieese_2024_012024_122024.xls'))

        def process_cb_year(df_cb, df, year):
            # Fazer a conversão wide → long
            df_cb_long = df_cb.melt(
                id_vars="data",
                var_name="estado",
                value_name="valor_cestabasica"
            )

            df_cb_long["data"] = pd.to_datetime(df_cb_long["data"], format="%m-%Y")
            df_cb_long["estado"] = df_cb_long["estado"].apply(capital_to_acronym)

            # df_year = df_year[['uf', 'Data_inicio']]

            # Garantir datetime
            df["Data_inicio"] = pd.to_datetime(df["Data_inicio"])
            df_cb_long["data"] = pd.to_datetime(df_cb_long["data"])

            df["preco_cesta_basica"] = np.nan  # nova coluna

            # Apenas linhas de 2023
            df_year = df[df["Data_inicio"].dt.year == year].copy()

            df_out = pd.DataFrame()
            # Loop estado por estado
            for uf in df_year["uf"].unique():
                print(f"Processando estado: {uf}")

                # Filtrar apenas aquele estado
                left = df_year[df_year["uf"] == uf].copy()
                right = df_cb_long[df_cb_long["estado"] == uf].copy()

                if right.empty:
                    continue  # esse estado não tem cesta básica → deixa como NaN

                # Ordenação obrigatória para merge_asof
                left = left.sort_values("Data_inicio")
                right = right.sort_values("data")

                merged = pd.merge_asof(
                    left,
                    right,
                    left_on="Data_inicio",
                    right_on="data",
                    direction="nearest"
                )

                df_out = pd.concat([df_out, merged], ignore_index=True)

            df_out = df_out.drop(columns=["data", "estado"])
            return df_out
        
        df_2023 = process_cb_year(df_cb_23, df, 2023)
        print(df_2023.columns)
        df_2024 = process_cb_year(df_cb_24, df, 2024)

        df_final = pd.concat([df_2023, df_2024], ignore_index=True)

        df_final['preco_cb_perc'] = df_final['valor_cestabasica']/df_final['renda_cliente']
        df_final.to_excel(df_cb_longbasica_path, index=False)
        
    # df = pd.read_excel(df_classified_products_path)
    # n_values = 200
    # common_values = df['Cidade_Loja'].value_counts().head(n_values)
    # # Create dictionary where each key is the index, and value is empty string
    # data_dict = {name: "" for name in common_values.index}

    # # Save as JSON
    # with open(os.path.join(FILE_FOLDER, 'dict_prof.txt'), "w", encoding="utf-8") as f:
    #     json.dump(data_dict, f, indent=4, ensure_ascii=False)
    # print(common_values)

    df = pd.read_excel(df_cb_longbasica_path)
    
    # Verificar NaNs antes de remover
    print("\n🔍 Verificando valores ausentes por coluna:")
    nan_counts = df.isnull().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
    if len(nan_counts) > 0:
        print(nan_counts)
    else:
        print("Nenhum valor ausente encontrado!")
    
    print(f"\n📊 Total de linhas antes de limpar: {len(df)}")
    
    # Drop NaN apenas nas colunas críticas (não em todas)
    critical_columns = ['lucro', 'valor_inicial_da_prestacao', 'valor_financiado', 
                       'plano_financiamento', 'Data_inicio']
    df = df.dropna(subset=critical_columns).reset_index(drop=True)
    
    print(f"📊 Total de linhas após limpar: {len(df)}")
    
    analisys(df)
    
    # Save final dataset as CSV with ; separator and , as decimal
    print(f"\n💾 Salvando dataset final em CSV: {df_final_csv_path}")
    df.to_csv(df_final_csv_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    print(f"✅ Dataset final salvo com sucesso!")
    print(f"   - Linhas: {len(df)}")
    print(f"   - Colunas: {len(df.columns)}")
    print(f"   - Separador: ';'")
    print(f"   - Decimal: ','")
    
    return df

#----------------------------------------------------------------------------------------------------------------#

### Main
if __name__ == "__main__":
    change_dataset(data_path)