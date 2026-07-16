
import os

import pandas as pd


# Caminho relativo ao proprio script, funciona independente do CWD.
caminho = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "conjunto_de_treinamento (5).csv",
)
df = pd.read_csv(caminho)

# Remocao de anomalias extremas.
ids_removidos = [5910, 2405, 4568, 5575, 6606, 6654, 6383]
df = df[~df["Id"].isin(ids_removidos)].copy()

# Features de auditoria: preco por metro quadrado.
df_auditoria = df.copy()
df_auditoria["preco_m2"] = df_auditoria["preco"] / df_auditoria["area_util"]

ids_removidos_preco_m2 = [6004, 4316]
df = df[~df["Id"].isin(ids_removidos_preco_m2)].copy()

# Comparacao por bairro. A auditoria e recriada a partir do df ja limpo.
df_auditoria = df.copy()
df_auditoria["preco_m2"] = df_auditoria["preco"] / df_auditoria["area_util"]

mediana_m2_bairro = df_auditoria.groupby("bairro")["preco_m2"].median()
df_auditoria["mediana_m2_bairro"] = df_auditoria["bairro"].map(mediana_m2_bairro)
df_auditoria["razao_m2_bairro"] = (
    df_auditoria["preco_m2"] / df_auditoria["mediana_m2_bairro"]
)

qtd_bairro = df_auditoria["bairro"].value_counts()
df_auditoria["qtd_imoveis_bairro"] = df_auditoria["bairro"].map(qtd_bairro)

suspeitos = df_auditoria[
    (df_auditoria["qtd_imoveis_bairro"] >= 10)
    & (
        (df_auditoria["razao_m2_bairro"] < 0.3)
        | (df_auditoria["razao_m2_bairro"] > 3.0)
    )
]

ids_removidos_preco_m2_bairro = [
    3553,
    4820,
    3604,
    2101,
    4220,
    6361,
    3937,
    3984,
    4075,
    2865,
    3998,
    3536,
    4708,
    3586,
    5540,
    3724,
    2705,
    4808,
    4280,
    2749,
]
df = df[~df["Id"].isin(ids_removidos_preco_m2_bairro)].copy()

# Base que seguira para criacao de features do modelo.



#Features para modelo
def criar_features_modelo(df):
    df_modelo = df.copy()

    df_modelo["area_total"] = df_modelo["area_util"]+ df_modelo["area_extra"]

    cols_comodidades = [
    "churrasqueira",
    "estacionamento",
    "piscina",
    "playground",
    "quadra",
    "s_festas",
    "s_jogos",
    "s_ginastica",
    "sauna",
    "vista_mar"
    ]

    df_modelo["n_comodidades"] = df_modelo[cols_comodidades].sum(axis=1)

    df_modelo["area_por_quarto"] = df_modelo["area_util"] / df_modelo["quartos"]

    df_modelo["tem_suite"] = (df_modelo["suites"] > 0).astype(int)

    df_modelo["tem_vaga"] = (df_modelo["vagas"] > 0).astype(int)

    df_modelo["tem_area_extra"] = (df_modelo["area_extra"] > 0).astype(int)
        
    df_modelo = df_modelo.drop(columns=["diferenciais"])

    df_modelo["tipo"] = df_modelo["tipo"].replace(
        ["Loft", "Quitinete"], 
        "Outro"
    )

    df_modelo["vendedor_imobiliaria"] = (
        df_modelo["tipo_vendedor"] == "Imobiliaria"
    ).astype(int)

    df_modelo = df_modelo.drop(columns=["tipo_vendedor"])

    df_modelo = pd.get_dummies(
        df_modelo,
        columns=["tipo", "bairro"],
        dtype=int,
    )
    
    return df_modelo


def selecionar_bairros_frequentes(df, minimo_imoveis=None):
    contagem_bairros = df["bairro"].value_counts()

    if minimo_imoveis is None:
        return set(contagem_bairros.index)

    return set(contagem_bairros[contagem_bairros >= minimo_imoveis].index)


def criar_features_modelo_bairro_categorico(df, bairros_mantidos):
    df_modelo = df.copy()

    df_modelo["area_total"] = df_modelo["area_util"] + df_modelo["area_extra"]

    cols_comodidades = [
        "churrasqueira",
        "estacionamento",
        "piscina",
        "playground",
        "quadra",
        "s_festas",
        "s_jogos",
        "s_ginastica",
        "sauna",
        "vista_mar",
    ]

    df_modelo["n_comodidades"] = df_modelo[cols_comodidades].sum(axis=1)
    df_modelo["area_por_quarto"] = df_modelo["area_util"] / df_modelo["quartos"]
    df_modelo["tem_suite"] = (df_modelo["suites"] > 0).astype(int)
    df_modelo["tem_vaga"] = (df_modelo["vagas"] > 0).astype(int)
    df_modelo["tem_area_extra"] = (df_modelo["area_extra"] > 0).astype(int)

    df_modelo = df_modelo.drop(columns=["diferenciais"])

    df_modelo["tipo"] = df_modelo["tipo"].replace(
        ["Loft", "Quitinete"],
        "Outro",
    )
    df_modelo["tipo"] = pd.Categorical(
        df_modelo["tipo"],
        categories=["Apartamento", "Casa", "Outro"],
    )

    df_modelo["vendedor_imobiliaria"] = (
        df_modelo["tipo_vendedor"] == "Imobiliaria"
    ).astype(int)
    df_modelo = df_modelo.drop(columns=["tipo_vendedor"])

    df_modelo = pd.get_dummies(
        df_modelo,
        columns=["tipo"],
        dtype=int,
    )

    bairros_mantidos = set(bairros_mantidos)
    df_modelo["bairro"] = df_modelo["bairro"].where(
        df_modelo["bairro"].isin(bairros_mantidos),
        "Outros",
    )

    categorias_bairro = sorted(bairros_mantidos) + ["Outros"]
    df_modelo["bairro"] = pd.Categorical(
        df_modelo["bairro"],
        categories=categorias_bairro,
    )

    return df_modelo


df_modelo = criar_features_modelo(df)

# Id identifica cada linha, mas nao e uma caracteristica do imovel.
ids = df_modelo["Id"].copy()

# X e y so devem ser separados depois das limpezas e transformacoes.
y = df_modelo["preco"].copy()
x = df_modelo.drop(columns=["Id", "preco"])

