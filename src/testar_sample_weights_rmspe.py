"""
Testa o efeito de sample weights (inversamente proporcionais ao preco)
no treino dos modelos LightGBM e XGBoost.

A ideia: RMSPE penaliza erros percentuais, entao imoveis baratos pesam
mais na metrica. Usando sample_weight = 1/preco, forcamos o modelo a
"se esforcar mais" nos imoveis baratos, alinhando a loss de treino (MSE)
com a metrica de avaliacao (RMSPE).

Compara: sem pesos (atual) vs com pesos para cada modelo e blend.
"""

import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from pre_processamento import (
    criar_features_modelo_bairro_categorico,
    selecionar_bairros_frequentes,
)


COLUNAS_CORRESPONDENCIA = [
    "tipo",
    "bairro",
    "area_util",
    "area_extra",
    "quartos",
    "suites",
    "vagas",
]


def calcular_rmspe(y_real, y_previsto):
    erros_percentuais = (y_real - y_previsto) / y_real
    return np.sqrt(np.mean(erros_percentuais**2))


def corrigir_dados(df, corrigir_alvo=False):
    df = df.copy()
    df["area_util"] = df["area_util"].astype(float)
    mascara_area = (
        (df["tipo"] == "Apartamento")
        & (df["quartos"] > 0)
        & ((df["area_util"] / df["quartos"]) > 200)
    )
    df.loc[mascara_area, "area_util"] /= 10

    if corrigir_alvo:
        df.loc[df["Id"].isin([2749, 4316]), "preco"] /= 10
        df.loc[df["Id"] == 6383, "vagas"] = 3

    return df


def preparar_features(df_treino, df_validacao):
    bairros_mantidos = selecionar_bairros_frequentes(
        df_treino,
        minimo_imoveis=10,
    )
    treino_modelo = criar_features_modelo_bairro_categorico(
        df_treino,
        bairros_mantidos,
    )
    validacao_modelo = criar_features_modelo_bairro_categorico(
        df_validacao,
        bairros_mantidos,
    )
    return treino_modelo, validacao_modelo


def criar_xgboost():
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=4,
        random_state=42,
        n_jobs=-1,
    )


def criar_lightgbm():
    return LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.06,
        num_leaves=15,
        max_depth=5,
        min_child_samples=20,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        cat_smooth=10.0,
        cat_l2=10.0,
        min_data_per_group=100,
        max_cat_threshold=32,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def calcular_pesos(y_preco):
    """Calcula sample weights inversamente proporcionais ao preco.

    Imoveis baratos recebem peso maior, alinhando a loss de treino
    com o RMSPE que penaliza erros percentuais.
    Os pesos sao normalizados para ter media 1.
    """
    pesos = 1.0 / y_preco.values
    pesos = pesos / pesos.mean()
    return pesos


def prever_modelos(df_treino, df_validacao, usar_pesos=False):
    treino_modelo, validacao_modelo = preparar_features(
        df_treino,
        df_validacao,
    )
    x_treino = treino_modelo.drop(columns=["Id", "preco"])
    y_treino_log = np.log1p(treino_modelo["preco"])
    x_validacao = validacao_modelo.drop(
        columns=["Id", "preco"],
        errors="ignore",
    )

    pesos = calcular_pesos(treino_modelo["preco"]) if usar_pesos else None

    modelo_xgb = criar_xgboost()
    modelo_xgb.fit(x_treino, y_treino_log, sample_weight=pesos)
    previsoes_xgb = np.expm1(modelo_xgb.predict(x_validacao))

    modelo_lgbm = criar_lightgbm()
    modelo_lgbm.fit(
        x_treino,
        y_treino_log,
        sample_weight=pesos,
        categorical_feature=["bairro"],
    )
    previsoes_lgbm = np.expm1(modelo_lgbm.predict(x_validacao))

    return previsoes_xgb, previsoes_lgbm


def prever_correspondencias(df_treino, df_validacao):
    tabela = (
        df_treino.groupby(COLUNAS_CORRESPONDENCIA, dropna=False)["preco"]
        .median()
        .rename("preco_correspondencia")
        .reset_index()
    )
    validacao_ordenada = df_validacao.reset_index(drop=True).reset_index(
        names="ordem_original"
    )
    return (
        validacao_ordenada.merge(
            tabela,
            on=COLUNAS_CORRESPONDENCIA,
            how="left",
        )
        .sort_values("ordem_original")["preco_correspondencia"]
        .to_numpy()
    )


def aplicar_correspondencias(previsoes_modelo, previsoes_correspondencias):
    previsoes = previsoes_modelo.copy()
    mascara = ~np.isnan(previsoes_correspondencias)
    previsoes[mascara] = (
        0.70 * previsoes_modelo[mascara]
        + 0.30 * previsoes_correspondencias[mascara]
    )
    return previsoes


def calcular_todas_previsoes(
    previsoes_xgb,
    previsoes_lgbm,
    previsoes_correspondencias,
):
    previsoes_blend = 0.50 * previsoes_xgb + 0.50 * previsoes_lgbm
    return {
        "xgboost": aplicar_correspondencias(
            previsoes_xgb,
            previsoes_correspondencias,
        ),
        "lightgbm": aplicar_correspondencias(
            previsoes_lgbm,
            previsoes_correspondencias,
        ),
        "blend_50": aplicar_correspondencias(
            previsoes_blend,
            previsoes_correspondencias,
        ),
    }


# Carregamento e correcao dos dados.
raiz = os.path.join(os.path.dirname(__file__), "..")
df_original = pd.read_csv(
    os.path.join(raiz, "data", "conjunto_de_treinamento (5).csv")
)
df_teste_original = pd.read_csv(
    os.path.join(raiz, "data", "conjunto_de_teste (3).csv")
)
df_base = df_original[
    ~df_original["Id"].isin([5910, 2405, 4568, 6004, 6654])
].copy()
df_corrigido = corrigir_dados(df_base, corrigir_alvo=True).reset_index(drop=True)
df_teste_corrigido = corrigir_dados(
    df_teste_original,
    corrigir_alvo=False,
).reset_index(drop=True)

# Testar com e sem sample weights.
variantes = [
    ("sem_pesos", False),
    ("com_pesos", True),
]

todos_resultados = []

for nome_variante, usar_pesos in variantes:
    print(f"\n{'='*60}")
    print(f"Variante: {nome_variante}")
    print(f"{'='*60}")

    # KFold cross-validation.
    validacao_cruzada = KFold(n_splits=5, shuffle=True, random_state=42)
    resultados_kfold = {
        "xgboost": [],
        "lightgbm": [],
        "blend_50": [],
    }

    for numero_fold, (indices_treino, indices_validacao) in enumerate(
        validacao_cruzada.split(df_corrigido),
        start=1,
    ):
        df_treino = df_corrigido.iloc[indices_treino]
        df_validacao = df_corrigido.iloc[indices_validacao]
        previsoes_xgb, previsoes_lgbm = prever_modelos(
            df_treino,
            df_validacao,
            usar_pesos=usar_pesos,
        )
        previsoes_correspondencias = prever_correspondencias(
            df_treino,
            df_validacao,
        )
        previsoes = calcular_todas_previsoes(
            previsoes_xgb,
            previsoes_lgbm,
            previsoes_correspondencias,
        )

        for nome_modelo, valores in previsoes.items():
            rmspe = calcular_rmspe(df_validacao["preco"], valores)
            resultados_kfold[nome_modelo].append(rmspe)
        print(f"  Fold {numero_fold}/5 concluido")

    # Holdout com os 20% menores IDs.
    quantidade_bloco = round(len(df_corrigido) * 0.2)
    ids_bloco = set(df_corrigido.nsmallest(quantidade_bloco, "Id")["Id"])
    df_treino_bloco = df_corrigido[~df_corrigido["Id"].isin(ids_bloco)]
    df_validacao_bloco = df_corrigido[df_corrigido["Id"].isin(ids_bloco)]
    previsoes_xgb_bloco, previsoes_lgbm_bloco = prever_modelos(
        df_treino_bloco,
        df_validacao_bloco,
        usar_pesos=usar_pesos,
    )
    previsoes_correspondencias_bloco = prever_correspondencias(
        df_treino_bloco,
        df_validacao_bloco,
    )
    previsoes_bloco = calcular_todas_previsoes(
        previsoes_xgb_bloco,
        previsoes_lgbm_bloco,
        previsoes_correspondencias_bloco,
    )
    resultados_bloco = {
        nome: calcular_rmspe(df_validacao_bloco["preco"], valores)
        for nome, valores in previsoes_bloco.items()
    }

    for nome_modelo, valores_fold in resultados_kfold.items():
        todos_resultados.append(
            {
                "variante": nome_variante,
                "modelo": nome_modelo,
                "rmspe_kfold": np.mean(valores_fold),
                "desvio_kfold": np.std(valores_fold),
                "rmspe_bloco_id": resultados_bloco[nome_modelo],
                **{
                    f"rmspe_fold_{i}": v
                    for i, v in enumerate(valores_fold, start=1)
                },
            }
        )

# Gerar submissions apenas se os pesos melhorarem.
resultados_df = pd.DataFrame(todos_resultados)
resultados_df = resultados_df.sort_values(["modelo", "variante"])

pasta_resultados = os.path.join(raiz, "resultados")
os.makedirs(pasta_resultados, exist_ok=True)
caminho_resultados = os.path.join(
    pasta_resultados,
    "comparacao_sample_weights_rmspe.csv",
)
resultados_df.to_csv(caminho_resultados, index=False)

print(f"\n{'='*60}")
print("COMPARACAO: SEM PESOS vs COM PESOS")
print(f"{'='*60}")
resumo = resultados_df[
    ["variante", "modelo", "rmspe_kfold", "desvio_kfold", "rmspe_bloco_id"]
]
print(resumo.to_string(index=False))

# Mostrar diferenca percentual.
print(f"\n{'='*60}")
print("DIFERENCA (com_pesos - sem_pesos)")
print(f"{'='*60}")
for modelo in ["xgboost", "lightgbm", "blend_50"]:
    sem = resultados_df[
        (resultados_df["variante"] == "sem_pesos")
        & (resultados_df["modelo"] == modelo)
    ].iloc[0]
    com = resultados_df[
        (resultados_df["variante"] == "com_pesos")
        & (resultados_df["modelo"] == modelo)
    ].iloc[0]
    diff_kfold = com["rmspe_kfold"] - sem["rmspe_kfold"]
    diff_bloco = com["rmspe_bloco_id"] - sem["rmspe_bloco_id"]
    sinal_kf = "+" if diff_kfold > 0 else ""
    sinal_bl = "+" if diff_bloco > 0 else ""
    print(
        f"  {modelo:12s}  KFold: {sinal_kf}{diff_kfold:.4f}"
        f"  Bloco ID: {sinal_bl}{diff_bloco:.4f}"
    )

# Se com_pesos melhorou o blend no bloco_id, gerar submission.
blend_sem = resultados_df[
    (resultados_df["variante"] == "sem_pesos")
    & (resultados_df["modelo"] == "blend_50")
]["rmspe_bloco_id"].iloc[0]
blend_com = resultados_df[
    (resultados_df["variante"] == "com_pesos")
    & (resultados_df["modelo"] == "blend_50")
]["rmspe_bloco_id"].iloc[0]

if blend_com < blend_sem:
    print("\n>>> Sample weights melhoraram o blend! Gerando submission...")

    previsoes_xgb_teste, previsoes_lgbm_teste = prever_modelos(
        df_corrigido,
        df_teste_corrigido,
        usar_pesos=True,
    )
    previsoes_correspondencias_teste = prever_correspondencias(
        df_corrigido,
        df_teste_corrigido,
    )
    previsoes_blend_teste = 0.50 * previsoes_xgb_teste + 0.50 * previsoes_lgbm_teste
    previsoes_finais = aplicar_correspondencias(
        previsoes_blend_teste,
        previsoes_correspondencias_teste,
    )

    modelo_resposta = pd.read_csv(
        os.path.join(raiz, "data", "exemplo_arquivo_respostas.csv")
    )
    submission = modelo_resposta.copy()
    submission["preco"] = previsoes_finais.round(2)

    pasta_saida = os.path.join(raiz, "submissions")
    os.makedirs(pasta_saida, exist_ok=True)
    caminho_saida = os.path.join(
        pasta_saida,
        "submission_blend_xgb_lgbm_sample_weights_corr30.csv",
    )
    submission.to_csv(caminho_saida, index=False)

    assert submission.shape == (2000, 2)
    assert submission["Id"].is_unique
    assert not submission.isna().any().any()
    assert not (submission["preco"] <= 0).any()
    print(f"Submission: {os.path.abspath(caminho_saida)}")
else:
    print("\n>>> Sample weights NAO melhoraram o blend. Nenhuma submission gerada.")

print(f"\nResultados: {os.path.abspath(caminho_resultados)}")
