import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from pre_processamento import (
    criar_features_modelo_bairro_categorico,
    df as df_limpo,
    selecionar_bairros_frequentes,
)


def criar_modelo():
    return LGBMRegressor(
        objective="regression",
        n_estimators=800,
        learning_rate=0.03,
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


def calcular_rmspe(y_real, y_previsto):
    erros_percentuais = (y_real - y_previsto) / y_real
    return np.sqrt(np.mean(erros_percentuais**2))


def preparar_treino_validacao(df_treino, df_validacao):
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

    return (
        treino_modelo.drop(columns=["Id", "preco"]),
        treino_modelo["preco"],
        validacao_modelo.drop(columns=["Id", "preco"]),
        validacao_modelo["preco"],
    )


def treinar_prever(df_treino, df_validacao):
    x_treino, y_treino, x_validacao, y_validacao = preparar_treino_validacao(
        df_treino,
        df_validacao,
    )
    modelo = criar_modelo()
    modelo.fit(
        x_treino,
        np.log1p(y_treino),
        categorical_feature=["bairro"],
    )
    previsoes = np.expm1(modelo.predict(x_validacao))
    return y_validacao, previsoes


raiz = os.path.join(os.path.dirname(__file__), "..")
df_original = pd.read_csv(
    os.path.join(raiz, "data", "conjunto_de_treinamento (5).csv")
)

ids_erros_incontestaveis = [5910, 2405, 4568, 6004, 6654, 6383]
df_quase_completo = df_original[
    ~df_original["Id"].isin(ids_erros_incontestaveis)
].copy()
df_recuperado = df_quase_completo[
    ~df_quase_completo["Id"].isin(df_limpo["Id"])
].copy()

assert len(df_recuperado) == 23

validacao_cruzada = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

resultados_folds = []

for numero_fold, (indices_treino, indices_validacao) in enumerate(
    validacao_cruzada.split(df_limpo),
    start=1,
):
    treino_limpo = df_limpo.iloc[indices_treino]
    validacao_comum = df_limpo.iloc[indices_validacao]
    treino_com_recuperados = pd.concat(
        [treino_limpo, df_recuperado],
        ignore_index=True,
    )

    y_validacao, previsoes_limpas = treinar_prever(
        treino_limpo,
        validacao_comum,
    )
    _, previsoes_com_recuperados = treinar_prever(
        treino_com_recuperados,
        validacao_comum,
    )

    rmspe_limpo = calcular_rmspe(y_validacao, previsoes_limpas)
    rmspe_recuperados = calcular_rmspe(y_validacao, previsoes_com_recuperados)
    resultados_folds.append(
        {
            "fold": numero_fold,
            "rmspe_treino_limpo": rmspe_limpo,
            "rmspe_treino_com_recuperados": rmspe_recuperados,
            "diferenca": rmspe_recuperados - rmspe_limpo,
        }
    )
    print(
        f"Fold {numero_fold}: limpo={rmspe_limpo * 100:.2f}% | "
        f"com recuperados={rmspe_recuperados * 100:.2f}%"
    )

resultados_folds = pd.DataFrame(resultados_folds)

# Mede separadamente o quanto as linhas recuperadas fogem do padrao aprendido.
y_recuperado, previsoes_recuperados = treinar_prever(df_limpo, df_recuperado)
auditoria_recuperados = df_recuperado[
    ["Id", "preco", "area_util", "area_extra", "quartos", "bairro"]
].copy()
auditoria_recuperados["previsao_modelo_limpo"] = previsoes_recuperados
auditoria_recuperados["erro_percentual"] = (
    auditoria_recuperados["preco"] - auditoria_recuperados["previsao_modelo_limpo"]
) / auditoria_recuperados["preco"]
auditoria_recuperados["erro_percentual_abs"] = auditoria_recuperados[
    "erro_percentual"
].abs()
auditoria_recuperados["erro_percentual_quadrado"] = (
    auditoria_recuperados["erro_percentual"] ** 2
)
auditoria_recuperados = auditoria_recuperados.sort_values(
    "erro_percentual_quadrado",
    ascending=False,
)

pasta_resultados = os.path.join(raiz, "resultados")
os.makedirs(pasta_resultados, exist_ok=True)
resultados_folds.to_csv(
    os.path.join(pasta_resultados, "comparacao_validacao_mesmas_linhas.csv"),
    index=False,
)
auditoria_recuperados.to_csv(
    os.path.join(pasta_resultados, "auditoria_23_linhas_recuperadas.csv"),
    index=False,
)

media_limpa = resultados_folds["rmspe_treino_limpo"].mean()
media_recuperados = resultados_folds["rmspe_treino_com_recuperados"].mean()
rmspe_23 = calcular_rmspe(y_recuperado, previsoes_recuperados)

print("\nComparacao sobre as mesmas linhas de validacao:")
print(f"Treino limpo: {media_limpa * 100:.2f}%")
print(f"Treino com 23 recuperados: {media_recuperados * 100:.2f}%")
print(f"Diferenca: {(media_recuperados - media_limpa) * 100:.2f} p.p.")
print(
    "Folds em que adicionar recuperados melhorou: "
    f"{(resultados_folds['diferenca'] < 0).sum()}/5"
)
print(f"RMSPE isolado das 23 linhas recuperadas: {rmspe_23 * 100:.2f}%")
print("\nLinhas recuperadas com maior erro:")
print(
    auditoria_recuperados[
        ["Id", "preco", "previsao_modelo_limpo", "erro_percentual_abs"]
    ].head(10).to_string(index=False)
)

# Holdouts por ID: aproximam o fato de o teste ocupar o bloco anterior ao treino.
resultados_blocos = []

for proporcao_validacao in [0.10, 0.15, 0.20, 0.25, 0.30]:
    quantidade_validacao_bloco = round(
        len(df_quase_completo) * proporcao_validacao
    )
    ids_validacao_bloco = set(
        df_quase_completo.nsmallest(quantidade_validacao_bloco, "Id")["Id"]
    )
    validacao_bloco = df_quase_completo[
        df_quase_completo["Id"].isin(ids_validacao_bloco)
    ]
    treino_bloco_limpo = df_limpo[
        ~df_limpo["Id"].isin(ids_validacao_bloco)
    ]
    treino_bloco_recuperado = df_quase_completo[
        ~df_quase_completo["Id"].isin(ids_validacao_bloco)
    ]

    y_bloco, previsoes_bloco_limpo = treinar_prever(
        treino_bloco_limpo,
        validacao_bloco,
    )
    _, previsoes_bloco_recuperado = treinar_prever(
        treino_bloco_recuperado,
        validacao_bloco,
    )

    rmspe_bloco_limpo = calcular_rmspe(y_bloco, previsoes_bloco_limpo)
    rmspe_bloco_recuperado = calcular_rmspe(y_bloco, previsoes_bloco_recuperado)
    resultados_blocos.append(
        {
            "proporcao_validacao": proporcao_validacao,
            "id_minimo": validacao_bloco["Id"].min(),
            "id_maximo": validacao_bloco["Id"].max(),
            "linhas_validacao": len(validacao_bloco),
            "rmspe_treino_limpo": rmspe_bloco_limpo,
            "rmspe_treino_com_recuperados": rmspe_bloco_recuperado,
            "diferenca": rmspe_bloco_recuperado - rmspe_bloco_limpo,
        }
    )

resultados_blocos = pd.DataFrame(resultados_blocos)
resultados_blocos.to_csv(
    os.path.join(pasta_resultados, "comparacao_holdout_blocos_id.csv"),
    index=False,
)

print("\nHoldouts com os menores IDs do treino:")
print(
    resultados_blocos.assign(
        rmspe_limpo_pct=resultados_blocos["rmspe_treino_limpo"] * 100,
        rmspe_recuperados_pct=(
            resultados_blocos["rmspe_treino_com_recuperados"] * 100
        ),
        diferenca_pp=resultados_blocos["diferenca"] * 100,
    )[
        [
            "proporcao_validacao",
            "id_minimo",
            "id_maximo",
            "linhas_validacao",
            "rmspe_limpo_pct",
            "rmspe_recuperados_pct",
            "diferenca_pp",
        ]
    ].to_string(index=False)
)
