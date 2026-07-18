"""Treina o juiz CatBoost/arvores ancorado e gera uma submission."""

import json
import os
import time

import numpy as np
import pandas as pd

from comparar_juizes_especialista import (
    criar_regressor,
    preparar_features_com_previsoes,
)
from gerar_submissions_juiz_especialista import (
    carregar_bases_oof,
    carregar_teste,
    rmspe,
)
from testar_catboost_alphas import COLUNAS_CATEGORICAS, carregar_treino
from testar_juiz_componentes import (
    ANCORA_ARVORES_PUBLICA,
    aplicar_mapeamento,
    enriquecer_features,
    montar_estado,
    predizer,
)


ESTRATEGIA = "ancorada_publico"
ARQUIVO_SUBMISSION = "submission_juiz_componentes_ancorado.csv"


def main():
    raiz, df = carregar_treino()
    df_teste = carregar_teste(raiz)
    pasta = os.path.join(raiz, "resultados")
    geral, especialista, cat, meta_binario, x_oof = carregar_bases_oof(raiz, df)
    meta = pd.read_csv(os.path.join(pasta, "juiz_continuo_oof.csv"))
    estado_oof = montar_estado(meta, geral)
    x_oof = enriquecer_features(x_oof, estado_oof)

    with open(
        os.path.join(pasta, "juiz_componentes_decisao.json"),
        encoding="utf-8",
    ) as arquivo:
        decisao = json.load(arquivo)
    assert decisao["gerar_submission"]
    assert decisao["estrategia_escolhida"] == ESTRATEGIA
    with open(
        os.path.join(pasta, "juiz_componentes_parametros_finais.json"),
        encoding="utf-8",
    ) as arquivo:
        parametros = json.load(arquivo)[ESTRATEGIA]

    componentes = pd.read_csv(
        os.path.join(pasta, "juiz_continuo_previsoes_teste.csv")
    )
    cat_teste = pd.read_csv(
        os.path.join(pasta, "catboost_previsoes_teste_alphas_corr30.csv")
    )
    for base in [componentes, cat_teste]:
        assert np.array_equal(df_teste["Id"].to_numpy(), base["Id"].to_numpy())

    estado_teste = {
        "especialista": componentes["especialista"].to_numpy(),
        "q": componentes["peso_juiz_continuo"].to_numpy(),
        "cat": componentes["catboost_global"].to_numpy(),
        "arvores": componentes["blend_arvores"].to_numpy(),
    }
    estado_teste["base_cat"] = (
        estado_teste["q"] * estado_teste["especialista"]
        + (1.0 - estado_teste["q"]) * estado_teste["cat"]
    )
    estado_teste["direcao"] = (1.0 - estado_teste["q"]) * (
        estado_teste["arvores"] - estado_teste["cat"]
    )
    x_teste = preparar_features_com_previsoes(
        df_teste.assign(preco=np.nan),
        componentes["generalista"],
        componentes["especialista"],
        componentes["catboost_global"],
        componentes["blend_arvores"],
        cat_teste["previsao_correspondencia"].notna(),
    )
    x_teste = enriquecer_features(x_teste, estado_teste)

    modelo = criar_regressor()
    inicio = time.perf_counter()
    modelo.fit(
        x_oof,
        estado_oof["alvo"],
        cat_features=COLUNAS_CATEGORICAS,
        sample_weight=estado_oof["peso"],
    )
    tempo = time.perf_counter() - inicio
    score = modelo.predict(x_teste)
    r_arvores = aplicar_mapeamento(
        score,
        parametros["intercepto"],
        parametros["inclinacao"],
    )
    pred = predizer(estado_teste, r_arvores)

    oof = pd.read_csv(os.path.join(pasta, "juiz_componentes_oof.csv"))
    r_oof_final = aplicar_mapeamento(
        oof["score_juiz_componentes"].to_numpy(),
        parametros["intercepto"],
        parametros["inclinacao"],
    )
    pred_oof_final = predizer(estado_oof, r_oof_final)
    rmspe_oof_final = rmspe(estado_oof["y"], pred_oof_final)

    peso_especialista = estado_teste["q"]
    peso_cat = (1.0 - peso_especialista) * (1.0 - r_arvores)
    peso_arvores = (1.0 - peso_especialista) * r_arvores
    assert np.allclose(peso_especialista + peso_cat + peso_arvores, 1.0)
    assert (peso_especialista >= 0).all()
    assert (peso_cat >= 0).all()
    assert (peso_arvores >= 0).all()

    resposta = pd.read_csv(
        os.path.join(raiz, "data", "exemplo_arquivo_respostas.csv")
    )
    assert np.array_equal(resposta["Id"].to_numpy(), df_teste["Id"].to_numpy())
    submission = resposta.copy()
    submission["preco"] = pred.round(2)
    caminho_submission = os.path.join(
        raiz, "submissions", ARQUIVO_SUBMISSION
    )
    submission.to_csv(caminho_submission, index=False)

    previsoes = componentes[
        ["Id", "generalista", "especialista", "catboost_global", "blend_arvores"]
    ].copy()
    previsoes["peso_especialista"] = peso_especialista
    previsoes["score_juiz_componentes"] = score
    previsoes["fracao_arvores_no_restante"] = r_arvores
    previsoes["peso_catboost_final"] = peso_cat
    previsoes["peso_arvores_final"] = peso_arvores
    previsoes["previsao_final"] = pred
    caminho_previsoes = os.path.join(
        pasta, "juiz_componentes_previsoes_teste.csv"
    )
    previsoes.to_csv(caminho_previsoes, index=False)

    assert submission.shape == (2000, 2)
    assert submission["Id"].is_unique
    assert not submission.isna().any().any()
    assert (submission["preco"] > 0).all()
    assert np.max(np.abs(submission["preco"].to_numpy() - pred)) <= 0.005001

    resumo_cf = pd.read_csv(
        os.path.join(pasta, "juiz_componentes_resumo.csv")
    ).set_index("estrategia").loc[ESTRATEGIA]
    resumo_h = pd.read_csv(
        os.path.join(pasta, "juiz_componentes_holdout_resumo.csv")
    ).set_index("estrategia").loc[ESTRATEGIA]
    relatorio = {
        "estrategia": ESTRATEGIA,
        "arquivo_submission": ARQUIVO_SUBMISSION,
        "score_publico_referencia": 0.2130,
        "ancora_arvores_publica": ANCORA_ARVORES_PUBLICA,
        "intercepto": parametros["intercepto"],
        "inclinacao": parametros["inclinacao"],
        "rmspe_crossfit": float(resumo_cf["rmspe_crossfit"]),
        "rmspe_holdout": float(resumo_h["rmspe_holdout"]),
        "rmspe_oof_mapeamento_final": rmspe_oof_final,
        "fracao_arvores_restante_media_teste": float(r_arvores.mean()),
        "fracao_arvores_restante_mediana_teste": float(np.median(r_arvores)),
        "peso_especialista_medio_teste": float(peso_especialista.mean()),
        "peso_catboost_medio_teste": float(peso_cat.mean()),
        "peso_arvores_medio_teste": float(peso_arvores.mean()),
        "alteracao_absoluta_media_vs_juiz_continuo": float(
            np.mean(np.abs(pred - componentes["previsao_juiz_continuo"]))
        ),
        "tempo_treino_final_segundos": tempo,
    }
    with open(
        os.path.join(pasta, "juiz_componentes_submission_resumo.json"),
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(relatorio, arquivo, ensure_ascii=False, indent=2)
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    print(f"Submission: {caminho_submission}")
    print(f"Previsoes: {caminho_previsoes}")


if __name__ == "__main__":
    main()
