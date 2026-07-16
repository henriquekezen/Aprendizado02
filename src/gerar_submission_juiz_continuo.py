"""Treina o juiz continuo vencedor e gera sua unica submission."""

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
from testar_juiz_continuo import (
    aplicar_mapeamento,
    construir_alvos,
)


VARIANTE = "alvo_projetado_01"
ARQUIVO_SUBMISSION = "submission_juiz_continuo_rmspe.csv"


def main():
    raiz, df = carregar_treino()
    df_teste = carregar_teste(raiz)
    _, _, _, meta, x_oof = carregar_bases_oof(raiz, df)
    alvos, pesos = construir_alvos(meta)

    pasta_resultados = os.path.join(raiz, "resultados")
    with open(
        os.path.join(pasta_resultados, "juiz_continuo_decisao.json"),
        encoding="utf-8",
    ) as arquivo:
        decisao = json.load(arquivo)
    assert decisao["elegivel_para_submission"]
    assert decisao["variante_vencedora"] == VARIANTE

    with open(
        os.path.join(
            pasta_resultados, "juiz_continuo_parametros_finais.json"
        ),
        encoding="utf-8",
    ) as arquivo:
        parametros = json.load(arquivo)[VARIANTE]

    componentes = pd.read_csv(
        os.path.join(pasta_resultados, "juiz_especialista_previsoes_teste.csv")
    )
    cat_teste = pd.read_csv(
        os.path.join(
            pasta_resultados, "catboost_previsoes_teste_alphas_corr30.csv"
        )
    )
    for base in [componentes, cat_teste]:
        assert np.array_equal(df_teste["Id"].to_numpy(), base["Id"].to_numpy())

    x_teste = preparar_features_com_previsoes(
        df_teste.assign(preco=np.nan),
        componentes["generalista"],
        componentes["especialista"],
        componentes["catboost_global"],
        componentes["blend_arvores"],
        cat_teste["previsao_correspondencia"].notna(),
    )
    modelo = criar_regressor()
    inicio = time.perf_counter()
    modelo.fit(
        x_oof,
        alvos[VARIANTE],
        cat_features=COLUNAS_CATEGORICAS,
        sample_weight=pesos,
    )
    tempo = time.perf_counter() - inicio
    score = modelo.predict(x_teste)
    peso = aplicar_mapeamento(
        score,
        parametros["intercepto"],
        parametros["inclinacao"],
    )
    geral = componentes["generalista"].to_numpy()
    especialista = componentes["especialista"].to_numpy()
    pred = geral + peso * (especialista - geral)

    oof = pd.read_csv(os.path.join(pasta_resultados, "juiz_continuo_oof.csv"))
    score_oof = oof[f"score_{VARIANTE}"].to_numpy()
    peso_oof_final = aplicar_mapeamento(
        score_oof,
        parametros["intercepto"],
        parametros["inclinacao"],
    )
    pred_oof_final = (
        oof["generalista"].to_numpy()
        + peso_oof_final
        * (
            oof["especialista"].to_numpy()
            - oof["generalista"].to_numpy()
        )
    )
    rmspe_oof_mapeamento_final = rmspe(oof["preco"], pred_oof_final)

    modelo_resposta = pd.read_csv(
        os.path.join(raiz, "data", "exemplo_arquivo_respostas.csv")
    )
    assert np.array_equal(
        modelo_resposta["Id"].to_numpy(), df_teste["Id"].to_numpy()
    )
    submission = modelo_resposta.copy()
    submission["preco"] = pred.round(2)
    caminho_submission = os.path.join(
        raiz, "submissions", ARQUIVO_SUBMISSION
    )
    submission.to_csv(caminho_submission, index=False)

    previsoes = componentes[
        ["Id", "generalista", "especialista", "catboost_global", "blend_arvores"]
    ].copy()
    previsoes["score_juiz_continuo"] = score
    previsoes["peso_juiz_continuo"] = peso
    previsoes["previsao_juiz_continuo"] = pred
    caminho_previsoes = os.path.join(
        pasta_resultados, "juiz_continuo_previsoes_teste.csv"
    )
    previsoes.to_csv(caminho_previsoes, index=False)

    assert submission.shape == (2000, 2)
    assert submission["Id"].is_unique
    assert not submission.isna().any().any()
    assert (submission["preco"] > 0).all()
    assert np.max(np.abs(submission["preco"].to_numpy() - pred)) <= 0.005001

    relatorio = {
        "variante": VARIANTE,
        "arquivo_submission": ARQUIVO_SUBMISSION,
        "intercepto": parametros["intercepto"],
        "inclinacao": parametros["inclinacao"],
        "rmspe_crossfit": decisao["rmspe_crossfit"],
        "rmspe_holdout": decisao["rmspe_holdout"],
        "rmspe_oof_mapeamento_final": rmspe_oof_mapeamento_final,
        "peso_teste_media": float(peso.mean()),
        "peso_teste_mediana": float(np.median(peso)),
        "peso_teste_p90": float(np.quantile(peso, 0.90)),
        "fracao_peso_zero_teste": float(np.mean(peso == 0)),
        "fracao_peso_um_teste": float(np.mean(peso == 1)),
        "tempo_treino_final_segundos": tempo,
    }
    with open(
        os.path.join(pasta_resultados, "juiz_continuo_submission_resumo.json"),
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(relatorio, arquivo, ensure_ascii=False, indent=2)

    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    print(f"Submission: {caminho_submission}")
    print(f"Previsoes: {caminho_previsoes}")


if __name__ == "__main__":
    main()
