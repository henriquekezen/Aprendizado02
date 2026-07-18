"""Intensifica em 1,75 o juiz CatBoost/arvores ancorado e gera submission."""

import json
import os

import numpy as np
import pandas as pd

from gerar_submissions_juiz_especialista import rmspe
from testar_catboost_correspondencias import aplicar_correspondencias


INTENSIDADE = 1.75
REFERENCIA_ARVORES = 0.40
PESO_CORRESPONDENCIA = 0.30
ARQUIVO_SUBMISSION = "submission_juiz_componentes_intensidade175.csv"


def intensificar(r_ancorado):
    r_ancorado = np.asarray(r_ancorado, dtype=float)
    return np.clip(
        REFERENCIA_ARVORES
        + INTENSIDADE * (r_ancorado - REFERENCIA_ARVORES),
        0.0,
        1.0,
    )


def combinar(q, especialista, catboost, arvores, r_arvores):
    q = np.asarray(q, dtype=float)
    return q * especialista + (1.0 - q) * (
        (1.0 - r_arvores) * catboost + r_arvores * arvores
    )


def avaliar_oof(raiz):
    caminho = os.path.join(raiz, "resultados", "juiz_componentes_oof.csv")
    oof = pd.read_csv(caminho)
    r = intensificar(oof["r_arvores_ancorada_publico"])
    pred = combinar(
        oof["peso_especialista"].to_numpy(),
        oof["especialista"].to_numpy(),
        oof["catboost_global"].to_numpy(),
        oof["blend_arvores"].to_numpy(),
        r,
    )
    return {
        "rmspe": rmspe(oof["preco"], pred),
        "media_r": float(r.mean()),
        "mediana_r": float(np.median(r)),
        "p10_r": float(np.quantile(r, 0.10)),
        "p90_r": float(np.quantile(r, 0.90)),
        "fracao_limitada": float(np.mean((r == 0.0) | (r == 1.0))),
    }


def avaliar_holdout(raiz):
    pasta = os.path.join(raiz, "resultados")
    holdout = pd.read_csv(os.path.join(pasta, "juiz_componentes_holdout.csv"))
    cat_holdout = pd.read_csv(
        os.path.join(pasta, "catboost_correspondencias_holdout_id.csv")
    )
    assert np.array_equal(
        holdout["Id"].to_numpy(), cat_holdout["Id"].to_numpy()
    )
    cat = aplicar_correspondencias(
        cat_holdout["catboost_alpha_1.50"].to_numpy(),
        cat_holdout["previsao_correspondencia"].to_numpy(),
        PESO_CORRESPONDENCIA,
    )
    arvores = (holdout["generalista"].to_numpy() - 0.60 * cat) / 0.40
    r = intensificar(holdout["r_arvores_ancorada_publico"])
    pred = combinar(
        holdout["peso_alvo_projetado_01"].to_numpy(),
        holdout["especialista"].to_numpy(),
        cat,
        arvores,
        r,
    )
    y = holdout["preco"].to_numpy()
    baratos = y <= 355_000
    return {
        "rmspe": rmspe(y, pred),
        "rmspe_baratos_ate_355k": rmspe(y[baratos], pred[baratos]),
        "rmspe_caros_acima_355k": rmspe(y[~baratos], pred[~baratos]),
        "media_r": float(r.mean()),
        "mediana_r": float(np.median(r)),
        "p10_r": float(np.quantile(r, 0.10)),
        "p90_r": float(np.quantile(r, 0.90)),
        "fracao_limitada": float(np.mean((r == 0.0) | (r == 1.0))),
    }


def gerar_submission(raiz):
    pasta = os.path.join(raiz, "resultados")
    previsoes_base = pd.read_csv(
        os.path.join(pasta, "juiz_componentes_previsoes_teste.csv")
    )
    r = intensificar(previsoes_base["fracao_arvores_no_restante"])
    pred = combinar(
        previsoes_base["peso_especialista"].to_numpy(),
        previsoes_base["especialista"].to_numpy(),
        previsoes_base["catboost_global"].to_numpy(),
        previsoes_base["blend_arvores"].to_numpy(),
        r,
    )
    peso_especialista = previsoes_base["peso_especialista"].to_numpy()
    peso_cat = (1.0 - peso_especialista) * (1.0 - r)
    peso_arvores = (1.0 - peso_especialista) * r
    assert np.allclose(peso_especialista + peso_cat + peso_arvores, 1.0)

    resposta = pd.read_csv(
        os.path.join(raiz, "data", "exemplo_arquivo_respostas.csv")
    )
    assert np.array_equal(
        resposta["Id"].to_numpy(), previsoes_base["Id"].to_numpy()
    )
    submission = resposta.copy()
    submission["preco"] = pred.round(2)
    caminho_submission = os.path.join(
        raiz, "submissions", ARQUIVO_SUBMISSION
    )
    submission.to_csv(caminho_submission, index=False)

    detalhe = previsoes_base[
        ["Id", "especialista", "catboost_global", "blend_arvores"]
    ].copy()
    detalhe["peso_especialista"] = peso_especialista
    detalhe["fracao_arvores_ancorada"] = previsoes_base[
        "fracao_arvores_no_restante"
    ]
    detalhe["fracao_arvores_intensidade175"] = r
    detalhe["peso_catboost_final"] = peso_cat
    detalhe["peso_arvores_final"] = peso_arvores
    detalhe["previsao_final"] = pred
    detalhe.to_csv(
        os.path.join(
            pasta, "juiz_componentes_intensidade175_previsoes_teste.csv"
        ),
        index=False,
    )

    assert submission.shape == (2000, 2)
    assert submission["Id"].is_unique
    assert not submission.isna().any().any()
    assert (submission["preco"] > 0).all()
    assert np.max(np.abs(submission["preco"].to_numpy() - pred)) <= 0.005001
    return caminho_submission, {
        "media_r": float(r.mean()),
        "mediana_r": float(np.median(r)),
        "p10_r": float(np.quantile(r, 0.10)),
        "p90_r": float(np.quantile(r, 0.90)),
        "fracao_limitada": float(np.mean((r == 0.0) | (r == 1.0))),
        "peso_especialista_medio": float(peso_especialista.mean()),
        "peso_catboost_medio": float(peso_cat.mean()),
        "peso_arvores_medio": float(peso_arvores.mean()),
        "alteracao_absoluta_media_vs_ancorada": float(
            np.mean(np.abs(pred - previsoes_base["previsao_final"]))
        ),
    }


def main():
    raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    oof = avaliar_oof(raiz)
    holdout = avaliar_holdout(raiz)
    caminho, teste = gerar_submission(raiz)
    resumo = {
        "intensidade": INTENSIDADE,
        "referencia_arvores": REFERENCIA_ARVORES,
        "score_publico_origem": 0.2119,
        "arquivo_submission": ARQUIVO_SUBMISSION,
        "oof": oof,
        "holdout": holdout,
        "teste": teste,
    }
    with open(
        os.path.join(
            raiz,
            "resultados",
            "juiz_componentes_intensidade175_resumo.json",
        ),
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print(f"Submission: {caminho}")


if __name__ == "__main__":
    main()
