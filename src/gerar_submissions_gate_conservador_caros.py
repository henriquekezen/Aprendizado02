"""Gera duas ablações do especialista apenas no top 5% do pipeline atual."""

import json
import os

import numpy as np
import pandas as pd

from gerar_submissions_juiz_especialista import rmspe
from testar_gate_conservador_caros import combinar, selecionar_top_por_fold


FRACAO_TOP = 0.05
PESOS = [0.50, 1.00]
NOMES = {
    0.50: "submission_diagnostico_caros_top05_w050.csv",
    1.00: "submission_diagnostico_caros_top05_w100.csv",
}


def bootstrap_pareado(y, base, pred, n=5000):
    rng = np.random.default_rng(42)
    ganhos = np.empty(n)
    for i in range(n):
        indices = rng.integers(0, len(y), len(y))
        ganhos[i] = rmspe(y[indices], base[indices]) - rmspe(
            y[indices], pred[indices]
        )
    return {
        "ganho_mediano_pp": float(100.0 * np.median(ganhos)),
        "ic95_inferior_pp": float(100.0 * np.quantile(ganhos, 0.025)),
        "ic95_superior_pp": float(100.0 * np.quantile(ganhos, 0.975)),
        "probabilidade_ganho": float(np.mean(ganhos > 0)),
    }


def executar():
    raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pasta = os.path.join(raiz, "resultados")
    pasta_sub = os.path.join(raiz, "submissions")
    atual_oof = pd.read_csv(os.path.join(pasta, "juiz_componentes_oof.csv"))
    esp_oof = pd.read_csv(
        os.path.join(pasta, "especialista_caros_estabilidade_oof.csv")
    )
    assert np.array_equal(atual_oof["Id"], esp_oof["Id"])
    y = atual_oof["preco"].to_numpy(dtype=float)
    base_oof = atual_oof["pred_ancorada_publico"].to_numpy(dtype=float)
    especialista_oof = esp_oof[
        "especialista_s42_calibrado_cf"
    ].to_numpy(dtype=float)
    folds = atual_oof["fold"].to_numpy(dtype=int)
    selecionado_oof = selecionar_top_por_fold(
        base_oof, folds, FRACAO_TOP
    )

    atual_teste = pd.read_csv(
        os.path.join(pasta, "juiz_componentes_previsoes_teste.csv")
    )
    esp_teste = pd.read_csv(
        os.path.join(pasta, "especialista_caros_previsoes_teste.csv")
    )
    assert np.array_equal(atual_teste["Id"], esp_teste["Id"])
    base_teste = atual_teste["previsao_final"].to_numpy(dtype=float)
    especialista_teste = esp_teste["especialista_caro"].to_numpy(dtype=float)
    quantidade = int(np.ceil(len(base_teste) * FRACAO_TOP))
    ordem = np.argsort(-base_teste)
    selecionado_teste = np.zeros(len(base_teste), dtype=bool)
    selecionado_teste[ordem[:quantidade]] = True
    assert selecionado_teste.sum() == 100

    resposta = pd.read_csv(
        os.path.join(raiz, "data", "exemplo_arquivo_respostas.csv")
    )
    base_publica = pd.read_csv(
        os.path.join(pasta_sub, "submission_juiz_componentes_ancorado.csv")
    )
    assert np.array_equal(resposta["Id"], atual_teste["Id"])
    assert np.array_equal(base_publica["Id"], atual_teste["Id"])
    assert np.max(
        np.abs(base_publica["preco"].to_numpy() - base_teste)
    ) <= 0.005001

    detalhe = pd.DataFrame({
        "Id": atual_teste["Id"],
        "rank_preco_pipeline": pd.Series(base_teste).rank(
            method="first", ascending=False
        ).astype(int),
        "selecionado_top05": selecionado_teste.astype(int),
        "pipeline_atual_21_19": base_teste,
        "especialista_caro": especialista_teste,
        "diferenca_especialista_pipeline": especialista_teste - base_teste,
    })
    resumos = []
    for peso in PESOS:
        pred_oof = combinar(
            base_oof, especialista_oof, selecionado_oof, peso
        )
        pred_teste = combinar(
            base_teste, especialista_teste, selecionado_teste, peso
        )
        submission = resposta.copy()
        submission["preco"] = pred_teste.round(2)
        caminho = os.path.join(pasta_sub, NOMES[peso])
        submission.to_csv(caminho, index=False)
        detalhe[f"previsao_w{int(peso * 100):03d}"] = pred_teste

        ganhos_folds = {}
        for fold in sorted(np.unique(folds)):
            m = folds == fold
            ganhos_folds[f"ganho_fold_{fold}_pp"] = float(100.0 * (
                rmspe(y[m], base_oof[m]) - rmspe(y[m], pred_oof[m])
            ))
        resumos.append({
            "fracao_top": FRACAO_TOP,
            "peso_especialista": peso,
            "quantidade_oof": int(selecionado_oof.sum()),
            "quantidade_teste": int(selecionado_teste.sum()),
            "rmspe_base_oof": rmspe(y, base_oof),
            "rmspe_gate_oof": rmspe(y, pred_oof),
            "ganho_oof_pp": float(100.0 * (
                rmspe(y, base_oof) - rmspe(y, pred_oof)
            )),
            "folds_melhores": int(sum(v > 0 for v in ganhos_folds.values())),
            "preco_pipeline_minimo_teste": float(
                base_teste[selecionado_teste].min()
            ),
            "peso_medio_global_teste": float(
                peso * selecionado_teste.mean()
            ),
            "alteracao_absoluta_media_selecionados": float(np.mean(
                np.abs(pred_teste[selecionado_teste] - base_teste[selecionado_teste])
            )),
            "alteracao_absoluta_media_global": float(np.mean(
                np.abs(pred_teste - base_teste)
            )),
            "arquivo_submission": NOMES[peso],
            "bootstrap": bootstrap_pareado(y, base_oof, pred_oof),
            **ganhos_folds,
        })

        assert submission.shape == (2000, 2)
        assert submission["Id"].is_unique
        assert np.isfinite(pred_teste).all() and (pred_teste > 0).all()
        assert np.array_equal(
            submission.loc[~selecionado_teste, "preco"].to_numpy(),
            base_publica.loc[~selecionado_teste, "preco"].to_numpy(),
        )
        assert np.max(
            np.abs(submission["preco"].to_numpy() - pred_teste)
        ) <= 0.005001

    detalhe.to_csv(
        os.path.join(pasta, "gate_caros_diagnostico_teste.csv"), index=False
    )
    consolidado = {
        "objetivo": (
            "Distinguir falha do juiz anterior de falha de transferencia "
            "do especialista caro."
        ),
        "score_publico_base": 0.2119,
        "fracao_top": FRACAO_TOP,
        "quantidade_teste_alterada": int(selecionado_teste.sum()),
        "criterio_ranking": "maior previsao do pipeline publico 21.19",
        "resultados": resumos,
        "interpretacao": {
            "ambas_melhoram": (
                "Especialista transfere; comparar intensidades e depois "
                "avaliar ampliar ou treinar residual."
            ),
            "w050_melhora_w100_piora": (
                "Especialista tem direcao util, mas precisa shrinkage/calibracao."
            ),
            "ambas_pioram": (
                "Especialista/calibracao nao transfere nem no topo; nao criar "
                "novo juiz antes de redesenhar o especialista."
            ),
            "w100_melhora_mais": (
                "Problema principal era falso positivo do juiz; especialista "
                "funciona quando selecionado por ranking extremo."
            ),
        },
    }
    with open(
        os.path.join(pasta, "gate_caros_diagnostico_resumo.json"),
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(consolidado, arquivo, ensure_ascii=False, indent=2)
    print(json.dumps(consolidado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    executar()
