"""Avalia gates simples no topo do preco previsto pelo pipeline atual.

Esta e uma ablacao diagnostica: nenhum modelo e retreinado. O especialista caro
so recebe peso nas maiores previsoes do pipeline publico de 21,19%.
"""

import os

import numpy as np
import pandas as pd

from gerar_submissions_juiz_especialista import rmspe
from testar_catboost_alphas import carregar_treino


FRACOES = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10]
PESOS = [0.10, 0.20, 0.25, 1.0 / 3.0, 0.50, 0.75, 1.0]
FAIXAS = [
    (0, 1_000_000),
    (1_000_000, 1_300_000),
    (1_300_000, 1_500_000),
    (1_500_000, 2_000_000),
    (2_000_000, np.inf),
]


def selecionar_top(valores, fracao):
    valores = np.asarray(valores, dtype=float)
    quantidade = max(1, int(np.ceil(len(valores) * fracao)))
    indices = np.argpartition(valores, -quantidade)[-quantidade:]
    mascara = np.zeros(len(valores), dtype=bool)
    mascara[indices] = True
    return mascara


def selecionar_top_por_fold(valores, folds, fracao):
    mascara = np.zeros(len(valores), dtype=bool)
    for fold in sorted(np.unique(folds)):
        indices = np.flatnonzero(folds == fold)
        mascara[indices[selecionar_top(valores[indices], fracao)]] = True
    return mascara


def combinar(base, especialista, selecionado, peso):
    previsao = np.asarray(base, dtype=float).copy()
    previsao[selecionado] += peso * (
        np.asarray(especialista, dtype=float)[selecionado]
        - previsao[selecionado]
    )
    return previsao


def avaliar_grade(y, base, especialista, folds, tipo):
    linhas, faixas = [], []
    base_rmspe = rmspe(y, base)
    for fracao in FRACOES:
        selecionado = (
            selecionar_top_por_fold(base, folds, fracao)
            if folds is not None
            else selecionar_top(base, fracao)
        )
        for peso in PESOS:
            pred = combinar(base, especialista, selecionado, peso)
            linha = {
                "avaliacao": tipo,
                "fracao_top": fracao,
                "peso_especialista": peso,
                "quantidade_selecionada": int(selecionado.sum()),
                "rmspe_base": base_rmspe,
                "rmspe_gate": rmspe(y, pred),
                "ganho_pp": 100.0 * (base_rmspe - rmspe(y, pred)),
                "preco_previsto_minimo_selecionado": float(base[selecionado].min()),
                "preco_real_mediano_selecionado": float(np.median(y[selecionado])),
                "proporcao_real_acima_1m": float(np.mean(y[selecionado] >= 1_000_000)),
                "proporcao_real_acima_1300": float(np.mean(y[selecionado] >= 1_300_000)),
                "proporcao_real_acima_1500": float(np.mean(y[selecionado] >= 1_500_000)),
                "proporcao_real_acima_2m": float(np.mean(y[selecionado] >= 2_000_000)),
            }
            if folds is not None:
                for fold in sorted(np.unique(folds)):
                    m = folds == fold
                    linha[f"ganho_fold_{fold}_pp"] = 100.0 * (
                        rmspe(y[m], base[m]) - rmspe(y[m], pred[m])
                    )
            linhas.append(linha)
            for inferior, superior in FAIXAS:
                m = (y >= inferior) & (y < superior)
                faixas.append({
                    "avaliacao": tipo,
                    "fracao_top": fracao,
                    "peso_especialista": peso,
                    "limite_inferior": inferior,
                    "limite_superior": superior,
                    "quantidade": int(m.sum()),
                    "quantidade_selecionada": int((m & selecionado).sum()),
                    "rmspe_base": rmspe(y[m], base[m]),
                    "rmspe_gate": rmspe(y[m], pred[m]),
                    "ganho_pp": 100.0 * (
                        rmspe(y[m], base[m]) - rmspe(y[m], pred[m])
                    ),
                })
    return pd.DataFrame(linhas), pd.DataFrame(faixas)


def executar():
    raiz, df = carregar_treino()
    pasta = os.path.join(raiz, "resultados")
    atual = pd.read_csv(os.path.join(pasta, "juiz_componentes_oof.csv"))
    especialista = pd.read_csv(
        os.path.join(pasta, "especialista_caros_estabilidade_oof.csv")
    )
    assert np.array_equal(df["Id"], atual["Id"])
    assert np.array_equal(atual["Id"], especialista["Id"])
    y = df["preco"].to_numpy(dtype=float)
    base = atual["pred_ancorada_publico"].to_numpy(dtype=float)
    esp = especialista["especialista_s42_calibrado_cf"].to_numpy(dtype=float)
    folds = atual["fold"].to_numpy(dtype=int)
    resumo_oof, faixas_oof = avaliar_grade(
        y, base, esp, folds, "oof_por_fold"
    )

    atual_h = pd.read_csv(
        os.path.join(pasta, "juiz_componentes_holdout.csv")
    )
    esp_h = pd.read_csv(
        os.path.join(pasta, "especialista_caros_holdout_id.csv")
    )
    assert np.array_equal(atual_h["Id"], esp_h["Id"])
    resumo_h, faixas_h = avaliar_grade(
        atual_h["preco"].to_numpy(dtype=float),
        atual_h["pred_ancorada_publico"].to_numpy(dtype=float),
        esp_h["especialista_caros"].to_numpy(dtype=float),
        None,
        "holdout_id",
    )

    resumo = pd.concat([resumo_oof, resumo_h], ignore_index=True)
    faixas = pd.concat([faixas_oof, faixas_h], ignore_index=True)
    resumo.to_csv(
        os.path.join(pasta, "gate_caros_conservador_grade.csv"), index=False
    )
    faixas.to_csv(
        os.path.join(pasta, "gate_caros_conservador_faixas.csv"), index=False
    )
    comparacao = resumo_oof.merge(
        resumo_h[
            ["fracao_top", "peso_especialista", "rmspe_gate", "ganho_pp"]
        ],
        on=["fracao_top", "peso_especialista"],
        suffixes=("_oof", "_holdout"),
    )
    comparacao["folds_melhores"] = comparacao.apply(
        lambda linha: sum(linha[f"ganho_fold_{fold}_pp"] > 0 for fold in range(1, 6)),
        axis=1,
    )
    comparacao = comparacao.sort_values(
        ["rmspe_gate_oof", "rmspe_gate_holdout"]
    )
    comparacao.to_csv(
        os.path.join(pasta, "gate_caros_conservador_comparacao.csv"),
        index=False,
    )
    print(comparacao.head(30).to_string(index=False))


if __name__ == "__main__":
    executar()
