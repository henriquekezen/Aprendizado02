"""Mede o teto oraculo e a fronteira entre generalista e especialista."""

import json
import os

import numpy as np
import pandas as pd


def rmspe(y, p):
    return np.sqrt(np.mean(np.square((p - y) / y)))


def procurar_limiar(y, sinal, geral, especialista, limites):
    resultados = []
    for limite in limites:
        usar_especialista = sinal <= limite
        pred = np.where(usar_especialista, especialista, geral)
        resultados.append(
            {
                "limite": float(limite),
                "rmspe": rmspe(y, pred),
                "quantidade_especialista": int(usar_especialista.sum()),
                "fracao_especialista": float(usar_especialista.mean()),
            }
        )
    return pd.DataFrame(resultados).sort_values("rmspe")


def resumir_faixas(df, coluna_faixa, nome_faixa):
    linhas = []
    for faixa, parte in df.groupby(coluna_faixa, observed=True):
        linhas.append(
            {
                "tipo_faixa": nome_faixa,
                "faixa": str(faixa),
                "quantidade": len(parte),
                "preco_real_min": parte["preco"].min(),
                "preco_real_max": parte["preco"].max(),
                "preco_previsto_geral_min": parte["generalista"].min(),
                "preco_previsto_geral_max": parte["generalista"].max(),
                "rmspe_generalista": rmspe(
                    parte["preco"], parte["generalista"]
                ),
                "rmspe_especialista": rmspe(
                    parte["preco"], parte["especialista"]
                ),
                "fracao_especialista_vence": parte[
                    "especialista_vence"
                ].mean(),
                "delta_loss_medio": parte["delta_loss"].mean(),
            }
        )
    return linhas


def main():
    raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pasta = os.path.join(raiz, "resultados")
    geral = pd.read_csv(os.path.join(pasta, "generalista_60_40_oof.csv"))
    esp = pd.read_csv(
        os.path.join(
            pasta,
            "catboost_baratos_curva_iteracoes_d7_rs0_oof.csv",
        )
    )
    assert np.array_equal(geral["Id"].to_numpy(), esp["Id"].to_numpy())
    assert np.array_equal(geral["fold"].to_numpy(), esp["fold_busca"].to_numpy())

    df = geral[["Id", "preco", "fold"]].copy()
    df["generalista"] = geral["generalista_cat60_arvores40"]
    df["especialista"] = esp["iter_600_calibrado_cf"]
    df["loss_generalista"] = np.square(
        (df["generalista"] - df["preco"]) / df["preco"]
    )
    df["loss_especialista"] = np.square(
        (df["especialista"] - df["preco"]) / df["preco"]
    )
    df["delta_loss"] = df["loss_generalista"] - df["loss_especialista"]
    df["especialista_vence"] = df["delta_loss"] > 0

    pred_oraculo_linha = np.where(
        df["especialista_vence"], df["especialista"], df["generalista"]
    )
    limites_reais = np.sort(df["preco"].unique())
    limites_previstos = np.quantile(
        df["generalista"], np.linspace(0.02, 0.98, 193)
    )
    busca_real = procurar_limiar(
        df["preco"].to_numpy(),
        df["preco"].to_numpy(),
        df["generalista"].to_numpy(),
        df["especialista"].to_numpy(),
        limites_reais,
    )
    busca_prevista = procurar_limiar(
        df["preco"].to_numpy(),
        df["generalista"].to_numpy(),
        df["generalista"].to_numpy(),
        df["especialista"].to_numpy(),
        limites_previstos,
    )

    df["decil_preco_real"] = pd.qcut(
        df["preco"], 10, duplicates="drop"
    )
    df["decil_preco_previsto"] = pd.qcut(
        df["generalista"], 10, duplicates="drop"
    )
    faixas = resumir_faixas(df, "decil_preco_real", "preco_real")
    faixas.extend(
        resumir_faixas(df, "decil_preco_previsto", "preco_previsto")
    )

    melhor_real = busca_real.iloc[0]
    melhor_previsto = busca_prevista.iloc[0]
    resumo = {
        "rmspe_generalista": rmspe(df["preco"], df["generalista"]),
        "rmspe_especialista_global": rmspe(df["preco"], df["especialista"]),
        "rmspe_oraculo_por_linha": rmspe(df["preco"], pred_oraculo_linha),
        "fracao_linhas_especialista_vence": float(
            df["especialista_vence"].mean()
        ),
        "limiar_oraculo_preco_real": float(melhor_real["limite"]),
        "rmspe_limiar_preco_real": float(melhor_real["rmspe"]),
        "fracao_especialista_limiar_real": float(
            melhor_real["fracao_especialista"]
        ),
        "melhor_limiar_preco_previsto_in_sample": float(
            melhor_previsto["limite"]
        ),
        "rmspe_limiar_preco_previsto_in_sample": float(
            melhor_previsto["rmspe"]
        ),
        "fracao_especialista_limiar_previsto": float(
            melhor_previsto["fracao_especialista"]
        ),
    }
    with open(
        os.path.join(pasta, "oraculo_generalista_especialista_resumo.json"),
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)
    df.to_csv(
        os.path.join(pasta, "oraculo_generalista_especialista_linhas.csv"),
        index=False,
    )
    busca_real.to_csv(
        os.path.join(pasta, "oraculo_limiares_preco_real.csv"), index=False
    )
    busca_prevista.to_csv(
        os.path.join(pasta, "oraculo_limiares_preco_previsto.csv"), index=False
    )
    pd.DataFrame(faixas).to_csv(
        os.path.join(pasta, "oraculo_desempenho_por_decil.csv"), index=False
    )

    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
