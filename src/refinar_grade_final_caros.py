"""Micro-refino cross-fit do blend XGB/LGB especialista caro."""

import argparse
import json
import os

import numpy as np
import pandas as pd

from buscar_especialista_caros import (
    LIMITE_CAUDA,
    LIMITE_PRINCIPAL,
    LIMITES_CURVA,
    carregar_treino,
    rmspe,
)
from refinar_especialista_caros import (
    aplicar_calibracao,
    aplicar_correspondencia,
    fit_calibracao,
)


GRADE_CORR = [0.20, 0.30, 0.40, 0.50]
GRADE_XGB = np.arange(0.55, 0.851, 0.025)


def escolher(y, px, pl, corr, mascara, grade_corr):
    melhor = None
    for cx in grade_corr:
        x = aplicar_correspondencia(px, corr, cx)
        for cl in grade_corr:
            l = aplicar_correspondencia(pl, corr, cl)
            for wx in GRADE_XGB:
                bruto = wx * x + (1.0 - wx) * l
                parametros = fit_calibracao(
                    y[mascara], bruto[mascara], "log_afim"
                )
                pred = aplicar_calibracao(
                    bruto[mascara], "log_afim", parametros
                )
                erro = rmspe(y[mascara], pred)
                if melhor is None or erro < melhor["rmspe"]:
                    melhor = {
                        "corr_xgb": cx,
                        "corr_lgb": cl,
                        "peso_xgb": float(wx),
                        "rmspe": float(erro),
                        "calibracao": parametros,
                    }
    return melhor


def aplicar_config(px, pl, corr, config):
    x = aplicar_correspondencia(px, corr, config["corr_xgb"])
    l = aplicar_correspondencia(pl, corr, config["corr_lgb"])
    bruto = config["peso_xgb"] * x + (1.0 - config["peso_xgb"]) * l
    return aplicar_calibracao(bruto, "log_afim", config["calibracao"])


def executar(grade_corr=None, prefixo="especialista_caros_grade_final"):
    grade_corr = GRADE_CORR if grade_corr is None else grade_corr
    raiz, df = carregar_treino()
    pasta = os.path.join(raiz, "resultados")
    r2 = pd.read_csv(os.path.join(pasta, "especialista_caros_rodada2_oof.csv"))
    corr_df = pd.read_csv(os.path.join(pasta, "catboost_correspondencias_oof.csv"))
    y = df["preco"].to_numpy(dtype=float)
    folds = r2["fold"].to_numpy(dtype=int)
    px = r2["xgb_thr900_bruto"].to_numpy(dtype=float)
    pl = r2["lgb_l31_d7_mc40_i600_bruto"].to_numpy(dtype=float)
    corr = corr_df["previsao_correspondencia"].to_numpy(dtype=float)
    pred_cf = np.full(len(df), np.nan)
    parametros_folds = []
    for fold in sorted(np.unique(folds)):
        treino = (folds != fold) & (y >= LIMITE_PRINCIPAL)
        validacao = folds == fold
        config = escolher(y, px, pl, corr, treino, grade_corr)
        pred_cf[validacao] = aplicar_config(
            px[validacao], pl[validacao], corr[validacao], config
        )
        parametros_folds.append({
            "fold": int(fold),
            **{k: v for k, v in config.items() if k != "calibracao"},
            **config["calibracao"],
        })
    mascara = y >= LIMITE_PRINCIPAL
    final = escolher(y, px, pl, corr, mascara, grade_corr)
    principal = y >= LIMITE_PRINCIPAL
    cauda = y >= LIMITE_CAUDA
    resumo = {
        "rmspe_950_crossfit": rmspe(y[principal], pred_cf[principal]),
        "rmspe_1300_crossfit": rmspe(y[cauda], pred_cf[cauda]),
        "configuracao_final": {
            **{k: v for k, v in final.items() if k != "calibracao"},
            "calibracao": final["calibracao"],
        },
        "parametros_folds": parametros_folds,
    }
    curvas = []
    for limite in LIMITES_CURVA:
        m = y >= limite
        curvas.append({
            "limite_preco": limite,
            "quantidade": int(m.sum()),
            "rmspe": rmspe(y[m], pred_cf[m]),
        })
    oof = df[["Id", "preco"]].copy()
    oof["fold"] = folds
    oof["xgb_bruto"] = px
    oof["lgb_bruto"] = pl
    oof["previsao_correspondencia"] = corr
    oof["especialista_caros_calibrado_cf"] = pred_cf
    oof.to_csv(
        os.path.join(pasta, f"{prefixo}_oof.csv"), index=False
    )
    pd.DataFrame(curvas).to_csv(
        os.path.join(pasta, f"{prefixo}_curvas.csv"), index=False
    )
    with open(
        os.path.join(pasta, f"{prefixo}_resumo.json"),
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print(pd.DataFrame(curvas).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corr", type=float, nargs="+", default=GRADE_CORR)
    parser.add_argument("--prefixo", default="especialista_caros_grade_final")
    args = parser.parse_args()
    executar(args.corr, args.prefixo)
