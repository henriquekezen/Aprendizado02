"""Treina o juiz caro vencedor e gera uma unica submission auditada."""

import json
import os
import time

import numpy as np
import pandas as pd

from gerar_submissions_juiz_especialista import carregar_teste, rmspe
from testar_catboost_alphas import carregar_treino
from testar_juiz_especialista_caros import (
    aplicar_estrategia,
    componentes_oof,
    preparar_features,
    treinar_todos_scores,
)


ESTRATEGIA = "utilidade_gate_faixa"
ARQUIVO_SUBMISSION = "submission_juiz_especialista_caros.csv"


def componentes_teste(raiz, df_teste):
    pasta = os.path.join(raiz, "resultados")
    atual = pd.read_csv(
        os.path.join(pasta, "juiz_componentes_previsoes_teste.csv")
    )
    exp = pd.read_csv(
        os.path.join(pasta, "especialista_caros_previsoes_teste.csv")
    )
    assert np.array_equal(df_teste["Id"], atual["Id"])
    assert np.array_equal(df_teste["Id"], exp["Id"])
    componentes = {
        "base": atual["previsao_final"],
        "generalista": atual["generalista"],
        "especialista_barato": atual["especialista"],
        "catboost_global": atual["catboost_global"],
        "blend_arvores": atual["blend_arvores"],
        "peso_especialista_barato": atual["peso_especialista"],
        "fracao_arvores": atual["fracao_arvores_no_restante"],
        "tem_correspondencia": exp["tem_correspondencia"],
    }
    return atual, exp, componentes


def executar():
    raiz, df = carregar_treino()
    teste = carregar_teste(raiz)
    pasta = os.path.join(raiz, "resultados")

    estabilidade = pd.read_csv(
        os.path.join(pasta, "juiz_caros_estabilidade_resumo.csv")
    ).sort_values("rmspe_medio")
    holdout = pd.read_csv(
        os.path.join(pasta, "juiz_caros_holdout_resumo.csv")
    ).set_index("estrategia")
    resumo_oof = pd.read_csv(
        os.path.join(pasta, "juiz_caros_resumo.csv")
    ).set_index("estrategia")
    vencedor = estabilidade.iloc[0]
    assert vencedor["estrategia"] == ESTRATEGIA
    assert vencedor["ganho_minimo_pp"] > 0.03
    assert vencedor["folds_melhores_total"] >= 13
    assert holdout.loc[ESTRATEGIA, "ganho_pp_vs_base"] > 0

    atual_oof, exp_oof, componentes_treino = componentes_oof(pasta)
    x_oof = preparar_features(
        df,
        componentes_treino,
        exp_oof["especialista_s42_calibrado_cf"],
    )
    atual_teste, exp_teste, componentes_t = componentes_teste(raiz, teste)
    x_teste = preparar_features(
        teste,
        componentes_t,
        exp_teste["especialista_caro"],
    )

    y = df["preco"].to_numpy(dtype=float)
    base_oof = atual_oof["pred_ancorada_publico"].to_numpy(dtype=float)
    especialista_oof = exp_oof[
        "especialista_s42_calibrado_cf"
    ].to_numpy(dtype=float)
    inicio = time.perf_counter()
    scores_teste = treinar_todos_scores(
        x_oof, y, base_oof, especialista_oof, x_teste
    )
    tempo = time.perf_counter() - inicio

    with open(
        os.path.join(pasta, "juiz_caros_parametros_finais.json"),
        encoding="utf-8",
    ) as arquivo:
        parametros = json.load(arquivo)[ESTRATEGIA]
    base_teste = atual_teste["previsao_final"].to_numpy(dtype=float)
    especialista_teste = exp_teste["especialista_caro"].to_numpy(dtype=float)
    peso = aplicar_estrategia(
        ESTRATEGIA, parametros, scores_teste, base_teste
    )
    pred = base_teste + peso * (especialista_teste - base_teste)

    oof = pd.read_csv(os.path.join(pasta, "juiz_caros_oof.csv"))
    scores_oof = {
        nome: oof[f"score_{nome}"].to_numpy()
        for nome in ["faixa_1m", "continuo_q95", "continuo_q99", "utilidade"]
    }
    peso_oof_final = aplicar_estrategia(
        ESTRATEGIA, parametros, scores_oof, base_oof
    )
    pred_oof_final = base_oof + peso_oof_final * (
        especialista_oof - base_oof
    )

    previsoes = pd.DataFrame({
        "Id": teste["Id"].to_numpy(),
        "pipeline_atual_21_19": base_teste,
        "especialista_caro": especialista_teste,
        "score_faixa_acima_1m": scores_teste["faixa_1m"],
        "score_utilidade_especialista": scores_teste["utilidade"],
        "peso_especialista_caro": peso,
        "previsao_final": pred,
    })
    caminho_previsoes = os.path.join(
        pasta, "juiz_caros_previsoes_teste.csv"
    )
    previsoes.to_csv(caminho_previsoes, index=False)

    resposta = pd.read_csv(
        os.path.join(raiz, "data", "exemplo_arquivo_respostas.csv")
    )
    assert np.array_equal(resposta["Id"], teste["Id"])
    submission = resposta.copy()
    submission["preco"] = pred.round(2)
    caminho_submission = os.path.join(
        raiz, "submissions", ARQUIVO_SUBMISSION
    )
    submission.to_csv(caminho_submission, index=False)

    assert submission.shape == (2_000, 2)
    assert submission["Id"].is_unique
    assert np.isfinite(pred).all() and (pred > 0).all()
    assert not submission.isna().any().any()
    assert np.max(np.abs(submission["preco"].to_numpy() - pred)) <= 0.005001

    faixas_teste = []
    for inferior, superior in [
        (0, 900_000),
        (900_000, 1_100_000),
        (1_100_000, 1_400_000),
        (1_400_000, 2_000_000),
        (2_000_000, np.inf),
    ]:
        m = (base_teste >= inferior) & (base_teste < superior)
        faixas_teste.append({
            "limite_inferior_pred_pipeline": inferior,
            "limite_superior_pred_pipeline": superior,
            "quantidade": int(m.sum()),
            "peso_medio_especialista_caro": float(peso[m].mean()),
            "peso_p90_especialista_caro": float(np.quantile(peso[m], 0.90)),
        })
    pd.DataFrame(faixas_teste).to_csv(
        os.path.join(pasta, "juiz_caros_pesos_teste_por_faixa.csv"),
        index=False,
    )

    decisao = {
        "estrategia": ESTRATEGIA,
        "familia_meta_modelo": "LightGBM",
        "protecao": "probabilidade_preco_maior_igual_1m_elevada_a_gamma",
        "parametros_mapeamento": parametros,
        "rmspe_base_oof": rmspe(y, base_oof),
        "rmspe_oof_aninhado_semente42": float(
            resumo_oof.loc[ESTRATEGIA, "rmspe_crossfit_aninhado"]
        ),
        "rmspe_oof_mapeamento_final": rmspe(y, pred_oof_final),
        "rmspe_medio_tres_sementes": float(vencedor["rmspe_medio"]),
        "ganho_medio_tres_sementes_pp": float(vencedor["ganho_medio_pp"]),
        "ganho_minimo_semente_pp": float(vencedor["ganho_minimo_pp"]),
        "folds_melhores_total": int(vencedor["folds_melhores_total"]),
        "rmspe_base_holdout": 0.20784386749850803,
        "rmspe_holdout": float(holdout.loc[ESTRATEGIA, "rmspe_holdout"]),
        "ganho_holdout_pp": float(
            holdout.loc[ESTRATEGIA, "ganho_pp_vs_base"]
        ),
        "peso_medio_teste": float(peso.mean()),
        "peso_mediano_teste": float(np.median(peso)),
        "peso_p90_teste": float(np.quantile(peso, 0.90)),
        "fracao_peso_menor_001": float(np.mean(peso < 0.01)),
        "fracao_peso_maior_05": float(np.mean(peso > 0.50)),
        "alteracao_absoluta_media_reais": float(
            np.mean(np.abs(pred - base_teste))
        ),
        "tempo_treino_final_segundos": tempo,
        "arquivo_submission": ARQUIVO_SUBMISSION,
        "arquivo_previsoes": os.path.basename(caminho_previsoes),
    }
    with open(
        os.path.join(pasta, "juiz_caros_decisao_final.json"),
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(decisao, arquivo, ensure_ascii=False, indent=2)
    print(json.dumps(decisao, ensure_ascii=False, indent=2))
    print(f"Submission: {caminho_submission}")


if __name__ == "__main__":
    executar()
