"""Gera as submissions do blend com o LightGBM de objetivo RMSPE exato.

O experimento `testar_objetivo_rmspe_global.py` encontrou que um LightGBM
global treinado com alvo bruto (`preco/1e6`) e `sample_weight = 1/preco^2`,
misturado com peso de 25% sobre o pipeline publico de 21,19%, melhora o OOF
aninhado em 0,068 p.p. e o holdout por ID em 0,189 p.p.

Como o ganho interno se concentra na faixa acima de R$ 1 milhao — regiao em
que a validacao interna ja divergiu do placar publico — sao geradas duas
intensidades: 25% (principal) e 12,5% (reserva conservadora). Fora do peso,
os dois arquivos usam exatamente as mesmas previsoes.
"""

import json
import os

import numpy as np
import pandas as pd

from buscar_especialista_caros import criar_lgb, preparar_arvores
from testar_catboost_alphas import carregar_treino, corrigir_dados
from testar_catboost_correspondencias import (
    aplicar_correspondencias,
    prever_correspondencias,
)
from testar_objetivo_rmspe_global import (
    PESO_CORRESPONDENCIA,
    alvo_bruto,
    pesos_rmspe,
    restaurar_bruto,
)


PESOS_BLEND = [0.25, 0.125]
SUBMISSION_BASE = "submission_juiz_componentes_ancorado.csv"


def carregar_teste(raiz):
    caminho = os.path.join(raiz, "data", "conjunto_de_teste (3).csv")
    return corrigir_dados(pd.read_csv(caminho), corrigir_alvo=False).reset_index(
        drop=True
    )


def treinar_lgb_raw_final(treino, teste):
    treino_m, teste_m = preparar_arvores(treino, teste)
    x_treino = treino_m.drop(columns=["Id", "preco"])
    x_teste = teste_m.drop(columns=["Id", "preco"], errors="ignore")
    y = treino_m["preco"].to_numpy(dtype=float)
    modelo = criar_lgb()
    modelo.fit(
        x_treino,
        alvo_bruto(y),
        sample_weight=pesos_rmspe(y),
        categorical_feature=["bairro"],
    )
    return restaurar_bruto(modelo.predict(x_teste))


def main():
    raiz, treino = carregar_treino()
    teste = carregar_teste(raiz)
    assert len(teste) == 2_000
    assert teste["Id"].is_unique

    base = pd.read_csv(os.path.join(raiz, "submissions", SUBMISSION_BASE))
    base = teste[["Id"]].merge(base, on="Id", how="left")
    assert base["preco"].notna().all()
    previsao_base = base["preco"].to_numpy(dtype=float)

    lgb_raw = treinar_lgb_raw_final(treino, teste)
    correspondencia = prever_correspondencias(treino, teste)
    lgb_raw = aplicar_correspondencias(
        lgb_raw, correspondencia, PESO_CORRESPONDENCIA
    )
    assert np.isfinite(lgb_raw).all() and (lgb_raw > 0).all()

    resumo = {
        "submission_base": SUBMISSION_BASE,
        "correspondencia_teste": int(np.isfinite(correspondencia).sum()),
        "arquivos": {},
    }
    for peso in PESOS_BLEND:
        previsao = (1 - peso) * previsao_base + peso * lgb_raw
        assert np.isfinite(previsao).all() and (previsao > 0).all()
        nome = f"submission_blend_rmspe_raw_w{int(peso * 1000):03d}.csv"
        saida = pd.DataFrame({"Id": teste["Id"], "preco": np.round(previsao, 2)})
        saida.to_csv(os.path.join(raiz, "submissions", nome), index=False)
        alteracao = np.abs(previsao - previsao_base)
        resumo["arquivos"][nome] = {
            "peso_lgb_raw": peso,
            "alteracao_media_reais": float(alteracao.mean()),
            "alteracao_maxima_reais": float(alteracao.max()),
            "linhas_alteradas_acima_1pct": int(
                (alteracao / previsao_base > 0.01).sum()
            ),
        }
        print(
            f"{nome}: alteracao media R$ {alteracao.mean():,.0f}, "
            f"maxima R$ {alteracao.max():,.0f}"
        )

    previsoes = pd.DataFrame(
        {
            "Id": teste["Id"],
            "previsao_base": previsao_base,
            "lgb_raw_corr30": lgb_raw,
        }
    )
    previsoes.to_csv(
        os.path.join(raiz, "resultados", "objetivo_rmspe_global_teste.csv"),
        index=False,
    )
    with open(
        os.path.join(
            raiz, "resultados", "objetivo_rmspe_global_submissions_resumo.json"
        ),
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)
    print("Previsoes preservadas em resultados/objetivo_rmspe_global_teste.csv.")


if __name__ == "__main__":
    main()
