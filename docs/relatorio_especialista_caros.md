# Especialista em imóveis caros

## Conclusão

A busca encontrou um especialista caro forte e estável. A configuração final é
um blend de XGBoost e LightGBM treinados diretamente para RMSPE e apenas na
parte cara da base. Ela não deve ser usada globalmente: o ganho começa de forma
confiável acima de aproximadamente R$ 1 milhão e cresce bastante depois de
R$ 1,3 milhão.

Na validação cruzada repetida, o RMSPE médio foi 19,81% para imóveis de pelo
menos R$ 950 mil e 22,65% para imóveis de pelo menos R$ 1,3 milhão. No holdout
estrito por ID, o especialista obteve 21,15% e 24,11% nessas duas faixas. O
pipeline público de 21,19% obteve 27,26% e 31,94% nas mesmas linhas do holdout.

Nenhum juiz, blend global ou arquivo de submission foi criado nesta etapa. As
previsões do especialista foram preservadas para a futura busca do roteador.

## Protocolo

- Faixa principal de seleção: `preco >= 950000`, com 947 imóveis.
- Faixa de segurança da cauda: `preco >= 1300000`, com 516 imóveis.
- Validação principal: KFold de cinco partes, embaralhado, semente 42.
- Validação final: as sementes 7, 42 e 2026, além de holdout dos 20% menores IDs.
- Correspondências e calibrações sempre foram aprendidas apenas no treino de
  cada fold.
- O critério de seleção priorizou a faixa principal, mas uma alternativa que
  piorasse de forma relevante a cauda era rejeitada.
- A GPU foi descartada depois de um benchmark curto: além de pouco ganho de
  tempo, ela alterou o RMSPE da amostra em 0,125 ponto percentual. A busca final
  foi feita em CPU para manter comparabilidade.

## Referências antes da busca

As referências abaixo receberam uma calibração cross-fit específica para a
faixa cara. Isso torna a comparação conservadora: não estamos comparando o novo
modelo contra uma referência propositalmente descalibrada.

| Referência | RMSPE >= R$ 950 mil | RMSPE >= R$ 1,3 milhão |
| :--- | ---: | ---: |
| Pipeline público de 21,19% | 23,32% | 25,36% |
| Blend XGB/LGB atual com correspondência | 22,88% | 24,61% |

## Rodada inicial: família, alvo e pesos

Foram testados XGBoost, LightGBM e CatBoost com alvo em log, pesos crescentes
para imóveis caros, cortes de treino e alvo bruto. O melhor desenho foi treinar
o preço bruto dividido por 1 milhão com `sample_weight = 1/preco²`. Nesse caso,
a perda quadrática ponderada fica alinhada ao erro percentual quadrático do
RMSPE.

| Candidato | RMSPE >= R$ 950 mil | RMSPE >= R$ 1,3 milhão |
| :--- | ---: | ---: |
| XGB bruto, treino >= R$ 950 mil | **20,58%** | **23,94%** |
| LightGBM bruto, treino >= R$ 950 mil | 21,12% | 24,60% |
| CatBoost em log, treino >= R$ 950 mil | 21,84% | 25,61% |

O CatBoost com prioridade invertida para caros funcionou, mas ficou 1,26 ponto
atrás do XGBoost na faixa principal. Como XGB e LGB também treinavam muito mais
rápido, as rodadas seguintes concentraram capacidade neles.

## Fronteira, capacidade e features

O XGBoost melhorou ao incluir no treino os imóveis a partir de R$ 900 mil. O
LightGBM preferiu o corte de R$ 950 mil e uma configuração um pouco mais capaz.

| Candidato | RMSPE >= R$ 950 mil | RMSPE >= R$ 1,3 milhão |
| :--- | ---: | ---: |
| XGB, corte R$ 900 mil | **20,33%** | 23,82% |
| XGB com target encoding | 20,50% | 23,79% |
| XGB com features caras + target encoding | 20,65% | **23,34%** |
| LightGBM 31 folhas, profundidade 7 | 20,76% | 23,81% |

Uma segunda rodada ao redor do corte e da capacidade não melhorou a referência:
R$ 875 mil, R$ 925 mil, árvores mais profundas, mais iterações e target encoding
permaneceram atrás do XGB com corte de R$ 900 mil. Isso encerrou a busca de
capacidade isolada.

## Correspondência, calibração e blend

A combinação entre erros diferentes produziu o maior ganho da busca. A seleção
foi feita de forma aninhada: pesos, correspondência e calibração de cada fold de
validação foram escolhidos usando somente os demais folds.

- XGB + LightGBM, correspondência até 30% e calibração log-afim: 19,49% na faixa
  principal e 22,41% na cauda.
- A grade refinada encontrou 50% de correspondência para os dois componentes e
  72,5% de peso para o XGB: 19,38% e 22,06%.
- Estender a correspondência até 80% piorou a faixa principal para 19,39%. A
  diferença de 0,017 ponto fechou essa direção de busca.

A calibração log-afim foi superior à multiplicativa. Seu termo de inclinação
negativo corrige a tendência residual de superestimar proporcionalmente a parte
mais alta da curva.

## Configuração congelada

### XGBoost, peso 72,5%

- Treino: 1.021 imóveis com preço de pelo menos R$ 900 mil.
- Alvo: `preco / 1e6`.
- Peso: `1/preco²`, normalizado.
- 800 árvores, taxa 0,03, profundidade 4 e `min_child_weight=5`.
- Correspondência estrutural: 50% quando disponível.

### LightGBM, peso 27,5%

- Treino: 947 imóveis com preço de pelo menos R$ 950 mil.
- Mesmo alvo e peso do XGBoost.
- 600 árvores, 31 folhas, profundidade 7 e mínimo de 40 linhas por folha.
- Correspondência estrutural: 50% quando disponível.

Depois do blend, aplica-se uma calibração log-afim aprendida nas previsões OOF
da faixa `preco >= 950000`.

## Estabilidade

| Semente | RMSPE >= R$ 950 mil | RMSPE >= R$ 1,3 milhão |
| ---: | ---: | ---: |
| 7 | 20,29% | 23,17% |
| 42 | **19,34%** | **22,02%** |
| 2026 | 19,81% | 22,76% |
| Média | **19,81%** | **22,65%** |
| Desvio | 0,47 p.p. | 0,58 p.p. |

O bootstrap pareado com 5.000 reamostragens confirmou o ganho:

| Comparação conservadora | Ganho mediano | IC95 | Probabilidade de ganho |
| :--- | ---: | :--- | ---: |
| vs. árvores atuais, >= R$ 950 mil | 3,54 p.p. | 2,62 a 4,43 p.p. | 100% |
| vs. árvores atuais, >= R$ 1,3 milhão | 2,60 p.p. | 1,37 a 3,73 p.p. | 100% |
| vs. pipeline 21,19 calibrado, >= R$ 950 mil | 3,98 p.p. | 3,04 a 4,89 p.p. | 100% |
| vs. pipeline 21,19 calibrado, >= R$ 1,3 milhão | 3,34 p.p. | 2,08 a 4,53 p.p. | 100% |

No holdout por ID havia 200 imóveis na faixa principal e 101 na cauda:

| Modelo | >= R$ 950 mil | >= R$ 1,3 milhão |
| :--- | ---: | ---: |
| Especialista caro | **21,15%** | **24,11%** |
| Árvores atuais calibradas | 24,29% | 25,97% |
| Pipeline público de 21,19% | 27,26% | 31,94% |

## Onde o especialista deve atuar

A tabela usa a média das três sementes para o especialista e a previsão OOF
real do pipeline público para a referência. Ela revela uma fronteira importante:
o modelo é perigoso abaixo de R$ 900 mil, ainda é instável entre R$ 900 mil e
R$ 1 milhão e passa a ganhar de forma consistente acima de R$ 1 milhão.

| Faixa real | N | Especialista | Pipeline atual | Ganho do especialista |
| :--- | ---: | ---: | ---: | ---: |
| R$ 830–875 mil | 118 | 28,11% | 17,51% | -10,60 p.p. |
| R$ 875–900 mil | 35 | 24,61% | 19,95% | -4,66 p.p. |
| R$ 900–925 mil | 59 | 23,09% | 22,84% | -0,25 p.p. |
| R$ 925–950 mil | 15 | 24,78% | 29,20% | +4,41 p.p. |
| R$ 950 mil–1 milhão | 95 | 20,43% | **18,41%** | -2,02 p.p. |
| R$ 1–1,3 milhão | 336 | **14,15%** | 20,66% | +6,51 p.p. |
| R$ 1,3–1,5 milhão | 146 | **18,95%** | 23,83% | +4,88 p.p. |
| R$ 1,5–2 milhões | 176 | **21,08%** | 30,18% | +9,10 p.p. |
| Acima de R$ 2 milhões | 194 | **26,28%** | 33,91% | +7,63 p.p. |

O corte não pode ser aplicado diretamente porque o preço real não existe no
teste. O próximo juiz deve aprender probabilidade de utilidade, usando as
previsões dos componentes e as features do imóvel. Um bom ponto de partida é
manter peso praticamente zero abaixo da região prevista de R$ 900 mil, fazer
uma transição conservadora entre R$ 900 mil e R$ 1,1 milhão e permitir peso alto
na cauda. O buraco entre R$ 950 mil e R$ 1 milhão mostra por que um corte rígido
em R$ 950 mil seria inferior a um juiz supervisionado.

## Artefatos

- `src/buscar_especialista_caros.py`: busca de família, alvo, cortes, capacidade
  e features.
- `src/refinar_especialista_caros.py`: correspondência, calibração e blends.
- `src/refinar_grade_final_caros.py`: refinamento aninhado da grade final.
- `src/validar_especialista_caros.py`: sementes, holdout, bootstrap e faixas.
- `src/gerar_previsoes_especialista_caros.py`: treino final sem submission.
- `resultados/especialista_caros_validacao_final.json`: validação consolidada.
- `resultados/especialista_caros_comparacao_faixas.csv`: fronteira de utilidade.
- `resultados/especialista_caros_previsoes_teste.csv`: componentes e previsão
  final dos 2.000 imóveis de teste.
- `resultados/especialista_caros_configuracao_final.json`: configuração
  congelada e auditoria das previsões.

No teste, 988 das 2.000 linhas, ou 49,4%, possuem correspondência estrutural.
Todas as previsões finais são finitas e positivas. O especialista tem mediana
de R$ 972 mil e intervalo entre os percentis 5 e 95 de R$ 672 mil a R$ 1,688
milhão. Esses valores são saídas do especialista, não uma decisão de roteamento.

## Juiz implementado

O passo seguinte foi concluído. Um juiz LightGBM de utilidade, protegido por um
classificador da faixa acima de R$ 1 milhão, reduziu o RMSPE médio aninhado de
21,2058% para 21,0307% em três sementes e o holdout de 20,7844% para 20,5392%.
O relatório está em `docs/relatorio_juiz_caros.md` e a única submission gerada
é `submissions/submission_juiz_especialista_caros.csv`.
