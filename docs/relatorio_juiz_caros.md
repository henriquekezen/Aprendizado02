# Juiz do especialista em imóveis caros

## Resultado

Foi implementado um novo juiz para inserir o especialista caro sobre o pipeline
público de 21,19%. A fórmula final é convexa:

```text
previsao = pipeline_atual
         + peso_caro * (especialista_caro - pipeline_atual)
```

O peso não é decidido apenas pelo preço previsto. Dois LightGBM independentes
estimam:

- a probabilidade de o especialista caro produzir menor perda RMSPE;
- a probabilidade de o imóvel custar pelo menos R$ 1 milhão.

O primeiro score passa por uma sigmoide calibrada. O segundo é elevado à sexta
potência e atua como proteção de faixa:

```text
peso_caro = sigmoid((score_utilidade - 0,50084) / 0,20988)
           * probabilidade_acima_1m ** 6
```

Isso permite peso alto na cauda, mas reduz rapidamente falsas ativações fora
dela.

## Protocolo sem vazamento direto

Os modelos-base e o especialista já estavam disponíveis como previsões OOF nos
mesmos cinco folds. A validação do juiz foi aninhada:

1. Em cada fold externo, o treino restante foi novamente dividido em quatro
   folds internos.
2. Os scores internos OOF escolheram o mapeamento de score para peso.
3. Os meta-modelos foram treinados em todo o treino externo.
4. O mapeamento congelado foi aplicado ao fold externo nunca usado.

O holdout dos 20% menores IDs repetiu esse processo. O juiz, seu mapeamento e
os modelos-base do holdout foram treinados fora das linhas avaliadas.

O CatBoost foi inicialmente usado como meta-modelo, mas seu boosting `Ordered`
paralelizou pouco e tornaria cada rodada excessivamente lenta. O meta-modelo foi
trocado por LightGBM regularizado, mantendo categorias codificadas somente com
o treino de cada ajuste e preservando toda a estrutura aninhada.

## Hipóteses avaliadas

Foram comparados:

- regressão do peso ótimo alinhada ao RMSPE, com dois níveis de clipping dos
  pesos de treino;
- classificador direto de utilidade;
- classificador da faixa acima de R$ 1 milhão;
- gate suave apenas pelo preço previsto do pipeline;
- proteções por probabilidade de faixa, preço e combinação das duas.

Na primeira grade, o juiz de utilidade foi o melhor com 21,1021% OOF e 20,5292%
no holdout. O expoente de proteção e o centro do gate de preço bateram nos
limites da grade. A rodada seguinte ampliou esses limites e testou gates duplos.

Na semente 42, o gate simples por preço chegou a 21,0425% e ganhou os cinco
folds. Porém, seu holdout foi apenas 20,7178%, e ele piorou as faixas reais de
R$ 1,3 a R$ 2 milhões. O juiz de utilidade obteve 21,0687% nessa semente, mas
20,5392% no holdout e comportamento melhor na cauda. Por isso, a decisão foi
levada para KFold aninhado repetido.

## Estabilidade em três sementes

| Estratégia | RMSPE médio | Desvio | Ganho médio | Pior ganho | Folds melhores |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Utilidade + proteção de faixa | **21,0307%** | 0,034 p.p. | **0,175 p.p.** | **0,137 p.p.** | **14/15** |
| Classificador de faixa puro | 21,0509% | 0,038 p.p. | 0,155 p.p. | 0,121 p.p. | 14/15 |
| Gate pelo preço previsto | 21,0535% | **0,021 p.p.** | 0,152 p.p. | 0,128 p.p. | 14/15 |
| Regressão contínua q99 + faixa | 21,0599% | 0,049 p.p. | 0,146 p.p. | 0,095 p.p. | 14/15 |

Resultados do vencedor por semente:

| Semente | RMSPE aninhado | Ganho sobre 21,2058% | Folds melhores |
| ---: | ---: | ---: | ---: |
| 7 | **21,0021%** | 0,204 p.p. | 5/5 |
| 42 | 21,0687% | 0,137 p.p. | 4/5 |
| 2026 | 21,0213% | 0,185 p.p. | 5/5 |

O juiz de utilidade foi escolhido porque combina o menor RMSPE médio, o maior
ganho mínimo entre sementes e o melhor holdout. A escolha não depende da única
semente em que o gate por preço ficou na frente.

## Holdout por ID

| Modelo | RMSPE |
| :--- | ---: |
| Pipeline atual | 20,7844% |
| Juiz caro | **20,5392%** |

O ganho fora do KFold foi de 0,245 ponto percentual, maior que o ganho médio de
0,175 ponto observado nas três sementes.

## Comportamento por preço real no OOF

| Faixa | Pipeline atual | Juiz caro | Ganho |
| :--- | ---: | ---: | ---: |
| Abaixo de R$ 740 mil | **19,72%** | 20,13% | -0,41 p.p. |
| R$ 740–900 mil | **19,49%** | 20,60% | -1,11 p.p. |
| R$ 900 mil–1 milhão | **21,17%** | 21,83% | -0,66 p.p. |
| R$ 1–1,3 milhão | 20,66% | **20,16%** | +0,50 p.p. |
| R$ 1,3–1,5 milhão | 23,83% | **23,36%** | +0,46 p.p. |
| R$ 1,5–2 milhões | 30,18% | **27,37%** | +2,81 p.p. |
| Acima de R$ 2 milhões | 33,91% | **28,64%** | +5,27 p.p. |

O custo nas faixas inferiores vem de um pequeno número de falsos positivos do
juiz. O ganho na cauda é maior em perda quadrática e produz a melhora global.
Gates duplos mais rígidos reduziram parte desses falsos positivos, mas perderam
resultado médio e estabilidade; por isso não foram escolhidos.

## Comportamento no teste

| Faixa do preço previsto pelo pipeline | Imóveis | Peso médio caro | P90 do peso |
| :--- | ---: | ---: | ---: |
| Abaixo de R$ 900 mil | 1.572 | 0,35% | ~0% |
| R$ 900 mil–1,1 milhão | 154 | 18,04% | 48,38% |
| R$ 1,1–1,4 milhão | 166 | 42,95% | 63,03% |
| R$ 1,4–2 milhões | 63 | 65,92% | 81,78% |
| Acima de R$ 2 milhões | 45 | 63,24% | 75,32% |

No conjunto inteiro, o peso médio do especialista caro é 8,73%, a mediana é
praticamente zero e o percentil 90 é 42,93%. Em 77,2% dos imóveis o peso fica
abaixo de 1%; somente 7,8% recebem mais de 50%. A alteração absoluta média em
relação ao pipeline de 21,19% é de R$ 18.706.

## Submission e artefatos

Foi gerada uma única submission:

- `submissions/submission_juiz_especialista_caros.csv`

Ela contém 2.000 IDs únicos, nenhum ausente e nenhum preço não positivo. Os
arquivos reproduzíveis são:

- `src/testar_juiz_especialista_caros.py`: busca e validação aninhada;
- `src/validar_juiz_caros_sementes.py`: estabilidade em três sementes;
- `src/gerar_submission_juiz_caros.py`: treino final e auditoria;
- `resultados/juiz_caros_oof.csv`: scores, pesos e previsões OOF;
- `resultados/juiz_caros_holdout.csv`: auditoria por ID;
- `resultados/juiz_caros_estabilidade_resumo.csv`: comparação repetida;
- `resultados/juiz_caros_previsoes_teste.csv`: previsão final detalhada;
- `resultados/juiz_caros_decisao_final.json`: decisão consolidada.

O próximo passo é observar o score público desta única direção. Como o ganho
interno é consistente, não há justificativa atual para gerar variações de
intensidade antes desse retorno.

## Resultado público e revisão da decisão

O juiz caro obteve **22,32% no Kaggle**, piorando muito a referência de 21,19%.
Essa diferença é grande demais para ser tratada como ruído; o juiz foi
descartado. A hipótese principal passou a ser falso positivo de roteamento, mas
o score isolado não permite separar completamente juiz e especialista.

Foram então geradas duas ablações que modificam somente o top 5% do preço
previsto pelo pipeline atual, com pesos de 50% e 100% no especialista. O desenho
e a interpretação dos futuros resultados estão em
`docs/relatorio_gate_conservador_caros.md`.

