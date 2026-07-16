# Relatório da busca do CatBoost especialista em imóveis baratos

## Resumo executivo

A busca partiu do CatBoost com `alpha=1,5`, 1.000 árvores, profundidade 6 e sem correspondência. O objetivo principal foi o RMSPE out-of-fold dos 1.178 imóveis com preço de até R$ 355 mil.

O melhor especialista encontrado utiliza:

```text
target=log1p(preco)
sample_weight=1 / preco^2
iterations=600
learning_rate=0,03
depth=7
l2_leaf_reg=5
random_strength=0
bootstrap_type=Bayesian
bagging_temperature=1
boosting_type=Ordered
correspondencia=0%
fator de calibração final=0,914489
```

O RMSPE até R$ 355 mil ficou em:

| Modelo | Sem calibração | Calibração cross-fit |
| :--- | ---: | ---: |
| CatBoost barato original, alpha 1,5 | 22,26% | 19,29% |
| CatBoost especialista final | **20,98%** | **18,90%** |

A melhora estrutural antes da calibração foi de 1,27 ponto percentual. Depois de calibrar ambos de forma comparável, o ganho do novo modelo foi de 0,39 ponto. O ganho pareado apareceu em quatro dos cinco folds; um bootstrap pareado por linha produziu intervalo indicativo de 0,14 a 0,66 ponto percentual, com mediana de 0,39 ponto.

O blend global de 60% CatBoost e 40% XGBoost/LightGBM obteve 21,45% no Kaggle e permanece como generalista. O especialista deste relatório não foi treinado na base completa, não gerou submission e ainda não foi combinado com um juiz.

## Protocolo

- 85 treinamentos completos de fold, distribuídos em sete blocos experimentais.
- Aproximadamente 76 minutos de execução acumulada dos blocos CatBoost.
- Base corrigida com 4.678 imóveis.
- `KFold(n_splits=5, shuffle=True, random_state=42)` em todas as rodadas.
- Métrica principal: RMSPE acumulado para `preco <= 355000`.
- Cortes auxiliares: R$ 250 mil, R$ 300 mil, R$ 400 mil, R$ 500 mil e R$ 830 mil.
- Correspondências calculadas apenas dentro do treino de cada fold quando avaliadas.
- Calibração multiplicativa aprendida fora do fold avaliado.
- Nenhum preço de validação entrou no treino do próprio CatBoost.

A calibração minimiza diretamente o erro percentual quadrático para previsões fixas. Para cada parte de treino, o multiplicador foi:

```text
c = soma(previsao / real) / soma((previsao / real)^2)
```

O fator final de 0,914489 foi calculado usando todas as previsões OOF baratas e deve ser aplicado somente ao especialista, antes do futuro roteamento.

## Descobertas por rodada

### Correspondência

A correspondência estrutural ajudava o CatBoost globalmente, mas prejudicava os baratos. No CatBoost `alpha=1,5`, o RMSPE até R$ 355 mil foi:

| Correspondência | RMSPE |
| ---: | ---: |
| 0% | **22,26%** |
| 15% | 22,36% |
| 30% | 22,67% |

Por isso, todas as rodadas do especialista usaram correspondência zero. O generalista pode continuar usando 30%.

### Alpha no alvo em log

Elevar o alpha reduziu o erro bruto, mas não o erro calibrado indefinidamente:

| Configuração | RMSPE bruto até R$ 355 mil | RMSPE calibrado |
| :--- | ---: | ---: |
| `log alpha=2` | 21,12% | **19,18%** |
| `log alpha=2,5` | 20,47% | 19,23% |
| `log alpha=3` | **20,07%** | 19,36% |

O `alpha=3` deslocava as previsões para baixo, mas perdia estrutura depois que o viés era removido. `Alpha=2` foi o melhor para a faixa ampla até R$ 355 mil. No extremo até R$ 250 mil, alphas entre 2,5 e 3 continuaram competitivos, mostrando que o alpha ideal depende da largura da faixa barata.

### Alvo bruto

Também foi testado o preço original dividido por 100 mil. Nessa formulação, `alpha=2` corresponde diretamente à soma dos erros percentuais quadráticos. O melhor candidato bruto foi `raw alpha=3`, com 19,22% calibrados, ligeiramente pior que os 19,18% de `log alpha=2`. A família de alvo bruto foi rejeitada.

### Quantidade de árvores do baseline depth 6

Uma única sequência de 1.600 árvores foi avaliada em vários estágios:

| Árvores | RMSPE calibrado até R$ 355 mil |
| ---: | ---: |
| 400 | 19,34% |
| 600 | 19,27% |
| 800 | 19,19% |
| 1.000 | **19,18%** |
| 1.200 | 19,21% |
| 1.400 | 19,22% |
| 1.600 | 19,24% |

O baseline começava a sobreajustar depois de aproximadamente 1.000 árvores.

### Capacidade e regularização

| Configuração | RMSPE calibrado até R$ 355 mil |
| :--- | ---: |
| Depth 7, 800 árvores | **19,18%** |
| Depth 6, baseline | 19,18% |
| Depth 6, `l2_leaf_reg=10` | 19,24% |
| Depth 5, 1.200 árvores | 19,32% |

Profundidade e L2 isoladamente não trouxeram ganho material. A melhora de depth 7 apareceu somente depois da retirada da força aleatória.

### Aleatoriedade

| Configuração depth 6 | RMSPE calibrado até R$ 355 mil |
| :--- | ---: |
| `random_strength=1`, `bagging_temperature=1` | 19,18% |
| `random_strength=0`, `bagging_temperature=1` | **19,04%** |
| `random_strength=0`, `bagging_temperature=0` | 19,15% |
| `random_strength=1`, `bagging_temperature=0` | 19,45% |

O bootstrap Bayesiano ajuda, mas a aleatoriedade aplicada à escolha dos splits prejudica o especialista. O refinamento confirmou o limite:

| Configuração | RMSPE calibrado até R$ 355 mil |
| :--- | ---: |
| Depth 6, `random_strength=0,50` | 19,03% |
| Depth 6, `random_strength=0,25` | 18,97% |
| Depth 6, `random_strength=0` | 19,04% |
| Depth 7, `random_strength=0`, 800 árvores | **18,92%** |

### Curva final de árvores

Para depth 7 e força aleatória zero:

| Árvores | RMSPE calibrado até R$ 355 mil |
| ---: | ---: |
| 400 | 18,903% |
| 600 | **18,901%** |
| 800 | 18,924% |
| 1.000 | 18,935% |
| 1.200 | 18,929% |

400 e 600 árvores estão tecnicamente empatadas. A configuração de 600 foi registrada como final por obter o menor valor, mas 400 é uma alternativa mais barata e praticamente indistinguível.

## Curva por limite de preço

Comparação entre o CatBoost barato original calibrado e o especialista final calibrado:

| Limite real | Imóveis | Original | Final | Ganho |
| ---: | ---: | ---: | ---: | ---: |
| R$ 250 mil | 330 | 21,84% | **21,18%** | 0,65 p.p. |
| R$ 300 mil | 695 | 20,39% | **19,87%** | 0,53 p.p. |
| R$ 355 mil | 1.178 | 19,29% | **18,90%** | 0,39 p.p. |
| R$ 400 mil | 1.657 | 19,21% | **18,91%** | 0,31 p.p. |
| R$ 500 mil | 2.316 | 20,08% | **19,89%** | 0,18 p.p. |
| R$ 830 mil | 3.515 | 21,53% | **21,49%** | 0,05 p.p. |

O ganho desaparece progressivamente conforme imóveis mais caros entram no conjunto. Isso confirma que o modelo final é um especialista. A fronteira operacional exata ainda não pode ser definida apenas por esta tabela: ela deve ser determinada comparando sua perda OOF com a do generalista de 21,45%, imóvel a imóvel.

## Estabilidade pareada

No corte de R$ 355 mil:

- o especialista final venceu o Cat barato original em 53,4% das linhas;
- a melhora veio principalmente da redução dos erros quadráticos mais graves;
- quatro dos cinco folds melhoraram;
- o único fold pior caiu apenas 0,005 ponto percentual;
- os demais ganhos por fold foram 0,68, 0,02, 0,47 e 0,67 ponto percentual;
- o fator de calibração do finalista variou de 0,9082 a 0,9175 entre folds.

O fato de vencer pouco mais da metade das linhas, mas reduzir claramente o RMSPE, mostra que o especialista não melhora todos os casos: ele reduz erros de maior impacto, exatamente o comportamento valorizado pela métrica.

## Ensemble entre Cats

Depth 6 e depth 7 apresentaram correlação de erro de 0,994. Um blend cross-fit entre os dois reduziu 19,18% para aproximadamente 19,15% antes da descoberta do finalista com `random_strength=0`. O ganho era pequeno e altamente correlacionado; por isso não foi escolhido como arquitetura final.

## O que foi encerrado e o que ficou pendente

Encerrado com evidência:

- correspondência de 30% não pertence ao especialista barato;
- alvo em log supera o alvo bruto;
- `alpha=2` é o melhor compromisso até R$ 355 mil;
- L2 maior e depth 5 não ajudam;
- `random_strength=0` é importante;
- bagging Bayesiano deve permanecer;
- 400–600 árvores são suficientes no depth 7 final.

Não executado após a solicitação de encerramento:

- repetição do finalista com outras sementes;
- ensemble de sementes do finalista;
- treinamento do finalista em toda a base;
- previsão do conjunto de teste;
- OOF do generalista global 60/40 nos mesmos folds;
- cálculo da vantagem oráculo e da fronteira contra o generalista;
- construção do juiz.

O próximo passo arquitetural correto é gerar o OOF do generalista 60/40 nos mesmos folds e calcular, para cada imóvel, a diferença entre as perdas quadráticas percentuais do generalista e do especialista. Isso definirá o teto do roteamento e o alvo econômico do juiz.

## Artefatos principais

- `resultados/catboost_barato_melhor_config.json`
- `resultados/catboost_baratos_curva_iteracoes_d7_rs0_resumo.csv`
- `resultados/catboost_baratos_curva_iteracoes_d7_rs0_curvas.csv`
- `resultados/catboost_baratos_curva_iteracoes_d7_rs0_folds.csv`
- `resultados/catboost_baratos_curva_iteracoes_d7_rs0_oof.csv`
- `src/buscar_catboost_baratos.py`
- `src/testar_iteracoes_catboost_baratos.py`
