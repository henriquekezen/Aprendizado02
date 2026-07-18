# Juiz entre generalista e especialista barato

## Arquitetura congelada

O generalista é o blend que obteve 21,45% no Kaggle:

- 60% CatBoost alpha 1,5;
- 20% XGBoost alpha 1,5;
- 20% LightGBM alpha 1,5;
- correspondência estrutural de 30% em todos os componentes.

O especialista é o CatBoost dedicado aos baratos:

- alvo `log1p(preco)`;
- `sample_weight = 1 / preco^2`;
- profundidade 7, 600 iterações e `random_strength=0`;
- nenhuma correspondência estrutural;
- fator de calibração final 0,914489.

O juiz escolhido é um `CatBoostClassifier` que aprende se o especialista
produziu menor perda percentual quadrática que o generalista. A classe é
ponderada pelo valor absoluto da diferença de perda; assim, decisões com
grande impacto no RMSPE recebem mais importância que trocas praticamente
neutras. Suas entradas incluem os atributos originais, as duas previsões,
a discordância entre elas, os componentes do generalista e a existência de
correspondência estrutural.

A previsão final para uma intensidade `lambda` é:

```text
previsao = generalista + lambda * peso_do_juiz * (especialista - generalista)
```

As intensidades geradas estão entre zero e um, portanto a previsão sempre
permanece entre os dois modelos-base.

## Reconstrução e teto do roteamento

O generalista foi reconstruído nos mesmos cinco folds do CatBoost. Seu RMSPE
OOF foi 21,4066%, praticamente igual aos 21,45% públicos, o que dá confiança
de que a validação representa o pipeline submetido.

Um oráculo que escolhe o melhor modelo em cada linha chegaria a 17,9545%, mas
esse número é apenas um teto inalcançável. Um corte usando o preço verdadeiro
em R$ 328,6 mil chegaria a 20,3845%. Já um corte pelo preço previsto do
generalista não melhorou o resultado: imóveis caros severamente subestimados
contaminam a faixa de menor previsão. Por isso o juiz usa várias evidências e
não apenas um limite de preço previsto.

## Comparação cross-fit dos juízes

Cada linha foi avaliada por um juiz que não a utilizou no treinamento. Os
resultados foram:

| Juiz | RMSPE cross-fit | Peso médio no especialista |
| --- | ---: | ---: |
| Classificador de utilidade | **21,3394%** | 19,43% |
| Regressor da diferença de perda | 21,3600% | 22,97% |
| Classificador de imóvel barato | 21,3735% | 8,43% |
| Classificador de barato ponderado por utilidade | 21,3793% | 8,24% |
| Corte suave pelo preço previsto | 21,4212% | 11,48% |
| Generalista sem juiz | 21,4066% | 0% |

O classificador de utilidade foi o vencedor. O ganho é pequeno, mas o teste
mostra que aprender diretamente a decisão útil é superior a tentar primeiro
classificar a faixa de preço.

## Auditoria por bloco de IDs

Antes do treino final, o juiz foi treinado fora dos 20% menores IDs e testado
nesse bloco. Os modelos-base também foram treinados sem essas linhas.

| Intensidade | RMSPE total | Até R$ 355 mil | Acima de R$ 355 mil |
| ---: | ---: | ---: | ---: |
| 0,00 | 21,1745% | 22,3921% | 20,7503% |
| 0,40 | 21,0787% | 21,9572% | 20,7759% |
| 0,70 | 21,0308% | 21,6506% | 20,8189% |
| 1,00 | **21,0035%** | **21,3614%** | 20,8822% |

O juiz reproduziu fora do KFold o comportamento desejado: ganho forte nos
baratos e pequena concessão nos caros, com melhora líquida de 0,171 ponto
percentual. O ótimo quadrático desse holdout estaria em intensidade 1,245,
enquanto o OOF aponta 0,908. As três submissions foram limitadas a 1,00 para
evitar extrapolar além da previsão do especialista antes de observar o
Kaggle.

## Três submissions

Com o mapeamento final do juiz, os resultados internos por intensidade são:

| Intensidade | RMSPE OOF | Peso médio efetivo no especialista |
| ---: | ---: | ---: |
| 0,00 | 21,4066% | 0,00% |
| 0,40 | 21,3507% | 6,93% |
| 0,70 | 21,3294% | 12,12% |
| 1,00 | **21,3260%** | 17,31% |

Foram gerados:

- `submissions/submission_juiz_utilidade_lambda040.csv`
- `submissions/submission_juiz_utilidade_lambda070.csv`
- `submissions/submission_juiz_utilidade_lambda100.csv`

No teste oficial, o peso-base médio do juiz foi 16,97%, a mediana 12,82% e o
percentil 90 36,58%. Essas estatísticas estão próximas das observadas em OOF,
sem sinal de deslocamento importante. Apenas 3,15% das linhas receberam peso
base maior que 50% no especialista; o roteamento final é deliberadamente
suave.

Os três CSVs têm 2.000 IDs únicos, não contêm ausentes ou preços não positivos
e foram verificados contra a fórmula do blend com erro máximo inferior a meio
centavo antes do arredondamento. As previsões intermediárias estão em
`resultados/juiz_especialista_previsoes_teste.csv` e o resumo reproduzível em
`resultados/juiz_especialista_submissions_resumo.json`.

## Leitura dos próximos resultados públicos

O score do generalista em `lambda=0` já é conhecido: 21,45%. Como a previsão
é linear em `lambda`, o quadrado do RMSPE público também forma uma parábola em
`lambda`. Os três novos scores permitem verificar a estabilidade e estimar a
intensidade pública ótima. Só depois disso vale considerar intensidades acima
de 1,00 ou redesenhar o juiz; não é necessário retreinar os modelos-base.

## Resultado público do juiz binário

As três intensidades confirmaram a ordenação interna:

| Intensidade | RMSPE OOF | Kaggle |
| ---: | ---: | ---: |
| 0,40 | 21,3507% | 21,38% |
| 0,70 | 21,3294% | 21,34% |
| 1,00 | 21,3260% | **21,33%** |

Com o generalista em 21,45%, a parábola ajustada aos quatro pontos públicos
indica ótimo próximo de intensidade 1,08 e score estimado de 21,328%. Portanto,
novos ajustes de intensidade do juiz binário não devem produzir melhora
visível no placar com duas casas decimais.

## Último experimento: juiz contínuo alinhado ao RMSPE

O juiz binário aprende apenas se o especialista vence. O último experimento
passou a estimar diretamente quanto do especialista usar. Para generalista
`G`, especialista `S`, diferença `D=S-G`, preço real `y` e peso `w`:

```text
((G + w*D - y) / y)² = (D/y)² * (w - (y-G)/D)²
```

Foram comparadas duas regressões CatBoost, sempre cross-fit:

- pseudo-alvo exato `(y-G)/D`, ponderado por `(D/y)²`;
- o mesmo pseudo-alvo projetado para `[0,1]`, com a mesma ponderação.

O peso previsto passou por uma transformação afim e projeção para `[0,1]`.
Os parâmetros dessa transformação também foram escolhidos somente fora do
fold avaliado.

| Formulação | Peso direto | Peso calibrado cross-fit |
| --- | ---: | ---: |
| Alvo exato | 21,3510% | 21,3467% |
| Alvo projetado em `[0,1]` | 21,6097% | **21,2873%** |

O alvo projetado superou o juiz binário de 21,3394% nos cinco folds. Os ganhos
por fold foram 0,0737, 0,0597, 0,0447, 0,0122 e 0,0683 ponto percentual. Isso
reduz a chance de a melhora de 0,052 ponto na média ter vindo de um único
recorte favorável.

No holdout por IDs, a comparação foi:

| Modelo | RMSPE total | Até R$ 355 mil | Acima de R$ 355 mil |
| --- | ---: | ---: | ---: |
| Juiz binário | 21,0035% | 21,3614% | **20,8822%** |
| Juiz contínuo | **20,9536%** | **20,9656%** | 20,9495% |

O contínuo ganhou mais 0,050 ponto no total. Ele melhorou fortemente os baratos
e aceitou uma piora pequena nos caros, preservando o desenho do especialista.

Como os critérios prévios foram satisfeitos — no máximo 21,30% cross-fit e
melhora no holdout — foi gerada uma única submission:

- `submissions/submission_juiz_continuo_rmspe.csv`

O mapeamento final usa `clip(-0,45 + 1,50 * score, 0, 1)`. No teste, o peso
médio do especialista é 18,09%, a mediana 17,48%, o percentil 90 35,00%, 14%
das linhas ficam em peso zero e nenhuma recebe peso um. O RMSPE OOF usando os
parâmetros finais é 21,2764%; a estimativa mais conservadora continua sendo o
21,2873% inteiramente cross-fit.

O script experimental é `src/testar_juiz_continuo.py`, o treino final está em
`src/gerar_submission_juiz_continuo.py` e todas as previsões de teste foram
preservadas em `resultados/juiz_continuo_previsoes_teste.csv`.

O juiz contínuo obteve **21,30% no Kaggle**, contra 21,33% do juiz binário. A
diferença pública de 0,03 ponto ficou próxima do ganho cross-fit conservador de
0,052 ponto e confirmou novamente a utilidade da validação interna.

## Segundo juiz: CatBoost global versus árvores

Depois de fixar o peso `q` do especialista barato, ainda restava uma proporção
fixa de 60% CatBoost e 40% árvores dentro de `1-q`. O segundo juiz passou a
estimar `r`, a fração de árvores nessa parcela:

```text
previsao = q * especialista
         + (1-q) * ((1-r) * CatBoost + r * arvores)
```

O pseudo-alvo ótimo de `r` foi treinado com a mesma derivação alinhada ao
RMSPE e projetado para `[0,1]`. Foram comparados um mapeamento livre e outro
dinâmico cuja média permanece ancorada em 48,745% de árvores. Essa âncora veio
da parábola pública formada pelo CatBoost de 21,82%, árvores de 21,86% e blend
60/40 de 21,45%.

| Estratégia | RMSPE cross-fit | Holdout por IDs | Fração média de árvores |
| --- | ---: | ---: | ---: |
| Proporção atual, 40% | 21,2873% | 20,9536% | 40,00% |
| Estática na âncora pública | 21,2407% | 20,8594% | 48,75% |
| Dinâmica ancorada | **21,2058%** | **20,7844%** | 48,69% OOF |
| Dinâmica livre | 21,1865% | 20,6853% | 66,13% OOF |

As duas versões dinâmicas melhoraram os cinco folds e o holdout. Apesar de a
livre ser 0,019 ponto melhor no OOF, ela desloca globalmente o peso para 66% de
árvores, em conflito com a evidência pública. Foi escolhida a versão ancorada,
que preserva quase todo o ganho dinâmico sem depender do viés CatBoost/árvores
observado apenas internamente.

No teste oficial, a fração média de árvores dentro do restante ficou em
48,730%. Os pesos efetivos médios finais são:

- 18,09% especialista barato;
- 41,90% CatBoost global;
- 40,00% blend XGBoost/LightGBM.

O novo peso não é apenas global: o percentil 10 de `r` é 35,85%, a mediana
49,05% e o percentil 90 60,33%. A submission gerada foi:

- `submissions/submission_juiz_componentes_ancorado.csv`

Seu RMSPE cross-fit é 21,2058%; com o mapeamento final treinado em todo o OOF,
21,1993%. O experimento está em `src/testar_juiz_componentes.py`, o treino
final em `src/gerar_submission_juiz_componentes.py` e as previsões completas
em `resultados/juiz_componentes_previsoes_teste.csv`.

O juiz de componentes ancorado obteve **21,19% no Kaggle**, contra 21,30% do
juiz contínuo anterior. Como os dois arquivos pertencem à mesma direção de
previsão, a resposta pública foi usada para um último ajuste de intensidade.
Com a curvatura observada no OOF e no holdout, o ótimo público foi estimado
entre 1,70 e 1,95. Foi escolhida uma única intensidade intermediária de 1,75:

```text
r_final = clip(0,40 + 1,75 * (r_ancorado - 0,40), 0, 1)
```

Esse ajuste preserva o peso do especialista barato e intensifica somente a
decisão CatBoost/árvores. Seus resultados internos são:

| Avaliação | Ancorado | Intensidade 1,75 |
| --- | ---: | ---: |
| OOF | 21,2058% | **21,1972%** |
| Holdout por IDs | 20,7844% | **20,7128%** |

No teste, a fração média de árvores dentro de `1-q` passa de 48,73% para
55,02%. Os pesos efetivos médios ficam em 18,09% para o especialista barato,
36,71% para o CatBoost global e 45,20% para as árvores. Apenas 2,20% das linhas
atingem os limites zero ou um. Foi gerada somente:

- `submissions/submission_juiz_componentes_intensidade175.csv`

O script reproduzível está em
`src/gerar_submission_juiz_componentes_intensidade175.py` e o resumo em
`resultados/juiz_componentes_intensidade175_resumo.json`.
