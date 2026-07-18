# Diagnóstico conservador do especialista caro

## Motivação

O juiz supervisionado do especialista caro obteve 22,32% no Kaggle, piorando
fortemente a referência de 21,19%. Esse resultado invalidou sua seleção interna
como evidência suficiente de generalização.

Para separar falha de roteamento de falha do próprio especialista, foram
criadas duas ablações simples. Nenhum modelo foi retreinado: o especialista é
aplicado somente aos 5% maiores preços previstos pelo pipeline público de
21,19%.

No teste, isso altera exatamente 100 dos 2.000 imóveis. O menor preço previsto
pelo pipeline dentro desse grupo é aproximadamente R$ 1,457 milhão.

## Qualidade interna da seleção

No OOF, o top 5% por fold contém 235 imóveis. Embora a seleção utilize somente
o preço previsto, sua composição real foi:

- 99,15% acima de R$ 1 milhão;
- 96,17% acima de R$ 1,3 milhão;
- 87,66% acima de R$ 1,5 milhão;
- 68,94% acima de R$ 2 milhões.

Isso é muito mais preciso que o juiz anterior e quase elimina a ambiguidade de
faixa.

## Duas intensidades sobre os mesmos imóveis

| Intensidade | RMSPE OOF | Ganho | Folds melhores | RMSPE holdout | Ganho holdout |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50% | 21,0431% | 0,163 p.p. | 5/5 | 20,7319% | 0,052 p.p. |
| 100% | **20,9956%** | **0,210 p.p.** | 5/5 | 20,7517% | 0,033 p.p. |

O bootstrap pareado com 5.000 reamostragens produziu:

| Intensidade | Ganho mediano | IC95 | Probabilidade de ganho |
| ---: | ---: | :--- | ---: |
| 50% | 0,164 p.p. | 0,088 a 0,232 p.p. | 100% |
| 100% | 0,215 p.p. | 0,049 a 0,345 p.p. | 99,52% |

A versão de 50% altera os 100 imóveis selecionados em média em R$ 126.617 e o
conjunto inteiro em R$ 6.331. A versão de 100% altera os selecionados em média
em R$ 253.234 e o conjunto inteiro em R$ 12.662.

## Submissions

Foram geradas:

- `submissions/submission_diagnostico_caros_top05_w050.csv`
- `submissions/submission_diagnostico_caros_top05_w100.csv`

As duas possuem 2.000 IDs únicos, nenhum ausente e nenhum preço não positivo.
Fora dos mesmos 100 IDs selecionados, elas são exatamente iguais à submission
de 21,19%.

A ordem mais segura de envio é 50% primeiro e 100% depois. Os três pontos
públicos — referência, 50% e 100% — permitem interpretar a direção:

- ambas melhoram, com 100% melhor: o especialista funciona e o problema era o
  falso positivo do juiz;
- 50% melhora e 100% piora: a direção funciona, mas precisa de shrinkage ou
  nova calibração;
- ambas pioram: o especialista atual não transfere nem no topo; o próximo passo
  passa a ser redesenhá-lo, provavelmente como correção residual ou treino
  específico acima de R$ 1,3 milhão;
- resultados praticamente iguais: o efeito público nos 100 imóveis é pequeno
  demais para justificar um novo juiz antes de melhorar o especialista.

## Artefatos

- `src/testar_gate_conservador_caros.py`: grade por fração e intensidade;
- `src/gerar_submissions_gate_conservador_caros.py`: geração e auditoria;
- `resultados/gate_caros_conservador_comparacao.csv`: grade OOF/holdout;
- `resultados/gate_caros_diagnostico_teste.csv`: 100 IDs e previsões;
- `resultados/gate_caros_diagnostico_resumo.json`: protocolo consolidado.

Nenhum novo especialista foi treinado. Essa decisão permanece condicionada ao
resultado público destas ablações.

