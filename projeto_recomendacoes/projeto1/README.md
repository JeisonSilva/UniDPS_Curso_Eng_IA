# Projeto 1

Projeto introdutorio de classificacao com TensorFlow.js em Node.js.
O experimento utiliza dados ficticios de pessoas para treinar uma rede neural capaz de prever em qual categoria uma nova entrada se encaixa: `premium`, `medium` ou `basic`.

## Visao geral

Este projeto foi desenvolvido como exercicio pratico para compreender as etapas basicas de um fluxo de machine learning aplicado em ambiente JavaScript.
O foco esta em preparar dados, treinar um modelo simples e interpretar a previsao gerada.

## Objetivo

Demonstrar, de forma simples e direta, como utilizar `@tensorflow/tfjs-node` para:

1. estruturar um conjunto de dados;
2. transformar entradas em formato numerico;
3. treinar um modelo de classificacao;
4. realizar inferencia sobre novos dados.

## Como funciona

1. Define um pequeno conjunto de dados de treinamento com idade, cor, localizacao e categoria.
2. Normaliza e codifica esses dados para que possam ser usados pelo modelo.
3. Treina uma rede neural com TensorFlow.js.
4. Faz a previsao para uma nova pessoa.
5. Exibe no terminal as referencias de treinamento e as probabilidades previstas para cada categoria.

## Stack utilizada

- Node.js
- TensorFlow.js com `@tensorflow/tfjs-node`
- JavaScript ES Modules

## Estrutura do projeto

- `index.js`: executa o fluxo completo de treino e previsao.
- `trainModel.js`: define o modelo e contem os dados de treinamento.
- `format_data.js`: normaliza os dados de entrada e formata a exibicao dos resultados.
- `predict.js`: executa a previsao com o modelo treinado.
- `tensorFlow.ipynb`: notebook usado como apoio para estudos ou experimentos.

## Aprendizados trabalhados

- representacao numerica de dados categoricos;
- normalizacao de atributos;
- treinamento supervisionado com classificacao multiclasse;
- leitura e interpretacao de probabilidades de saida de um modelo.

## Papel deste projeto no repositorio

Este projeto representa uma introducao pratica ao uso de ferramentas de IA dentro do ecossistema JavaScript, servindo como base para estudos mais avancados em treinamento, avaliacao e aplicacao de modelos.
