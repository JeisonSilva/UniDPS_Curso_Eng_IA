# Projeto 2 — Recomendacao de Produtos com TensorFlow.js

## Objetivo

Treinar um modelo de recomendacao colaborativa simplificada usando TensorFlow.js em Node.js.
O modelo aprende padroes de compra a partir de dados de usuarios e produtos, e gera recomendacoes personalizadas.

## Como funciona

1. **Dados de entrada** — usuarios com idade e historico de compras, produtos com categoria, valor e cor.
2. **Exemplos de treino** — para cada usuario, pares positivos (produto comprado) e negativos (produto nao comprado) sao gerados e normalizados.
3. **Modelo** — rede neural sequencial com duas camadas ocultas (`relu`) e saida sigmoide para prever probabilidade de compra.
4. **Recomendacao** — apos o treino, para cada usuario o modelo pontua todos os produtos ainda nao comprados e retorna os top-N mais provaveis.

## Estrutura

```
projeto2/
├── data/
│   ├── produtos.json   # catalogo de 30 produtos com categoria, valor e cor
│   └── usuarios.json   # 18 usuarios com idade e historico de compras
├── index.js            # treinamento e geracao de recomendacoes
├── jsconfig.json
└── package.json
```

## Features usadas pelo modelo

| Feature             | Descricao                              |
|---------------------|----------------------------------------|
| idadeNormalizada    | idade do usuario dividida por 100      |
| valorNormalizado    | valor do produto dividido pelo maximo  |
| categoriaNormalizada| indice da categoria normalizado (0–1)  |
| corNormalizada      | indice da cor normalizado (0–1)        |

## Arquitetura do modelo

```
Dense(8, relu)  →  Dense(4, relu)  →  Dense(1, sigmoid)
```

- Otimizador: Adam
- Loss: Binary Crossentropy
- Metrica: Accuracy
- Epocas: 100

## Como executar

```bash
cd projeto2
npm install
npm start
```

## Exemplo de saida esperada

```
Treino concluido — loss: 0.3412, accuracy: 0.8750

=== Recomendacoes por usuario ===
Ana: Teclado Mecanico (82.3%) | Mouse Gamer (78.1%) | Smartphone X (71.5%)
Bruno: Luva de Academia (85.0%) | Garrafa Termica (79.2%) | Tapete de Yoga (74.8%)
...
```

## Tecnologias

- Node.js (ES Modules)
- TensorFlow.js (`@tensorflow/tfjs-node`)
