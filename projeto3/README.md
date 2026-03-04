# Projeto 3 — Predicao de Aprovacao de Credito

Cenario de estudo para aprender como **definir regras**, **preparar dados de treino em JSON** e **treinar um modelo de classificacao binaria** com TensorFlow.js.

---

## Cenario

Um banco ficticio precisa decidir automaticamente se aprova ou reprova pedidos de credito pessoal.

Em vez de programar regras `if/else` manualmente, vamos **ensinar uma rede neural** a tomar essa decisao com base em exemplos historicos de clientes.

Cada cliente tem:

| Campo                   | Descricao                                      | Tipo     |
|-------------------------|------------------------------------------------|----------|
| `score_credito`         | Pontuacao de 0 a 1000 (historico financeiro)   | numero   |
| `renda_mensal`          | Renda bruta mensal em reais                    | numero   |
| `divida_atual`          | Total de dividas ativas em reais               | numero   |
| `meses_empregado`       | Meses de emprego continuo                      | numero   |
| `historico_inadimplencia` | 1 se ja caloteou, 0 se nunca                 | binario  |
| `idade`                 | Idade em anos                                  | numero   |
| `aprovado`              | **Label: 1 = aprovado, 0 = reprovado**         | binario  |

---

## Regras de Negocio (de onde vieram os labels)

As regras abaixo foram usadas para rotular os dados em `clientes.json`.
O modelo vai **aprender a replicar essas regras** a partir dos exemplos, sem recebe-las explicitamente.

```
APROVADO se:
  score_credito >= 650  E  divida / renda < 0.35
  score_credito >= 800  (independente de outros fatores)

REPROVADO se:
  score_credito < 350
  score_credito < 500  E  historico_inadimplencia = 1
  meses_empregado < 10  E  score_credito < 600
  renda_mensal < 2000   E  divida_atual > 1000
```

> Entender as regras antes de treinar e essencial: elas definem o que o modelo precisa aprender.

---

## Estrutura dos Dados

```
data/
  clientes.json   -> 30 exemplos rotulados de clientes (features + label)
  regras.json     -> explicacao das regras, campos e dicas de estudo
```

### Exemplo de entrada (clientes.json)

```json
{
  "id": 1,
  "nome": "Ana Lima",
  "idade": 34,
  "renda_mensal": 5800,
  "score_credito": 720,
  "divida_atual": 800,
  "meses_empregado": 36,
  "historico_inadimplencia": 0,
  "aprovado": 1
}
```

---

## Como Implementar o Modelo (guia de estudo)

### Passo 1 — Entender as Regras

Leia `data/regras.json` antes de escrever qualquer codigo.
Pergunte-se: *quais campos mais influenciam a aprovacao?*

### Passo 2 — Normalizar os Dados

Redes neurais funcionam melhor com valores entre 0 e 1.
Cada campo precisa ser normalizado de forma diferente:

```js
// Score: dividir pelo maximo possivel
const scoreNorm = cliente.score_credito / 1000;

// Renda: dividir pelo maximo do dataset
const rendaMax = Math.max(...clientes.map(c => c.renda_mensal));
const rendaNorm = cliente.renda_mensal / rendaMax;

// Razao divida/renda (feature derivada — feature engineering)
const razaoDivida = cliente.divida_atual / cliente.renda_mensal;

// Meses empregado: maximo considerado = 240 (20 anos)
const empregadoNorm = cliente.meses_empregado / 240;

// Historico de inadimplencia: ja e 0 ou 1, usar direto
const inadimplencia = cliente.historico_inadimplencia;

// Idade: dividir por 100
const idadeNorm = cliente.idade / 100;
```

### Passo 3 — Montar as Features e Labels

```js
const features = clientes.map(c => [
  c.score_credito / 1000,
  c.renda_mensal / rendaMax,
  c.divida_atual / c.renda_mensal,   // razao divida/renda
  c.meses_empregado / 240,
  c.historico_inadimplencia,
  c.idade / 100
]);

const labels = clientes.map(c => c.aprovado);
```

### Passo 4 — Criar o Modelo

```js
const model = tf.sequential();

// Camada de entrada: 6 features
model.add(tf.layers.dense({ inputShape: [6], units: 8, activation: 'relu' }));

// Camada oculta
model.add(tf.layers.dense({ units: 4, activation: 'relu' }));

// Camada de saida: 1 neuronio com sigmoid (saida entre 0 e 1)
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

model.compile({
  optimizer: 'adam',
  loss: 'binaryCrossentropy',   // perda para classificacao binaria
  metrics: ['accuracy']
});
```

> **Por que sigmoid na saida?**
> Porque o resultado e binario (aprovado ou nao). O sigmoid converte qualquer valor em uma probabilidade entre 0 e 1.

> **Por que binaryCrossentropy?**
> E a funcao de perda correta para classificacao binaria. Ela penaliza mais quando o modelo erra com alta confianca.

### Passo 5 — Treinar

```js
const inputTensor  = tf.tensor2d(features);
const outputTensor = tf.tensor1d(labels);

await model.fit(inputTensor, outputTensor, {
  epochs: 150,
  callbacks: {
    onEpochEnd: (epoch, logs) => {
      console.log(`Epoca ${epoch + 1}: loss=${logs.loss.toFixed(4)} acc=${logs.acc?.toFixed(4)}`);
    }
  }
});
```

### Passo 6 — Fazer uma Previsao

```js
// Cliente novo (nunca visto pelo modelo)
const clienteNovo = {
  score_credito: 690,
  renda_mensal: 4800,
  divida_atual: 900,
  meses_empregado: 28,
  historico_inadimplencia: 0,
  idade: 35
};

const featureNova = tf.tensor2d([[
  clienteNovo.score_credito / 1000,
  clienteNovo.renda_mensal / rendaMax,
  clienteNovo.divida_atual / clienteNovo.renda_mensal,
  clienteNovo.meses_empregado / 240,
  clienteNovo.historico_inadimplencia,
  clienteNovo.idade / 100
]]);

const previsao = model.predict(featureNova);
const probabilidade = (await previsao.data())[0];

console.log(`Probabilidade de aprovacao: ${(probabilidade * 100).toFixed(1)}%`);
console.log(`Decisao: ${probabilidade >= 0.5 ? 'APROVADO' : 'REPROVADO'}`);
```

---

## Conceitos Trabalhados

| Conceito               | Onde aparece neste projeto                              |
|------------------------|---------------------------------------------------------|
| **Regras de negocio**  | Arquivo `regras.json` — base para os labels             |
| **Feature engineering**| Razao divida/renda calculada a partir de dois campos    |
| **Normalizacao**       | Cada campo tem sua propria escala                       |
| **Classificacao binaria** | Saida 0 ou 1, funcao sigmoid + binaryCrossentropy  |
| **Overfitting**        | Dataset pequeno (30 exemplos) — observe a acuracia      |
| **Inferencia**         | Prever para um cliente nunca visto                      |

---

## Experimentos Sugeridos

Depois de implementar o modelo basico, tente:

1. **Remover o campo `score_credito`** das features e treinar novamente. Quanto a acuracia cai?

2. **Adicionar mais clientes** no `clientes.json` (50, 100...). O modelo melhora?

3. **Mudar a arquitetura**: adicionar mais camadas ou mais neuronios. Qual o impacto na loss?

4. **Testar um cliente de borda**: score 599, divida/renda = 0.33, 11 meses empregado, sem inadimplencia. O modelo aprova ou reprova? Faz sentido pelas regras?

5. **Ajustar o threshold**: em vez de `>= 0.5`, use `>= 0.6` ou `>= 0.4`. Como isso afeta falsos positivos e falsos negativos?

---

## Papel deste Projeto no Repositorio

Este projeto foca em **entender como as regras definem o que o modelo aprende**.
Diferente dos projetos anteriores, aqui os dados e as regras estao explicitamente documentados, permitindo que voce valide manualmente se o modelo esta acertando e por que.
