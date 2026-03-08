# Duck Hunt com IA — Implementação passo a passo

Este documento explica como o jogo Duck Hunt foi extendido com uma IA capaz de identificar patos na tela e atirar neles automaticamente, usando o modelo YOLOv5 e TensorFlow.js.

---

## Visão geral da arquitetura

```
main.js
  └── Game (PixiJS)          ← jogo original, gerencia patos, placar, ondas
  └── machine-learning/
        ├── main.js          ← conecta o jogo ao worker, dispara tiros
        ├── layout.js        ← HUD da IA (score e coordenadas)
        ├── worker.js        ← orquestra o pipeline de inferência
        ├── config.js        ← constantes e parâmetros
        ├── model.js         ← carrega e executa o modelo YOLO
        ├── image.js         ← pré-processa frames do jogo
        └── predictions.js   ← filtra e converte saídas do modelo
```

O jogo roda normalmente no thread principal (JavaScript + PixiJS). A IA roda em um **Web Worker** separado para não travar a renderização. A cada 200ms, um frame do jogo é capturado e enviado ao worker, que devolve a coordenada do pato detectado.

---

## Passo 1 — Entendendo o modelo YOLOv5

### O que é YOLO?

YOLO (You Only Look Once) é uma família de modelos de detecção de objetos em tempo real. Ao contrário de abordagens anteriores que varriam a imagem várias vezes, o YOLO analisa a imagem **uma única vez** e já produz todas as detecções.

### Versão usada: YOLOv5n (nano)

O sufixo **n** significa *nano* — a menor e mais rápida variante do YOLOv5. Ideal para rodar no navegador com TensorFlow.js, onde recursos de GPU são limitados.

### Como o modelo processa uma imagem

```
Imagem do jogo (qualquer tamanho)
       │
       ▼
  Redimensionar para 640×640 px   ← tamanho fixo exigido pelo modelo
       │
       ▼
  Normalizar pixels para [0, 1]   ← dividir cada canal RGB por 255
       │
       ▼
  Adicionar dimensão de batch      ← shape [1, 640, 640, 3]
       │
       ▼
  Executar inferência YOLOv5
       │
       ▼
  Saída: boxes, scores, classes    ← uma linha por detecção candidata
```

### Formato de saída do modelo

O modelo retorna três arrays:

| Saída     | Conteúdo                                         |
|-----------|--------------------------------------------------|
| `boxes`   | Coordenadas de cada caixa: `[x1, y1, x2, y2]`   |
| `scores`  | Confiança da detecção para cada caixa (0 a 1)    |
| `classes` | Índice da classe detectada (ex: 16 = kite)       |

Cada elemento nos três arrays corresponde à mesma detecção pelo índice `i`. A caixa `i` tem confiança `scores[i]` e classe `classes[i]`.

### Por que "kite" (pipa)?

O modelo foi treinado no dataset **COCO**, que tem 80 classes. Não existe uma classe "pato" no COCO. A classe mais próxima visualmente aos patos voando no jogo é **kite** (pipa), pois ambos são objetos que se movem no céu. Isso demonstra uma técnica comum em ML: reutilizar um modelo genérico para um domínio relacionado.

---

## Passo 2 — Formato do modelo para o navegador (`model.json`)

O YOLOv5 foi exportado para o formato **TensorFlow.js Graph Model**, que consiste em:

```
yolov5n_web_model/
  ├── model.json              ← arquitetura da rede e metadados
  ├── group1-shard1of2.bin    ← pesos (primeira metade)
  ├── group1-shard2of2.bin    ← pesos (segunda metade)
  └── labels.json             ← mapeamento índice → nome da classe
```

O arquivo `model.json` descreve o grafo computacional. Os `.bin` contêm os pesos treinados, divididos em shards para facilitar o carregamento paralelo pelo navegador.

---

## Passo 3 — Web Worker: por que e como

### O problema sem o Worker

JavaScript é single-threaded. Inferência de ML é pesada (dezenas de operações de álgebra linear). Se rodasse no thread principal, o jogo travaria a cada frame analisado.

### A solução: `new Worker()`

```js
// machine-learning/main.js
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
```

O `Worker` roda em um thread separado. O tipo `'module'` habilita `import/export` dentro do worker (ES modules).

### Comunicação via `postMessage`

A comunicação entre o thread principal e o worker é feita por troca de mensagens. Os dados são **copiados ou transferidos** entre os dois threads.

```
Thread principal                   Worker
      │                               │
      │── postMessage(frame) ────────►│
      │                               │  processa
      │◄─── postMessage(prediction) ──│
```

### Transferable Objects

```js
// main.js — enviando o frame
worker.postMessage({ type: 'predict', image: bitmap }, [bitmap]);
//                                                      ^^^^^^^^
//                                              lista de transferíveis
```

O `ImageBitmap` é um *transferable object*: em vez de copiar os bytes da imagem (custoso), o objeto é **transferido** ao worker — o thread principal perde a referência, o worker a recebe instantaneamente. Isso elimina a cópia de memória.

---

## Passo 4 — Carregamento do modelo (`model.js`)

```js
export async function loadModelAndLabels() {
    await tf.ready();

    const [labels, model] = await Promise.all([
        fetch(LABELS_PATH).then(r => r.json()),
        tf.loadGraphModel(MODEL_PATH),
    ]);
    //   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // Carrega labels e modelo em paralelo — nenhum depende do outro
```

**`tf.ready()`** garante que o backend do TensorFlow.js está inicializado. O TF.js detecta automaticamente o melhor backend disponível: WebGL (GPU via navegador) > WASM > CPU puro. No navegador, geralmente usa WebGL, o que acelera as operações matriciais dramaticamente.

**`tf.loadGraphModel()`** carrega o `model.json` e os shards `.bin`. Um *Graph Model* é um modelo "frozen" — não é treinável, apenas executável. É o formato mais eficiente para inferência.

### Warmup (aquecimento)

```js
const dummyInput = tf.ones(model.inputs[0].shape);
try {
    const warmupOutput = await model.executeAsync(dummyInput);
    tf.dispose(Array.isArray(warmupOutput) ? warmupOutput : [warmupOutput]);
} finally {
    dummyInput.dispose();
}
```

A primeira execução do modelo é sempre mais lenta porque o WebGL precisa compilar os shaders (programas que rodam na GPU). O warmup faz essa compilação acontecer com um tensor fictício, para que a primeira inferência real seja rápida.

O bloco `try/finally` garante que os tensores são sempre descartados, mesmo se `executeAsync` lançar um erro — evitando vazamentos de memória na GPU.

---

## Passo 5 — Pré-processamento da imagem (`image.js`)

```js
export function preprocessImage(input) {
    return tf.tidy(() => {
        const image = tf.browser.fromPixels(input);
        return tf.image
            .resizeBilinear(image, [INPUT_MODEL_DIMENSIONS, INPUT_MODEL_DIMENSIONS])
            .div(255)
            .expandDims(0);
    });
}
```

### `tf.tidy()`

Tensores em TensorFlow.js alocam memória na GPU. Se não forem descartados explicitamente, causam vazamento. `tf.tidy()` registra todos os tensores criados dentro do callback e os descarta automaticamente ao final — **exceto o tensor retornado**, que passa a ser responsabilidade do chamador.

### `tf.browser.fromPixels(input)`

Converte um `ImageBitmap` (ou `canvas`, `video`) em um tensor de shape `[altura, largura, 3]`, onde 3 = canais RGB. Os valores são inteiros de 0 a 255.

### `resizeBilinear`

```js
.resizeBilinear(image, [640, 640])
```

Redimensiona a imagem para 640×640, que é o tamanho de entrada esperado pelo YOLOv5n. A interpolação bilinear é um bom equilíbrio entre qualidade e velocidade — considera os 4 pixels vizinhos ao calcular cada novo pixel.

### Normalização

```js
.div(255)
```

Converte os pixels de `[0, 255]` para `[0.0, 1.0]`. Redes neurais treinadas com imagens normalizadas esperam entradas nessa faixa. Usar valores não normalizados geraria predições completamente erradas.

### `expandDims(0)`

```
Antes:  [640, 640, 3]       ← uma imagem
Depois: [1, 640, 640, 3]    ← batch de uma imagem
```

O modelo espera um **batch** de imagens (dimensão extra no índice 0). Mesmo enviando uma única imagem, é necessário adicionar essa dimensão.

---

## Passo 6 — Inferência (`model.js`)

```js
export async function runInference(model, tensor) {
    const output = await model.executeAsync(tensor);
    try {
        const [boxes, scores, classes] = output;
        const [boxesData, scoresData, classesData] = await Promise.all([
            boxes.data(),
            scores.data(),
            classes.data(),
        ]);
        return { boxes: boxesData, scores: scoresData, classes: classesData };
    } finally {
        tf.dispose(tensor);
        output.forEach(el => el.dispose());
    }
}
```

### `model.executeAsync(tensor)`

Executa a inferência. Retorna um array de tensores de saída. Usamos `executeAsync` (assíncrono) porque o backend WebGL opera de forma assíncrona — os cálculos rodam na GPU e precisamos aguardar a conclusão.

### `.data()`

Baixa os valores do tensor da GPU para a memória principal do JavaScript, retornando um `Float32Array`. Chamamos os três em paralelo com `Promise.all` para não esperar um de cada vez.

### Descarte de tensores no `finally`

O bloco `finally` garante que tanto o tensor de entrada (`tensor`) quanto todos os tensores de saída (`output`) sejam descartados mesmo em caso de erro. Sem isso, cada inferência vazaria memória na GPU até o programa travar.

---

## Passo 7 — Processamento das predições (`predictions.js`)

```js
export function* processPredictions({ boxes, scores, classes }, labels, width, height) {
    for (let i = 0; i < scores.length; i++) {
        if (scores[i] < CLASS_THRESHOLD) continue;       // 1. filtro de confiança
        if (labels[classes[i]] !== TARGET_LABEL) continue; // 2. filtro de classe

        let [x1, y1, x2, y2] = boxes.slice(i * 4, (i + 1) * 4);
        x1 += width;   // 3. ajuste de coordenadas
        x2 += width;
        y1 += height;
        y2 += height;

        yield {
            x: x1 + (x2 - x1) / 2,   // 4. centro da caixa
            y: y1 + (y2 - y1) / 2,
            score: Math.round(scores[i] * 10000) / 100,
        };
    }
}
```

### Filtro de confiança (`CLASS_THRESHOLD = 0.4`)

O modelo gera dezenas de detecções candidatas para cada frame. A maioria tem confiança muito baixa — são falsos positivos. Ignoramos qualquer detecção com score abaixo de 40%.

Ajustar esse threshold é um tradeoff:
- **Muito alto** (ex: 0.8): o modelo só atira quando tem certeza, mas erra muitos patos
- **Muito baixo** (ex: 0.1): atira em tudo, incluindo o céu e o cachorro

### Filtro de classe

```js
if (labels[classes[i]] !== TARGET_LABEL) continue;
```

`classes[i]` é um índice numérico. `labels` é o array de nomes carregado de `labels.json`. Verificamos se o nome corresponde a `'kite'`, descartando todas as outras classes (pessoas, carros, etc.) que o modelo possa detectar no fundo da tela.

### A função geradora (`function*` / `yield`)

Em vez de construir um array de resultados, a função usa o protocolo de iterador do JavaScript. Isso permite processar cada detecção imediatamente conforme ela é gerada, sem alocar memória para o array completo. O chamador consome com `for...of`.

### Cálculo do centro

```js
x: x1 + (x2 - x1) / 2,
y: y1 + (y2 - y1) / 2,
```

O modelo retorna os cantos da caixa delimitadora. Para atirar, precisamos do centro. O ponto de mira é posicionado exatamente no centro do pato detectado.

---

## Passo 8 — Orquestração no Worker (`worker.js`)

```js
const { model, labels } = await loadModelAndLabels();
postMessage({ type: 'model-loaded' });

let _inferring = false;

self.onmessage = async ({ data }) => {
    if (data.type !== 'predict' || _inferring) return;

    _inferring = true;
    const input = preprocessImage(data.image);
    const { width, height } = data.image;

    try {
        const results = await runInference(model, input);
        for (const prediction of processPredictions(results, labels, width, height)) {
            postMessage({ type: 'prediction', ...prediction });
        }
    } finally {
        _inferring = false;
    }
};
```

### Top-level `await`

O `await loadModelAndLabels()` roda antes de `self.onmessage` ser registrado. Isso garante que o modelo está pronto antes de qualquer mensagem poder ser processada — sem race conditions.

### Flag `_inferring`

Se frames chegam mais rápido que a inferência termina (200ms de intervalo, mas inferência pode levar mais), chamadas se sobreporiam, multiplicando o uso de memória GPU e gerando resultados fora de ordem. A flag descarta frames que chegam enquanto o anterior ainda está sendo processado.

---

## Passo 9 — Captura de frames e disparo (`main.js`)

```js
// Captura um frame do jogo a cada 200ms
setInterval(async () => {
    const canvas = game.app.renderer.extract.canvas(game.stage);
    const bitmap = await createImageBitmap(canvas);

    worker.postMessage({ type: 'predict', image: bitmap }, [bitmap]);
}, 200);

// Recebe a predição e atira
worker.onmessage = ({ data }) => {
    if (data.type === 'prediction') {
        game.stage.aim.setPosition(data.x, data.y);
        const position = game.stage.aim.getGlobalPosition();
        game.handleClick({ global: position });
    }
};
```

### `renderer.extract.canvas()`

O PixiJS renderiza o jogo em um contexto WebGL. `extract.canvas()` lê os pixels do framebuffer e os copia para um `<canvas>` 2D — necessário para o `createImageBitmap`.

### `createImageBitmap(canvas)`

Converte o canvas em um `ImageBitmap`, que pode ser transferido ao worker sem cópia de memória.

### `game.handleClick()`

Simula um clique do mouse nas coordenadas retornadas pelo modelo. O jogo trata esse clique exatamente como se fosse um tiro manual do jogador — desconta bala, verifica colisão com patos, atualiza pontuação.

---

## Diagrama do pipeline completo

```
PixiJS render loop (60fps)
       │
       │ a cada 200ms
       ▼
renderer.extract.canvas()          ← captura o frame atual
       │
createImageBitmap()                ← prepara para transferência
       │
postMessage(bitmap) ──────────────► Web Worker
                                         │
                                   preprocessImage()
                                    - fromPixels
                                    - resizeBilinear 640×640
                                    - div(255)
                                    - expandDims
                                         │
                                   runInference()
                                    - model.executeAsync()
                                    - extrai boxes/scores/classes
                                         │
                                   processPredictions()
                                    - filtra score < 0.4
                                    - filtra classe != 'kite'
                                    - calcula centro da caixa
                                         │
postMessage(prediction) ◄───────────────┘
       │
game.stage.aim.setPosition(x, y)
       │
game.handleClick()                 ← tiro automático
```

---

## Conceitos-chave do TensorFlow.js resumidos

| Conceito | O que é | Por que importa |
|----------|---------|-----------------|
| **Tensor** | Array multidimensional na GPU | Unidade básica de dados no TF.js |
| **tf.tidy()** | Escopo de limpeza automática | Evita vazamento de memória GPU |
| **tensor.dispose()** | Descarta tensor manualmente | Necessário para tensores que escapam do tidy |
| **tf.ready()** | Aguarda inicialização do backend | Garante que WebGL está pronto antes de usar |
| **executeAsync()** | Executa inferência assíncrona | O resultado fica na GPU; `.data()` baixa para JS |
| **Graph Model** | Modelo frozen (só inferência) | Mais eficiente que LayersModel para deploy |
| **WebGL backend** | Cálculos na GPU via navegador | 10–100x mais rápido que CPU puro |
| **Warmup** | Primeira inferência com dados fictícios | Compila shaders WebGL antes do uso real |
| **Transferable** | Objeto passado sem cópia entre threads | Elimina overhead de serialização no postMessage |

---

## Parâmetros ajustáveis (`config.js`)

```js
const INPUT_MODEL_DIMENSIONS = 640; // tamanho de entrada do YOLOv5n (não alterar)
const CLASS_THRESHOLD = 0.4;        // confiança mínima para aceitar detecção
const TARGET_LABEL = 'kite';        // classe do COCO mais próxima de "pato voando"
```

O único parâmetro que faz sentido experimentar é `CLASS_THRESHOLD`:
- Diminuir para `0.2` faz a IA atirar mais (mais falsos positivos)
- Aumentar para `0.6` faz a IA ser mais seletiva (pode perder patos rápidos)

O intervalo de captura de frames (200ms em `main.js`) também afeta o comportamento — valores menores aumentam a taxa de detecção mas exigem mais da GPU.
