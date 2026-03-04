import { readFile } from 'node:fs/promises';
import tf from '@tensorflow/tfjs';

async function readJson(path) {
    const content = await readFile(path, 'utf8');
    return JSON.parse(content);
}

function normalizarIdade(idade) {
    return idade / 100;
}

function normalizarValor(valor, valorMaximo) {
    return valor / valorMaximo;
}

function criarMapaCategorias(produtos) {
    const categorias = [...new Set(produtos.map((produto) => produto.categoria))];
    return new Map(categorias.map((categoria, index) => [categoria, index]));
}

function criarMapaCores(produtos) {
    const cores = [...new Set(produtos.map((produto) => produto.color))];
    return new Map(cores.map((cor, index) => [cor, index]));
}

function normalizarIndice(indice, total) {
    if (total <= 1) {
        return 0;
    }

    return indice / (total - 1);
}

function criarExemploNormalizado(usuario, produto, categoriasMap, coresMap, valorMaximo, label) {
    return {
        usuarioId: usuario.id,
        produtoId: produto.id,
        idadeNormalizada: normalizarIdade(usuario.idade),
        valorNormalizado: normalizarValor(produto.valor, valorMaximo),
        categoriaNormalizada: normalizarIndice(
            categoriasMap.get(produto.categoria),
            categoriasMap.size
        ),
        corNormalizada: normalizarIndice(
            coresMap.get(produto.color),
            coresMap.size
        ),
        label
    };
}

function criarModelo(quantidadeFeatures) {
    const model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [quantidadeFeatures],
        units: 8,
        activation: 'relu'
    }));

    model.add(tf.layers.dense({
        units: 4,
        activation: 'relu'
    }));

    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));

    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

async function main() {
    const produtos = await readJson('./data/produtos.json');
    const usuarios = await readJson('./data/usuarios.json');

    const categoriasMap = criarMapaCategorias(produtos);
    const coresMap = criarMapaCores(produtos);
    const valorMaximo = Math.max(...produtos.map((produto) => produto.valor));
    const produtosPorId = new Map(produtos.map((produto) => [produto.id, produto]));

    const exemplosTreino = [];

    usuarios.forEach((usuario) => {
        const produtosComprados = new Set(usuario.produtos);
        const produtosNaoComprados = produtos.filter(
            (produto) => !produtosComprados.has(produto.id)
        );

        usuario.produtos.forEach((produtoId) => {
            const produtoComprado = produtosPorId.get(produtoId);

            if (!produtoComprado) {
                return;
            }

            exemplosTreino.push(
                criarExemploNormalizado(
                    usuario,
                    produtoComprado,
                    categoriasMap,
                    coresMap,
                    valorMaximo,
                    1
                )
            );
        });

        const quantidadeNegativos = Math.min(
            Math.max(usuario.produtos.length, 1),
            produtosNaoComprados.length
        );

        produtosNaoComprados
            .slice(0, quantidadeNegativos)
            .forEach((produtoNaoComprado) => {
                exemplosTreino.push(
                    criarExemploNormalizado(
                        usuario,
                        produtoNaoComprado,
                        categoriasMap,
                        coresMap,
                        valorMaximo,
                        0
                    )
                );
            });
    });

    const features = exemplosTreino.map((exemplo) => [
        exemplo.idadeNormalizada,
        exemplo.valorNormalizado,
        exemplo.categoriaNormalizada,
        exemplo.corNormalizada
    ]);

    const labels = exemplosTreino.map((exemplo) => exemplo.label);

    const inputTensor = tf.tensor2d(features);
    const outputTensor = tf.tensor1d(labels);
    const model = criarModelo(features[0].length);

    const history = await model.fit(inputTensor, outputTensor, {
        epochs: 100,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(
                    `Epoca ${epoch + 1}: loss=${logs.loss.toFixed(4)} accuracy=${logs.acc?.toFixed(4) ?? logs.accuracy?.toFixed(4)}`
                );
            }
        }
    });

    const usuarioTeste = usuarios[0];
    const produtoTeste = produtosPorId.get(10);
    const exemploTeste = criarExemploNormalizado(
        usuarioTeste,
        produtoTeste,
        categoriasMap,
        coresMap,
        valorMaximo,
        0
    );
    const featureTeste = [[
        exemploTeste.idadeNormalizada,
        exemploTeste.valorNormalizado,
        exemploTeste.categoriaNormalizada,
        exemploTeste.corNormalizada
    ]];
    const tensorTeste = tf.tensor2d(featureTeste);
    const previsao = model.predict(tensorTeste);
    const probabilidadeCompra = (await previsao.data())[0];


    console.log('Categorias mapeadas:', Object.fromEntries(categoriasMap));
    console.log('Cores mapeadas:', Object.fromEntries(coresMap));
    console.log('Primeiros exemplos de treino:', exemplosTreino.slice(0, 10));
    console.log('Primeiras features:', features.slice(0, 5));
    console.log('Primeiras labels:', labels.slice(0, 5));
    console.log('Total de exemplos:', exemplosTreino.length);
    console.log('Total de positivos:', labels.filter((label) => label === 1).length);
    console.log('Total de negativos:', labels.filter((label) => label === 0).length);
    console.log('Shape inputTensor:', inputTensor.shape);
    console.log('Shape outputTensor:', outputTensor.shape);
    console.log('Modelo compilado com sucesso');
    model.summary();
    console.log('Ultima loss:', history.history.loss.at(-1));
    console.log('Ultima accuracy:', history.history.acc?.at(-1) ?? history.history.accuracy?.at(-1));
    console.log('Usuario teste:', usuarioTeste.nome);
    console.log('Produto teste:', produtoTeste.nome);
    console.log('Feature de teste:', featureTeste[0]);
    console.log('Probabilidade prevista de compra:', probabilidadeCompra);
    console.log('Backend TensorFlow:', tf.getBackend());
}

await main();
