import tf from '@tensorflow/tfjs';
import { criarFeatureVector } from './features.js';
import { criarModelo } from './modelo.js';

export function gerarExemplosTreino(usuarios, produtos, categoriasMap, coresMap, valorMaximo) {
    const produtosPorId = new Map(produtos.map((produto) => [produto.id, produto]));
    const features = [];
    const labels = [];

    for (const usuario of usuarios) {
        const produtosComprados = new Set(usuario.produtos);
        const produtosNaoComprados = produtos.filter((produto) => !produtosComprados.has(produto.id));

        for (const produtoId of usuario.produtos) {
            const produto = produtosPorId.get(produtoId);
            if (!produto) continue;
            features.push(criarFeatureVector(usuario, produto, categoriasMap, coresMap, valorMaximo));
            labels.push(1);
        }

        const quantidadeNegativos = Math.min(Math.max(usuario.produtos.length, 1), produtosNaoComprados.length);
        for (const produto of produtosNaoComprados.slice(0, quantidadeNegativos)) {
            features.push(criarFeatureVector(usuario, produto, categoriasMap, coresMap, valorMaximo));
            labels.push(0);
        }
    }

    return { features, labels };
}

export async function treinarModelo(features, labels) {
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

    inputTensor.dispose();
    outputTensor.dispose();

    return { model, history };
}
