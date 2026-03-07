import tf from '@tensorflow/tfjs';
import { criarFeatureVector } from './features.js';

export async function recomendarProdutos(usuario, produtos, model, categoriasMap, coresMap, valorMaximo, topN = 5) {
    const produtosComprados = new Set(usuario.produtos);
    const candidatos = produtos.filter((produto) => !produtosComprados.has(produto.id));

    if (candidatos.length === 0) {
        return [];
    }

    const features = candidatos.map((produto) =>
        criarFeatureVector(usuario, produto, categoriasMap, coresMap, valorMaximo)
    );

    const tensor = tf.tensor2d(features);
    const previsoes = model.predict(tensor);
    const scores = await previsoes.data();
    tensor.dispose();
    previsoes.dispose();

    return candidatos
        .map((produto, i) => ({ produto, score: scores[i] }))
        .sort((a, b) => b.score - a.score)
        .slice(0, topN);
}
