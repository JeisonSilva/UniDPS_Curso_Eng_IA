import { normalizarIdade, normalizarValor, normalizarIndice } from './normalizacao.js';

export function criarMapaCategorias(produtos) {
    const categorias = [...new Set(produtos.map((produto) => produto.categoria))];
    return new Map(categorias.map((categoria, index) => [categoria, index]));
}

export function criarMapaCores(produtos) {
    const cores = [...new Set(produtos.map((produto) => produto.color))];
    return new Map(cores.map((cor, index) => [cor, index]));
}

export function criarFeatureVector(usuario, produto, categoriasMap, coresMap, valorMaximo) {
    return [
        normalizarIdade(usuario.idade),
        normalizarValor(produto.valor, valorMaximo),
        normalizarIndice(categoriasMap.get(produto.categoria), categoriasMap.size),
        normalizarIndice(coresMap.get(produto.color), coresMap.size)
    ];
}
