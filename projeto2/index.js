import { readFile } from 'node:fs/promises';
import { criarMapaCategorias, criarMapaCores } from './src/features.js';
import { gerarExemplosTreino, treinarModelo } from './src/treino.js';
import { recomendarProdutos } from './src/recomendacao.js';

async function readJson(path) {
    const content = await readFile(path, 'utf8');
    return JSON.parse(content);
}

async function main() {
    const produtos = await readJson('./data/produtos.json');
    const usuarios = await readJson('./data/usuarios.json');

    const categoriasMap = criarMapaCategorias(produtos);
    const coresMap = criarMapaCores(produtos);
    const valorMaximo = Math.max(...produtos.map((produto) => produto.valor));

    const { features, labels } = gerarExemplosTreino(usuarios, produtos, categoriasMap, coresMap, valorMaximo);

    console.log(`Exemplos de treino: ${features.length} (${labels.filter((l) => l === 1).length} positivos, ${labels.filter((l) => l === 0).length} negativos)`);

    const { model, history } = await treinarModelo(features, labels);

    model.summary();
    console.log(`\nTreino concluido — loss: ${history.history.loss.at(-1).toFixed(4)}, accuracy: ${(history.history.acc?.at(-1) ?? history.history.accuracy?.at(-1)).toFixed(4)}`);

    console.log('\n=== Recomendacoes por usuario ===');
    for (const usuario of usuarios) {
        const recomendacoes = await recomendarProdutos(usuario, produtos, model, categoriasMap, coresMap, valorMaximo, 3);
        const nomes = recomendacoes.map((r) => `${r.produto.nome} (${(r.score * 100).toFixed(1)}%)`);
        console.log(`${usuario.nome}: ${nomes.join(' | ')}`);
    }
}

await main();
