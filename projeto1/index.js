import * as tf from '@tensorflow/tfjs';

await tf.setBackend('cpu');
await tf.ready();



async function trainModel(input, outPut) {
    const model = tf.sequential()

    model.add(tf.layers.dense({ inputShape: [6], units: 16, activation: "elu" }))
    model.add(tf.layers.dense({ units: 3, activation: "softmax" }))

    model.compile({
        optimizer: "adam",
        loss: "categoricalCrossentropy",
        metrics: ['accuracy']
    })

    await model.fit(
        input,
        outPut,
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
            callbacks: {
                onEpochEnd: async (epoch, log) => {
                    console.log(`epoch: ${epoch + 1}: loss = ${log.loss}`)
                }
            }
        }
    )

    return model
}

async function predict(model, pessoa) {
    const tfInput = tf.tensor2d(pessoa)

    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray
}

function normalizePessoa(pessoa) {
    const idadeMin = 25
    const idadeMax = 40

    const idadeNormalizada =
        Number(((pessoa.idade - idadeMin) / (idadeMax - idadeMin)).toFixed(2))
    const corAzul = pessoa.cor === "azul" ? 1 : 0
    const corVermelho = pessoa.cor === "vermelho" ? 1 : 0
    const corVerde = pessoa.cor === "verde" ? 1 : 0
    const localizacaoSaoPaulo = pessoa.localizacao === "Sao Paulo" ? 1 : 0
    const localizacaoRio = pessoa.localizacao === "Rio de Janeiro" ? 1 : 0

    return [[
        idadeNormalizada,
        corAzul,
        corVermelho,
        corVerde,
        localizacaoSaoPaulo,
        localizacaoRio
    ]]
}

function encodeCategoria(categoria) {
    const categorias = {
        premium: [1, 0, 0],
        medium: [0, 1, 0],
        basic: [0, 0, 1]
    }

    return categorias[categoria]
}

function exibirReferenciasTreinamento(pessoasReais, tensores, categorias, labels) {
    const referencias = pessoasReais.map((pessoa, index) => {
        const categoriaIndex = categorias[index].findIndex((valor) => valor === 1)

        return {
            idade: pessoa.idade,
            cor: pessoa.cor,
            localizacao: pessoa.localizacao,
            dadosNormalizados: JSON.stringify(tensores[index]),
            categoria: labels[categoriaIndex]
        }
    })

    console.log("Referencias de treinamento:")
    console.table(referencias)
}

function exibirNovaPessoa(dadosReais, dadosNormalizados) {
    console.log("Nova pessoa:")
    console.table([{
        idade: dadosReais.idade,
        cor: dadosReais.cor,
        localizacao: dadosReais.localizacao,
        dadosNormalizados: JSON.stringify(dadosNormalizados[0])
    }])
}

function exibirResultadoFormatado(resultados) {
    console.log("Resultado da previsao:")
    console.table(
        resultados.map((resultado, index) => ({
            posicao: index + 1,
            classificacao: resultado.label,
            probabilidade: `${(resultado.prob * 100).toFixed(2)}%`
        }))
    )
}


// Dados de treinamento
const pessoasTreinamentoReais = [
    { idade: 30, cor: "azul", localizacao: "Sao Paulo", categoria: "premium" },
    { idade: 27, cor: "azul", localizacao: "Sao Paulo", categoria: "premium" },
    { idade: 29, cor: "azul", localizacao: "Sao Paulo", categoria: "premium" },
    { idade: 31, cor: "azul", localizacao: "Sao Paulo", categoria: "premium" },
    { idade: 32, cor: "azul", localizacao: "Sao Paulo", categoria: "premium" },
    { idade: 25, cor: "vermelho", localizacao: "Rio de Janeiro", categoria: "medium" },
    { idade: 26, cor: "vermelho", localizacao: "Rio de Janeiro", categoria: "medium" },
    { idade: 28, cor: "vermelho", localizacao: "Rio de Janeiro", categoria: "medium" },
    { idade: 29, cor: "vermelho", localizacao: "Rio de Janeiro", categoria: "medium" },
    { idade: 33, cor: "vermelho", localizacao: "Rio de Janeiro", categoria: "medium" },
    { idade: 40, cor: "verde", localizacao: "Sao Paulo", categoria: "basic" },
    { idade: 38, cor: "verde", localizacao: "Sao Paulo", categoria: "basic" },
    { idade: 36, cor: "verde", localizacao: "Sao Paulo", categoria: "basic" },
    { idade: 35, cor: "verde", localizacao: "Sao Paulo", categoria: "basic" },
    { idade: 39, cor: "verde", localizacao: "Sao Paulo", categoria: "basic" }
]

const tensorPessoas = pessoasTreinamentoReais
    .map((pessoa) => normalizePessoa(pessoa)[0])

const tensorCategoriaPessoa = pessoasTreinamentoReais
    .map((pessoa) => encodeCategoria(pessoa.categoria))

const novaPessoaDadosReais = {
    idade: 28,
    cor: "azul",
    localizacao: "Sao Paulo"
}

const novaPessoa = normalizePessoa(novaPessoaDadosReais)
const inputTensorX = tf.tensor2d(tensorPessoas)
const outTensorY = tf.tensor2d(tensorCategoriaPessoa)

const model = await trainModel(inputTensorX, outTensorY)
const result = await predict(model, novaPessoa)

const labelsNomes = ["premium", "medium", "basic"]

exibirReferenciasTreinamento(
    pessoasTreinamentoReais,
    tensorPessoas,
    tensorCategoriaPessoa,
    labelsNomes
)

const resultFormatado = result[0]
    .map((prob, index) => ({ index, prob }))
    .sort((a, b) => b.prob - a.prob)
    .map((p) => ({
        label: labelsNomes[p.index],
        prob: p.prob
    }))

console.log("\n")
exibirNovaPessoa(novaPessoaDadosReais, novaPessoa)
exibirResultadoFormatado(resultFormatado)
