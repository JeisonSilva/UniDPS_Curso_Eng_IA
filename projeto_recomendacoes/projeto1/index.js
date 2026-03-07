import * as tf from '@tensorflow/tfjs';
import { trainModel , pessoasTreinamentoReais} from './trainModel.js'
import {normalizePessoa, 
    encodeCategoria, 
    exibirReferenciasTreinamento, 
    exibirNovaPessoa, 
    exibirResultadoFormatado,
    labelsNomes} from './format_data.js'

import { predict } from './predict.js';


await tf.setBackend('cpu');
await tf.ready();


const tensorPessoas = pessoasTreinamentoReais
    .map((pessoa) => normalizePessoa(pessoa)[0])

const tensorCategoriaPessoa = pessoasTreinamentoReais
    .map((pessoa) => encodeCategoria(pessoa.categoria))

const inputTensorX = tf.tensor2d(tensorPessoas)
const outTensorY = tf.tensor2d(tensorCategoriaPessoa)

const model = await trainModel(inputTensorX, outTensorY)


const novaPessoaDadosReais = {
    idade: 28,
    cor: "azul",
    localizacao: "Sao Paulo"
}
const novaPessoa = normalizePessoa(novaPessoaDadosReais)
const result = await predict(model, novaPessoa)


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
