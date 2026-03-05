import tf from "@tensorflow/tfjs-node"
import {readFile} from "node:fs/promises"

function calcularMeiaIdade(clientes, cliente) {
    const maximo = Math.max(...clientes.map(c => c.idade))
    const minimo = Math.min(...clientes.map(c => c.idade))
    const media = (cliente.idade - minimo) / (maximo - minimo)
    return media
}

function calcularMediaRenda(clientes, cliente) {
    const maximo = Math.max(...clientes.map(c => c.renda_mensal))
    const minimo = Math.min(...clientes.map(c => c.renda_mensal))
    const media = (cliente.renda_mensal - minimo) / (maximo - minimo)
    return media
}

function calcularScoreCredito(clientes, cliente) {
    const maximo = Math.max(...clientes.map(c => c.score_credito))
    const minimo = Math.min(...clientes.map(c => c.score_credito))
    const media = (cliente.score_credito - minimo) / (maximo - minimo)
    return media
}

function calcularDividaAtual(clientes, cliente) {
    const maximo = Math.max(...clientes.map(c => c.divida_atual))
    const minimo = Math.min(...clientes.map(c => c.divida_atual))
    const media = (cliente.divida_atual - minimo) / (maximo - minimo)
    return media
}

function calcularMesesEmpregado(clientes, cliente) {
    const maximo = Math.max(...clientes.map(c => c.meses_empregado))
    const minimo = Math.min(...clientes.map(c => c.meses_empregado))
    const media = (cliente.meses_empregado - minimo) / (maximo - minimo)
    return media
}


function criarFeature(dadosNormalizados) {
    return [...dadosNormalizados.map(media => {
        return [
            media.mediaIdade,
            media.mediaRenda,
            media.mediaScoreCredito,
            media.mediaDividaAtual,
            media.mediaMesesEmpregado,
            media.mediaHistoricoInadimplencia
        ]
    })]
}



function normalizarDados(clientes) {

    const clientesNormalizados = clientes.map(cliente =>{
    const mediaIdade = calcularMeiaIdade(clientes, cliente)
    const mediaRenda = calcularMediaRenda(clientes, cliente)
    const mediaScoreCredito = calcularScoreCredito(clientes, cliente)
    const mediaDividaAtual = calcularDividaAtual(clientes, cliente)
    const mediaMesesEmpregado = calcularMesesEmpregado(clientes, cliente)
    const mediaHistoricoInadimplencia = cliente.historico_inadimplencia
    const aprovado = cliente.aprovado

        return {
            mediaIdade,
            mediaRenda,
            mediaScoreCredito,
            mediaDividaAtual,
            mediaMesesEmpregado,
            mediaHistoricoInadimplencia,
            aprovado   
        }
    })


    return clientesNormalizados
    
}

function normalizarCliente(clientes, cliente) {
    const mediaIdade = calcularMeiaIdade(clientes, cliente)
    const mediaRenda = calcularMediaRenda(clientes, cliente)
    const mediaScoreCredito = calcularScoreCredito(clientes, cliente)
    const mediaDividaAtual = calcularDividaAtual(clientes, cliente)
    const mediaMesesEmpregado = calcularMesesEmpregado(clientes, cliente)
    const mediaHistoricoInadimplencia = cliente.historico_inadimplencia
    const aprovado = cliente.aprovado

    return [
        mediaIdade,
        mediaRenda,
        mediaScoreCredito,
        mediaDividaAtual,
        mediaMesesEmpregado,
        mediaHistoricoInadimplencia]
}

async function treinar() {
    const clientes = JSON.parse(await readFile('./data/clientes.json'))
    const dadosNormalizados = normalizarDados(clientes)
    const features = criarFeature(dadosNormalizados)
    const labels = dadosNormalizados.map(n => n.aprovado)

    const model = tf.sequential()
    model.add(tf.layers.dense({ inputShape: [6], units: 16, activation: "elu" }))
    model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }))

    model.compile({
        optimizer: "adam",
        loss: "binaryCrossentropy",
        metrics: ["accuracy"]
    })

    const inputX = tf.tensor2d(features)
    const outputY = tf.tensor2d(labels.map(l => [l]))

    await model.fit(inputX, outputY, {
        epochs: 300,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoca ${epoch + 1}: loss=${logs.loss.toFixed(4)} acc=${logs.acc?.toFixed(4)}`);
            }
        }
    })

    await model.save('file://./modelo')
    console.log('Modelo salvo em ./modelo')
}

async function rodar(cliente) {
    const clientes = JSON.parse(await readFile('./data/clientes.json'))
    const model = await tf.loadLayersModel('file://./modelo/model.json')

    const clienteNormalizado = normalizarCliente(clientes, cliente)
    const clienteTensor = tf.tensor2d([clienteNormalizado])
    const previsao = model.predict(clienteTensor)

    const result = (await previsao.data())[0]
    console.log(`Probabilidade de aprovação: ${(result * 100).toFixed(2)}%`)
    console.log(`Decisão: ${result >= 0.5 ? 'APROVADO' : 'NEGADO'}`)
}

const clienteNovo = {
    score_credito: 10,
    renda_mensal: 4800,
    divida_atual: 100,
    meses_empregado: 28,
    historico_inadimplencia: 0,
    idade: 35
}

const modo = process.argv[2]

if (modo === 'treinar') {
    await treinar()
} else {
    await rodar(clienteNovo)
}

