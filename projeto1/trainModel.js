import * as tf from '@tensorflow/tfjs';

export async function trainModel(input, outPut) {
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

// Dados de treinamento
export const pessoasTreinamentoReais = [
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

