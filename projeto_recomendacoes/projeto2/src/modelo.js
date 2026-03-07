import tf from '@tensorflow/tfjs';

export function criarModelo(quantidadeFeatures) {
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
