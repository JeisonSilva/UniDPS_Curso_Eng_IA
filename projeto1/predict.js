import * as tf from '@tensorflow/tfjs';

export async function predict(model, pessoa) {
    const tfInput = tf.tensor2d(pessoa)

    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray
}
