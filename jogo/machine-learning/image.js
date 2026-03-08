import * as tf from '@tensorflow/tfjs';

// O bitmap já chega pré-escalado para INPUT_MODEL_DIMENSIONS × INPUT_MODEL_DIMENSIONS
// (redimensionado nativamente pelo browser em main.js antes da transferência),
// portanto não é necessário resizeBilinear aqui.
export function preprocessImage(input) {
    return tf.tidy(() => {
        return tf.browser.fromPixels(input)
            .div(255)
            .expandDims(0);
    });
}
