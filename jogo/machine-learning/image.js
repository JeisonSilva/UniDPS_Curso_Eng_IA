import * as tf from '@tensorflow/tfjs';
import { INPUT_MODEL_DIMENSIONS } from './config.js';

export function preprocessImage(input) {
    return tf.tidy(() => {
        const image = tf.browser.fromPixels(input);
        return tf.image
            .resizeBilinear(image, [INPUT_MODEL_DIMENSIONS, INPUT_MODEL_DIMENSIONS])
            .div(255)
            .expandDims(0);
    });
}
