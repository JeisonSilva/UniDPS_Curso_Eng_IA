import * as tf from '@tensorflow/tfjs';
import { MODEL_PATH, LABELS_PATH } from './config.js';

export async function loadModelAndLabels() {
    await tf.ready();

    const [labels, model] = await Promise.all([
        fetch(LABELS_PATH).then(r => r.json()),
        tf.loadGraphModel(MODEL_PATH),
    ]);

    const dummyInput = tf.ones(model.inputs[0].shape);
    try {
        const warmupOutput = await model.executeAsync(dummyInput);
        tf.dispose(Array.isArray(warmupOutput) ? warmupOutput : [warmupOutput]);
    } finally {
        dummyInput.dispose();
    }

    return { model, labels };
}

export async function runInference(model, tensor) {
    const output = await model.executeAsync(tensor);
    try {
        const [boxes, scores, classes] = output;
        const [boxesData, scoresData, classesData] = await Promise.all([
            boxes.data(),
            scores.data(),
            classes.data(),
        ]);
        return { boxes: boxesData, scores: scoresData, classes: classesData };
    } finally {
        tf.dispose(tensor);
        output.forEach(el => el.dispose());
    }
}
