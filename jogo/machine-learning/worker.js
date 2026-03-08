import { loadModelAndLabels, runInference } from './model.js';
import { preprocessImage } from './image.js';
import { processPredictions } from './predictions.js';

const { model, labels } = await loadModelAndLabels();
postMessage({ type: 'model-loaded' });

let _inferring = false;

self.onmessage = async ({ data }) => {
    if (data.type !== 'predict' || _inferring) return;

    _inferring = true;
    const input = preprocessImage(data.image);
    const { width, height } = data.image;

    try {
        const results = await runInference(model, input);
        for (const prediction of processPredictions(results, labels, width, height)) {
            postMessage({ type: 'prediction', ...prediction });
        }
    } finally {
        _inferring = false;
    }
};
