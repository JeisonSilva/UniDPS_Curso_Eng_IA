import { loadModelAndLabels, runInference } from './model.js';
import { preprocessImage } from './image.js';
import { bestPrediction } from './predictions.js';

const { model, labels } = await loadModelAndLabels();
postMessage({ type: 'model-loaded' });

let _inferring = false;

self.onmessage = async ({ data }) => {
    if (data.type !== 'predict' || _inferring) return;

    _inferring = true;
    const input = preprocessImage(data.image);

    try {
        const results = await runInference(model, input);
        const prediction = bestPrediction(results, labels);
        postMessage(prediction
            ? { type: 'prediction', ...prediction }
            : { type: 'idle' }
        );
    } finally {
        _inferring = false;
    }
};
