import { CLASS_THRESHOLD, TARGET_LABEL } from './config.js';

export function* processPredictions({ boxes, scores, classes }, labels, width, height) {
    for (let i = 0; i < scores.length; i++) {
        if (scores[i] < CLASS_THRESHOLD) continue;
        if (labels[classes[i]] !== TARGET_LABEL) continue;

        let [x1, y1, x2, y2] = boxes.slice(i * 4, (i + 1) * 4);
        x1 += width;
        x2 += width;
        y1 += height;
        y2 += height;

        yield {
            x: x1 + (x2 - x1) / 2,
            y: y1 + (y2 - y1) / 2,
            score: Math.round(scores[i] * 10000) / 100,
        };
    }
}
