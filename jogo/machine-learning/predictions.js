import { CLASS_THRESHOLD, TARGET_LABEL, STAGE_WIDTH, STAGE_HEIGHT } from './config.js';

// Retorna apenas a detecção com maior score do frame, ou null se nenhuma passar no threshold.
export function bestPrediction({ boxes, scores, classes }, labels) {
    let best = null;

    for (let i = 0; i < scores.length; i++) {
        if (scores[i] < CLASS_THRESHOLD) continue;
        if (labels[classes[i]] !== TARGET_LABEL) continue;
        if (best !== null && scores[i] <= scores[best]) continue;

        best = i;
    }

    if (best === null) return null;

    // O modelo retorna coordenadas normalizadas em [0, 1].
    // Alguns exports YOLOv5/TF.js usam convenção [y1, x1, y2, x2] (TensorFlow).
    // Escalonar para o espaço interno do stage (800×600).
    let [y1, x1, y2, x2] = boxes.slice(best * 4, (best + 1) * 4);

    x1 *= STAGE_WIDTH;
    x2 *= STAGE_WIDTH;
    y1 *= STAGE_HEIGHT;
    y2 *= STAGE_HEIGHT;

    return {
        x: x1 + (x2 - x1) / 2,
        y: y1 + (y2 - y1) / 2,
        score: Math.round(scores[best] * 10000) / 100,
    };
}
