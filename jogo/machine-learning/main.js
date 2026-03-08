import { buildLayout } from "./layout";
import { INPUT_MODEL_DIMENSIONS } from "./config.js";

async function captureFrame(game) {
    const canvas = game.app.renderer.extract.canvas(game.stage);
    return createImageBitmap(canvas, {
        resizeWidth: INPUT_MODEL_DIMENSIONS,
        resizeHeight: INPUT_MODEL_DIMENSIONS,
        resizeQuality: 'low',
    });
}

// Tempo médio entre captura do frame e disparo do tiro (inferência + transferência).
// Usado para antecipar a posição do pato e compensar a latência.
const SHOT_LATENCY_MS = 200;

async function sendNextFrame(worker, game) {
    try {
        const bitmap = await captureFrame(game);
        worker.postMessage({ type: 'predict', image: bitmap }, [bitmap]);
    } catch {
        // Falha ao capturar (ex: contexto WebGL perdido). Tenta novamente em 200ms
        // para evitar que a cadeia quebre silenciosamente.
        setTimeout(() => sendNextFrame(worker, game), 200);
    }
}

export default async function main(game) {
    const container = buildLayout(game.app);
    const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

    game.stage.aim.visible = false;

    // Rastreia a última predição para calcular velocidade e antecipar posição.
    let prevPrediction = null;
    let prevPredictionTime = 0;

    worker.onmessage = async ({ data }) => {
        if (data.type === 'idle') {
            // Pato fora de vista — reseta velocidade para evitar extrapolação errada
            // quando um pato diferente aparecer.
            prevPrediction = null;
        }

        if (data.type === 'prediction') {
            const now = performance.now();
            let { x, y } = data;

            // Antecipa posição com base na velocidade calculada entre predições consecutivas.
            // Compensa os ~200ms de latência entre a captura do frame e o disparo.
            if (prevPrediction && (now - prevPredictionTime) < 1000) {
                const dt = now - prevPredictionTime;
                const vx = (data.x - prevPrediction.x) / dt;
                const vy = (data.y - prevPrediction.y) / dt;
                x = Math.max(0, Math.min(800, data.x + vx * SHOT_LATENCY_MS));
                y = Math.max(0, Math.min(600, data.y + vy * SHOT_LATENCY_MS));
            }

            prevPrediction = { x: data.x, y: data.y };
            prevPredictionTime = now;

            container.updateHUD(data);
            game.stage.aim.visible = true;
            game.stage.aim.setPosition(x, y);
            game.handleClick({ global: game.stage.aim.getGlobalPosition() });
        }

        if (data.type === 'prediction' || data.type === 'idle') {
            sendNextFrame(worker, game);
        }
    };

    // Inicia o ciclo assim que o modelo sinalizar que está pronto.
    worker.addEventListener('message', async function onReady({ data }) {
        if (data.type !== 'model-loaded') return;
        worker.removeEventListener('message', onReady);
        sendNextFrame(worker, game);
    });

    return container;
}
