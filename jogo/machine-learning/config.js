export const MODEL_PATH = 'yolov5n_web_model/model.json';
export const LABELS_PATH = 'yolov5n_web_model/labels.json';
export const INPUT_MODEL_DIMENSIONS = 640;
export const CLASS_THRESHOLD = 0.4;
export const TARGET_LABEL = 'kite';
export const CAPTURE_INTERVAL_MS = 100;

// Dimensões internas do stage do jogo (Stage.js: MAX_X = 800, MAX_Y = 600).
// Usadas para converter coordenadas do modelo (640×640) para o espaço do stage.
export const STAGE_WIDTH = 800;
export const STAGE_HEIGHT = 600;
