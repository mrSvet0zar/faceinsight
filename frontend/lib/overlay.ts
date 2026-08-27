import type { FaceResult } from "./types";

const BOX_COLOR = "#12b5ab";
const LABEL_BG = "rgba(22, 34, 46, 0.82)";

/**
 * Dessine les bounding boxes + étiquettes sur le canvas overlay.
 * Les coordonnées de l'API sont dans le repère de l'image analysée ;
 * scale = taille affichée / taille analysée.
 */
export function drawFaces(
  ctx: CanvasRenderingContext2D,
  faces: FaceResult[],
  scaleX: number,
  scaleY: number,
): void {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.lineWidth = 2;
  ctx.strokeStyle = BOX_COLOR;
  ctx.font = "12px var(--font-plex-mono), monospace";

  for (const face of faces) {
    const { x, y, width, height } = face.bounding_box;
    const bx = x * scaleX;
    const by = y * scaleY;
    const bw = width * scaleX;
    const bh = height * scaleY;

    ctx.strokeRect(bx, by, bw, bh);

    const label = `${face.emotion.label} · ${face.age_estimate.range} ans`;
    const metrics = ctx.measureText(label);
    const ly = Math.max(16, by - 6);
    ctx.fillStyle = LABEL_BG;
    ctx.fillRect(bx - 1, ly - 13, metrics.width + 10, 18);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, bx + 4, ly);
  }
}
