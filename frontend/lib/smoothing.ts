import type { FaceResult } from "./types";

/**
 * Lissage temporel des prédictions webcam : vote majoritaire sur l'émotion et
 * médiane sur l'âge, calculés sur les N dernières frames. Une prédiction
 * isolée (« neutre » pendant 300 ms au milieu d'une expression) n'atteint
 * plus l'affichage — sans rien changer au modèle.
 */
const WINDOW = 5;

export class FaceSmoother {
  private buffer: FaceResult[] = [];

  reset(): void {
    this.buffer = [];
  }

  smooth(face: FaceResult | null): FaceResult | null {
    if (!face) {
      // Visage perdu : on purge pour ne pas mélanger deux personnes/moments
      this.reset();
      return null;
    }
    this.buffer.push(face);
    if (this.buffer.length > WINDOW) this.buffer.shift();
    if (this.buffer.length < 2) return face;

    // Émotion : label majoritaire de la fenêtre, confiance moyenne du label
    const votes = new Map<string, number[]>();
    for (const f of this.buffer) {
      const list = votes.get(f.emotion.label) ?? [];
      list.push(f.emotion.confidence);
      votes.set(f.emotion.label, list);
    }
    const [label, confs] = [...votes.entries()].sort(
      (a, b) => b[1].length - a[1].length,
    )[0];

    // Âge : médiane de la fenêtre
    const ages = this.buffer
      .map((f) => f.age_estimate.value)
      .sort((a, b) => a - b);
    const age = ages[Math.floor(ages.length / 2)];

    return {
      ...face,
      emotion: {
        label,
        confidence: confs.reduce((a, b) => a + b, 0) / confs.length,
      },
      age_estimate: { ...face.age_estimate, value: age },
    };
  }
}
