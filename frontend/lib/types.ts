/** Types miroirs du format de réponse de l'API FastAPI (cf. CLAUDE.md). */

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface FaceResult {
  bounding_box: BoundingBox;
  detection_confidence: number;
  emotion: { label: string; confidence: number };
  age_estimate: { value: number; range: string };
  gender: { label: string; confidence: number };
  facial_hair: { barbe: boolean; moustache: boolean; confidence: number };
  hair: { couleur: string; longueur_estimee: string; confidence: number };
  eye_color: { label: string; method: string; confidence: string } | null;
}

export interface AnalyzeResponse {
  faces: FaceResult[];
  disclaimer: string;
  explainability_available: boolean;
  model_trained: boolean;
}

export interface VideoAggregate {
  frames_sampled: number;
  frames_with_faces: number;
  dominant_emotion: string | null;
  emotion_distribution: Record<string, number>;
  mean_age_estimate: number | null;
}

export interface VideoAnalyzeResponse {
  frames: { timestamp_s: number; faces: FaceResult[] }[];
  aggregate: VideoAggregate;
  disclaimer: string;
  model_trained: boolean;
}

export interface ExplainResponse {
  attribute: string;
  predicted_label: string;
  heatmap_overlay_base64: string;
  explanation_note: string;
  model_trained: boolean;
}

/** Attributs pour lesquels l'endpoint /explain accepte une heatmap. */
export type ExplainableAttribute = "emotion" | "gender" | "facial_hair" | "hair";
