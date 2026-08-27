"use client";

import { useState } from "react";

import AttributeCard from "./AttributeCard";
import HeatmapModal from "./HeatmapModal";
import { explainAttribute } from "@/lib/api";
import type {
  ExplainableAttribute,
  ExplainResponse,
  FaceResult,
} from "@/lib/types";

interface Props {
  face: FaceResult | null;
  /** Image analysée (base64 sans préfixe data:) — requise pour « Pourquoi ? ». */
  imageBase64: string | null;
  modelTrained: boolean;
}

function facialHairText(fh: FaceResult["facial_hair"]): string {
  if (fh.barbe && fh.moustache) return "barbe et moustache";
  if (fh.barbe) return "barbe";
  if (fh.moustache) return "moustache";
  return "aucune détectée";
}

/** Colonne de résultats pour un visage : une carte par attribut. */
export default function FaceResults({ face, imageBase64, modelTrained }: Props) {
  const [heatmap, setHeatmap] = useState<ExplainResponse | null>(null);
  const [explainLoading, setExplainLoading] = useState<string | null>(null);
  const [explainError, setExplainError] = useState<string | null>(null);

  if (!face) {
    return (
      <div className="rounded-md border border-dashed border-line p-6 text-center text-sm text-muted">
        Aucun visage détecté pour l&apos;instant.
      </div>
    );
  }

  const onExplain = async (attribute: ExplainableAttribute) => {
    if (!imageBase64) return;
    setExplainLoading(attribute);
    setExplainError(null);
    try {
      setHeatmap(await explainAttribute(imageBase64, attribute));
    } catch (err) {
      setExplainError(err instanceof Error ? err.message : String(err));
    } finally {
      setExplainLoading(null);
    }
  };

  const canExplain = imageBase64 !== null;

  return (
    <div className="space-y-2.5">
      {!modelTrained && (
        <p className="rounded border border-iris/40 bg-iris-soft px-3 py-2 text-xs">
          Modèle non entraîné chargé (mode développement) — les valeurs
          ci-dessous ne sont pas significatives.
        </p>
      )}
      <AttributeCard
        title="Émotion perçue"
        value={face.emotion.label}
        confidence={face.emotion.confidence}
        explainable={canExplain ? "emotion" : undefined}
        onExplain={onExplain}
      />
      <AttributeCard
        title="Âge estimé"
        value={`${face.age_estimate.value} ans (${face.age_estimate.range})`}
        confidence={null}
        note="Régression du modèle, marge d'erreur typique de quelques années."
      />
      <AttributeCard
        title="Genre perçu par le modèle"
        value={face.gender.label}
        confidence={face.gender.confidence}
        note="Classification binaire du modèle, pas l'identité réelle de la personne."
        explainable={canExplain ? "gender" : undefined}
        onExplain={onExplain}
      />
      <AttributeCard
        title="Pilosité faciale"
        value={facialHairText(face.facial_hair)}
        confidence={face.facial_hair.confidence}
        explainable={canExplain ? "facial_hair" : undefined}
        onExplain={onExplain}
      />
      <AttributeCard
        title="Cheveux"
        value={`${face.hair.couleur} · ${face.hair.longueur_estimee}`}
        confidence={face.hair.confidence}
        note="Longueur approximée (pas de label direct dans les données d'entraînement)."
        explainable={canExplain ? "hair" : undefined}
        onExplain={onExplain}
      />
      <AttributeCard
        title="Couleur des yeux"
        value={face.eye_color?.label ?? "non déterminée"}
        confidence={null}
      />
      {explainLoading && (
        <p className="font-mono text-xs text-muted">
          calcul de la heatmap ({explainLoading})…
        </p>
      )}
      {explainError && (
        <p className="text-xs text-alert">
          La heatmap n&apos;a pas pu être calculée : {explainError}
        </p>
      )}
      {heatmap && (
        <HeatmapModal result={heatmap} onClose={() => setHeatmap(null)} />
      )}
    </div>
  );
}
