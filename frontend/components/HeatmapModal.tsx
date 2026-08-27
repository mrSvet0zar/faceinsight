"use client";

import type { ExplainResponse } from "@/lib/types";

const ATTRIBUTE_LABELS: Record<string, string> = {
  emotion: "émotion",
  gender: "genre",
  facial_hair: "pilosité faciale",
  hair: "cheveux",
};

interface Props {
  result: ExplainResponse;
  onClose: () => void;
}

/** Overlay Grad-CAM : quelles zones du visage ont pesé dans la prédiction. */
export default function HeatmapModal({ result, onClose }: Props) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/50 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-sm rounded-lg border border-line bg-panel p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <p className="font-mono text-[11px] uppercase tracking-widest text-signal">
          Explicabilité · Grad-CAM
        </p>
        <h3 className="mt-1 font-display text-lg font-semibold">
          {ATTRIBUTE_LABELS[result.attribute] ?? result.attribute} :{" "}
          {result.predicted_label}
        </h3>
        {/* eslint-disable-next-line @next/next/no-img-element -- image base64 générée à la volée */}
        <img
          src={`data:image/png;base64,${result.heatmap_overlay_base64}`}
          alt={`Carte de chaleur Grad-CAM pour l'attribut ${result.attribute}`}
          className="mt-3 w-full rounded border border-line"
        />
        <p className="mt-2 text-xs text-muted">{result.explanation_note}.</p>
        <button
          onClick={onClose}
          className="mt-4 w-full rounded border border-line px-4 py-2 text-sm hover:bg-paper"
        >
          Fermer
        </button>
      </div>
    </div>
  );
}
