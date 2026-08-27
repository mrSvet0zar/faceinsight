"use client";

import type { ExplainableAttribute } from "@/lib/types";

interface Props {
  title: string;
  value: string;
  /** Confiance numérique 0-1, ou null pour une estimation indicative. */
  confidence: number | null;
  note?: string;
  explainable?: ExplainableAttribute;
  onExplain?: (attribute: ExplainableAttribute) => void;
}

/** Carte d'attribut : valeur + jauge de confiance + bouton « Pourquoi ? ». */
export default function AttributeCard({
  title,
  value,
  confidence,
  note,
  explainable,
  onExplain,
}: Props) {
  return (
    <div className="rounded-md border border-line bg-panel p-3">
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-mono text-[11px] uppercase tracking-wider text-muted">
          {title}
        </span>
        {explainable && onExplain && (
          <button
            onClick={() => onExplain(explainable)}
            className="rounded border border-line px-2 py-0.5 font-mono text-[11px] text-iris hover:border-iris"
            title="Voir les zones du visage qui ont influencé cette prédiction"
          >
            Pourquoi ?
          </button>
        )}
      </div>
      <p className="mt-1 text-lg font-medium">{value}</p>
      {confidence !== null ? (
        <div className="mt-2 flex items-center gap-2">
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-paper">
            <div
              className="h-full rounded-full bg-signal"
              style={{ width: `${Math.round(confidence * 100)}%` }}
            />
          </div>
          <span className="font-mono text-[11px] text-muted">
            {Math.round(confidence * 100)}%
          </span>
        </div>
      ) : (
        <p className="mt-2 font-mono text-[11px] text-muted">
          précision indicative — heuristique, pas un modèle appris
        </p>
      )}
      {note && <p className="mt-1.5 text-xs text-muted">{note}</p>}
    </div>
  );
}
