"use client";

import type { VideoAnalyzeResponse } from "@/lib/types";

/** Synthèse d'une vidéo analysée : émotion dominante + répartition. */
export default function VideoSummary({ result }: { result: VideoAnalyzeResponse }) {
  const { aggregate } = result;
  const total = Object.values(aggregate.emotion_distribution).reduce(
    (a, b) => a + b,
    0,
  );

  return (
    <div className="space-y-2.5">
      {!result.model_trained && (
        <p className="rounded border border-iris/40 bg-iris-soft px-3 py-2 text-xs">
          Modèle non entraîné chargé (mode développement) — les valeurs
          ci-dessous ne sont pas significatives.
        </p>
      )}
      <div className="rounded-md border border-line bg-panel p-3">
        <span className="font-mono text-[11px] uppercase tracking-wider text-muted">
          Synthèse vidéo
        </span>
        <p className="mt-1 text-lg font-medium">
          {aggregate.dominant_emotion
            ? `émotion dominante : ${aggregate.dominant_emotion}`
            : "aucun visage détecté"}
        </p>
        <p className="mt-1 text-xs text-muted">
          {aggregate.frames_with_faces}/{aggregate.frames_sampled} images
          échantillonnées avec visage
          {aggregate.mean_age_estimate !== null &&
            ` · âge moyen estimé : ${aggregate.mean_age_estimate} ans`}
        </p>
      </div>
      {total > 0 && (
        <div className="rounded-md border border-line bg-panel p-3">
          <span className="font-mono text-[11px] uppercase tracking-wider text-muted">
            Répartition des émotions
          </span>
          <div className="mt-2 space-y-1.5">
            {Object.entries(aggregate.emotion_distribution)
              .sort(([, a], [, b]) => b - a)
              .map(([emotion, count]) => (
                <div key={emotion} className="flex items-center gap-2">
                  <span className="w-20 text-sm">{emotion}</span>
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-paper">
                    <div
                      className="h-full rounded-full bg-signal"
                      style={{ width: `${(count / total) * 100}%` }}
                    />
                  </div>
                  <span className="font-mono text-[11px] text-muted">
                    {count}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
