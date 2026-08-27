"use client";

/** Écran de consentement obligatoire avant toute demande caméra (contrainte #4). */

interface Props {
  onAccept: () => void;
  onDecline: () => void;
}

export default function ConsentModal({ onAccept, onDecline }: Props) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="consent-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/50 p-4"
    >
      <div className="w-full max-w-lg rounded-lg border border-line bg-panel p-6 shadow-xl">
        <p className="font-mono text-[11px] uppercase tracking-widest text-signal">
          Consentement caméra
        </p>
        <h2 id="consent-title" className="mt-2 font-display text-xl font-semibold">
          Avant d&apos;activer votre webcam
        </h2>
        <ul className="mt-4 space-y-2 text-sm leading-relaxed">
          <li>
            <strong>Ce qui est analysé :</strong> des attributs visuels
            génériques (émotion apparente, âge estimé, genre perçu, pilosité,
            cheveux, couleur des yeux) sur les visages détectés.
          </li>
          <li>
            <strong>Ce qui n&apos;est jamais fait :</strong> aucune
            identification de personne, aucune comparaison à une base de
            visages, aucune image enregistrée — chaque image est analysée en
            mémoire puis immédiatement supprimée.
          </li>
          <li>
            <strong>Où vont les images :</strong> envoyées au serveur
            d&apos;analyse le temps du traitement (~1 s), jamais écrites sur
            disque, jamais partagées.
          </li>
        </ul>
        <p className="mt-4 text-xs text-muted">
          Les prédictions sont des estimations statistiques d&apos;un modèle,
          pas des faits. Vous pouvez couper la caméra à tout moment.
        </p>
        <div className="mt-6 flex justify-end gap-3">
          <button
            onClick={onDecline}
            className="rounded border border-line px-4 py-2 text-sm hover:bg-paper"
          >
            Refuser
          </button>
          <button
            onClick={onAccept}
            className="rounded bg-iris px-4 py-2 text-sm font-medium text-white hover:opacity-90"
          >
            J&apos;accepte, activer la caméra
          </button>
        </div>
      </div>
    </div>
  );
}
