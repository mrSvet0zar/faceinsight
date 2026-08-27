"use client";

import { useState } from "react";

import ConsentModal from "@/components/ConsentModal";
import FaceResults from "@/components/FaceResults";
import TelemetryStrip from "@/components/TelemetryStrip";
import UploadPanel from "@/components/UploadPanel";
import VideoSummary from "@/components/VideoSummary";
import WebcamPanel from "@/components/WebcamPanel";
import type { AnalyzeResponse, VideoAnalyzeResponse } from "@/lib/types";

type Mode = "webcam" | "photo" | "video";

const MODE_LABELS: Record<Mode, string> = {
  webcam: "Webcam",
  photo: "Photo",
  video: "Vidéo",
};

export default function DemoPage() {
  const [mode, setMode] = useState<Mode>("photo");
  const [consent, setConsent] = useState(false);
  const [askingConsent, setAskingConsent] = useState(false);

  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [videoResult, setVideoResult] = useState<VideoAnalyzeResponse | null>(
    null,
  );
  const [frameBase64, setFrameBase64] = useState<string | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [status, setStatus] = useState("en attente d'une image");

  const selectMode = (next: Mode) => {
    setResult(null);
    setVideoResult(null);
    setFrameBase64(null);
    setLatency(null);
    setStatus("en attente d'une image");
    if (next === "webcam" && !consent) {
      setAskingConsent(true);
      return; // le mode ne bascule qu'après consentement explicite
    }
    setMode(next);
  };

  const onAnalysis = (
    r: AnalyzeResponse,
    base64: string,
    latencyMs: number,
  ) => {
    setResult(r);
    setFrameBase64(base64);
    setLatency(latencyMs);
  };

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <section className="max-w-3xl">
        <p className="font-mono text-[11px] uppercase tracking-widest text-signal">
          Démo · deep learning multi-tâches
        </p>
        <h1 className="mt-2 font-display text-3xl font-semibold leading-tight sm:text-4xl">
          Ce qu&apos;un modèle voit sur un visage —
          <br className="hidden sm:block" /> et pourquoi il le voit.
        </h1>
        <p className="mt-3 text-sm leading-relaxed text-muted">
          Un seul réseau (ResNet-18 fine-tuné, 5 têtes) estime émotion, âge,
          genre perçu, pilosité et cheveux en une passe. Chaque prédiction est
          une estimation statistique — le bouton « Pourquoi ? » montre les zones
          du visage qui l&apos;ont influencée (Grad-CAM). Aucune image
          n&apos;est conservée, aucune identité recherchée.
        </p>
      </section>

      <div className="mt-6 flex gap-1 border-b border-line" role="tablist">
        {(Object.keys(MODE_LABELS) as Mode[]).map((m) => (
          <button
            key={m}
            role="tab"
            aria-selected={mode === m}
            onClick={() => selectMode(m)}
            className={`-mb-px rounded-t-md border px-4 py-2 text-sm ${
              mode === m
                ? "border-line border-b-panel bg-panel font-medium text-iris"
                : "border-transparent text-muted hover:text-ink"
            }`}
          >
            {MODE_LABELS[m]}
          </button>
        ))}
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[1.6fr_1fr]">
        <div className="space-y-3">
          {mode === "webcam" && consent ? (
            <WebcamPanel onAnalysis={onAnalysis} onStatus={setStatus} />
          ) : (
            <UploadPanel
              mode={mode === "video" ? "video" : "image"}
              onImageAnalysis={onAnalysis}
              onVideoAnalysis={(r) => setVideoResult(r)}
              onStatus={setStatus}
            />
          )}
          <TelemetryStrip
            mode={MODE_LABELS[mode]}
            faces={result?.faces.length ?? 0}
            latencyMs={latency}
            status={status}
          />
        </div>

        <aside aria-label="Résultats d'analyse">
          {mode === "video" ? (
            videoResult ? (
              <VideoSummary result={videoResult} />
            ) : (
              <div className="rounded-md border border-dashed border-line p-6 text-center text-sm text-muted">
                La synthèse de la vidéo s&apos;affichera ici.
              </div>
            )
          ) : (
            <FaceResults
              face={result?.faces[0] ?? null}
              imageBase64={frameBase64}
              modelTrained={result?.model_trained ?? true}
            />
          )}
        </aside>
      </div>

      {askingConsent && (
        <ConsentModal
          onAccept={() => {
            setConsent(true);
            setAskingConsent(false);
            setMode("webcam");
          }}
          onDecline={() => setAskingConsent(false)}
        />
      )}
    </div>
  );
}
