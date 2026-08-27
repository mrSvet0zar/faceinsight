"use client";

import { useEffect, useRef, useState } from "react";

import { analyzeImage, analyzeVideo } from "@/lib/api";
import { drawFaces } from "@/lib/overlay";
import type {
  AnalyzeResponse,
  VideoAnalyzeResponse,
} from "@/lib/types";

type Mode = "image" | "video";

interface Props {
  mode: Mode;
  onImageAnalysis: (
    result: AnalyzeResponse,
    imageBase64: string,
    latencyMs: number,
  ) => void;
  onVideoAnalysis: (result: VideoAnalyzeResponse) => void;
  onStatus: (status: string) => void;
}

const ACCEPT: Record<Mode, string> = {
  image: "image/jpeg,image/png,image/webp",
  video: "video/mp4,video/webm,video/quicktime",
};

/** Zone de dépôt photo/vidéo + aperçu annoté (photo). */
export default function UploadPanel({
  mode,
  onImageAnalysis,
  onVideoAnalysis,
  onStatus,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [imageResult, setImageResult] = useState<AnalyzeResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    // Aperçu local via object URL, révoqué dès qu'il est remplacé
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const drawOverlay = (result: AnalyzeResponse) => {
    const img = imgRef.current;
    const overlay = overlayRef.current;
    if (!img || !overlay) return;
    overlay.width = img.naturalWidth;
    overlay.height = img.naturalHeight;
    drawFaces(overlay.getContext("2d")!, result.faces, 1, 1);
  };

  const handleFile = async (file: File) => {
    setError(null);
    setBusy(true);
    setImageResult(null);
    onStatus("analyse en cours…");
    try {
      if (mode === "image") {
        setPreviewUrl(URL.createObjectURL(file));
        const started = performance.now();
        const result = await analyzeImage(file);
        const base64 = await fileToBase64(file);
        setImageResult(result);
        onImageAnalysis(result, base64, Math.round(performance.now() - started));
        onStatus(
          result.faces.length > 0
            ? `${result.faces.length} visage(s) détecté(s)`
            : "aucun visage détecté",
        );
      } else {
        setPreviewUrl(null);
        onStatus("analyse vidéo (≈1 image/s)…");
        const result = await analyzeVideo(file);
        onVideoAnalysis(result);
        onStatus(`${result.aggregate.frames_sampled} images échantillonnées`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      onStatus("erreur d'analyse");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="viewfinder rounded-md bg-panel">
      <span className="vf-corners" aria-hidden />
      {previewUrl && mode === "image" ? (
        <div className="relative">
          {/* eslint-disable-next-line @next/next/no-img-element -- aperçu local (object URL) */}
          <img
            ref={imgRef}
            src={previewUrl}
            alt="Image analysée"
            className="block max-h-[520px] w-full rounded-md object-contain"
            onLoad={() => imageResult && drawOverlay(imageResult)}
          />
          <canvas
            ref={overlayRef}
            className="pointer-events-none absolute inset-0 h-full w-full object-contain"
            aria-hidden
          />
          <button
            onClick={() => {
              setPreviewUrl(null);
              setImageResult(null);
            }}
            className="absolute right-2 top-2 rounded bg-ink/70 px-2 py-1 font-mono text-[11px] text-white hover:bg-ink"
          >
            changer d&apos;image
          </button>
        </div>
      ) : (
        <button
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files[0];
            if (file) void handleFile(file);
          }}
          disabled={busy}
          className={`flex aspect-video w-full flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed p-6 text-center transition-colors ${
            dragging ? "border-signal bg-signal-soft" : "border-line hover:border-signal"
          }`}
        >
          <span className="font-mono text-[11px] uppercase tracking-widest text-muted">
            {mode === "image" ? "photo · jpeg / png / webp" : "vidéo · mp4 / webm"}
          </span>
          <span className="text-sm">
            {busy
              ? "Analyse en cours…"
              : "Déposez un fichier ici ou cliquez pour parcourir"}
          </span>
          <span className="text-xs text-muted">
            Analysé en mémoire puis immédiatement supprimé — jamais stocké.
          </span>
        </button>
      )}
      {error && <p className="p-3 text-xs text-alert">{error}</p>}
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT[mode]}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void handleFile(file);
          e.target.value = "";
        }}
      />
    </div>
  );
}

async function fileToBase64(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}
