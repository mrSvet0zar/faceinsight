"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { openRealtimeSocket } from "@/lib/api";
import { drawFaces } from "@/lib/overlay";
import { FaceSmoother } from "@/lib/smoothing";
import type { AnalyzeResponse } from "@/lib/types";

/** Largeur des frames envoyées au serveur (le flux affiché reste en pleine résolution). */
const CAPTURE_WIDTH = 480;
/** Intervalle minimal entre deux frames analysées (cf. CLAUDE.md : pas chaque frame). */
const MIN_INTERVAL_MS = 700;

interface Props {
  onAnalysis: (
    result: AnalyzeResponse,
    frameBase64: string,
    latencyMs: number,
  ) => void;
  onStatus: (status: string) => void;
}

/** Flux webcam + overlay temps réel via WebSocket. */
export default function WebcamPanel({ onAnalysis, onStatus }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const captureRef = useRef<HTMLCanvasElement | null>(null);
  const smootherRef = useRef(new FaceSmoother());
  const [error, setError] = useState<string | null>(null);

  // Les callbacks parent changent à chaque rendu : refs pour ne pas
  // redémarrer caméra + WebSocket à chaque analyse reçue.
  const onAnalysisRef = useRef(onAnalysis);
  const onStatusRef = useRef(onStatus);
  useEffect(() => {
    onAnalysisRef.current = onAnalysis;
    onStatusRef.current = onStatus;
  });

  const start = useCallback(() => {
    let stream: MediaStream | null = null;
    let ws: WebSocket | null = null;
    let stopped = false;
    let sentAt = 0;
    let lastFrame = "";

    const sendFrame = () => {
      const video = videoRef.current;
      if (stopped || !ws || ws.readyState !== WebSocket.OPEN || !video) return;
      if (video.videoWidth === 0) {
        setTimeout(sendFrame, 150);
        return;
      }
      captureRef.current ??= document.createElement("canvas");
      const canvas = captureRef.current;
      canvas.width = CAPTURE_WIDTH;
      canvas.height = Math.round(
        (video.videoHeight / video.videoWidth) * CAPTURE_WIDTH,
      );
      canvas
        .getContext("2d")!
        .drawImage(video, 0, 0, canvas.width, canvas.height);
      lastFrame = canvas.toDataURL("image/jpeg", 0.75).split(",")[1];
      sentAt = performance.now();
      ws.send(JSON.stringify({ frame: lastFrame }));
    };

    const run = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 } },
          audio: false,
        });
      } catch {
        setError(
          "Caméra inaccessible — vérifiez la permission du navigateur.",
        );
        onStatusRef.current("caméra refusée");
        return;
      }
      if (stopped || !videoRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      videoRef.current.srcObject = stream;
      await videoRef.current.play().catch(() => undefined);

      ws = openRealtimeSocket();
      ws.onopen = () => {
        onStatusRef.current("analyse en direct");
        sendFrame();
      };
      ws.onmessage = (event) => {
        if (stopped) return;
        const body = JSON.parse(event.data);
        if (!body.error) {
          const raw = body as AnalyzeResponse;
          // Lissage temporel du visage principal (anti-flicker émotion/âge)
          const smoothed = smootherRef.current.smooth(raw.faces[0] ?? null);
          const result: AnalyzeResponse = smoothed
            ? { ...raw, faces: [smoothed, ...raw.faces.slice(1)] }
            : raw;
          const latency = Math.round(performance.now() - sentAt);
          const overlay = overlayRef.current;
          const capture = captureRef.current;
          if (overlay && capture) {
            overlay.width = capture.width;
            overlay.height = capture.height;
            drawFaces(overlay.getContext("2d")!, result.faces, 1, 1);
          }
          onAnalysisRef.current(result, lastFrame, latency);
        }
        const elapsed = performance.now() - sentAt;
        setTimeout(sendFrame, Math.max(0, MIN_INTERVAL_MS - elapsed));
      };
      ws.onclose = () => {
        if (!stopped) onStatusRef.current("connexion au serveur perdue");
      };
      ws.onerror = () => {
        if (!stopped)
          setError(
            "Le serveur d'analyse ne répond pas — est-il démarré (port 8000) ?",
          );
      };
    };

    void run();

    return () => {
      stopped = true;
      ws?.close();
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  useEffect(() => start(), [start]);

  return (
    <div className="viewfinder overflow-hidden rounded-md bg-ink">
      <span className="vf-corners" aria-hidden />
      {error ? (
        <div className="flex aspect-video items-center justify-center p-6 text-center text-sm text-white/80">
          {error}
        </div>
      ) : (
        <div className="relative">
          <video
            ref={videoRef}
            muted
            playsInline
            className="block w-full"
            aria-label="Flux webcam en direct"
          />
          <canvas
            ref={overlayRef}
            className="pointer-events-none absolute inset-0 h-full w-full"
            aria-hidden
          />
        </div>
      )}
    </div>
  );
}
