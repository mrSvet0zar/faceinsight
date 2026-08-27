"use client";

/** Bandeau de télémétrie sous le viseur — l'élément signature de l'interface.
 * Le compteur « images conservées : 0 » est une constante délibérée :
 * la garantie de non-persistance fait partie du produit. */

interface Props {
  mode: string;
  faces: number;
  latencyMs: number | null;
  status: string;
}

export default function TelemetryStrip({ mode, faces, latencyMs, status }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-1 rounded-md border border-line bg-ink px-3 py-2 font-mono text-[11px] text-white/85">
      <span className="uppercase tracking-widest text-signal">● {mode}</span>
      <span>visages : {faces}</span>
      <span>latence : {latencyMs === null ? "—" : `${latencyMs} ms`}</span>
      <span className="text-white/60">{status}</span>
      <span className="ml-auto text-signal">images conservées : 0</span>
    </div>
  );
}
