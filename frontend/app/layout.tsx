import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Sora } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const sora = Sora({
  variable: "--font-sora",
  subsets: ["latin"],
  weight: ["500", "600", "700"],
});

const plexSans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "FaceInsight — analyse faciale multi-attributs",
  description:
    "Démo d'analyse faciale par deep learning multi-tâches : émotion, âge, genre, attributs physiques. Privacy-first : aucune image conservée, aucune identification.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="fr"
      className={`${sora.variable} ${plexSans.variable} ${plexMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <header className="border-b border-line bg-panel">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
            <Link href="/" className="flex items-baseline gap-2">
              <span className="font-display text-lg font-semibold tracking-tight">
                FaceInsight
              </span>
              <span className="hidden font-mono text-[11px] text-muted sm:inline">
                analyse d&apos;attributs · jamais d&apos;identité
              </span>
            </Link>
            <nav className="flex items-center gap-5 text-sm">
              <Link href="/" className="hover:text-iris">
                Démo
              </Link>
              <Link href="/methodologie" className="hover:text-iris">
                Comment ça marche
              </Link>
              <a
                href="https://github.com/mrSvet0zar/faceinsight"
                target="_blank"
                rel="noreferrer"
                className="rounded border border-line px-2.5 py-1 font-mono text-xs hover:border-iris hover:text-iris"
              >
                GitHub
              </a>
            </nav>
          </div>
        </header>
        <main className="flex-1">{children}</main>
        <footer className="border-t border-line bg-panel">
          <div className="mx-auto flex max-w-6xl flex-col gap-1 px-4 py-4 font-mono text-[11px] text-muted sm:flex-row sm:items-center sm:justify-between">
            <span>
              Estimations statistiques d&apos;un modèle, à but démonstratif
              uniquement.
            </span>
            <span>traitement en mémoire · images conservées : 0</span>
          </div>
        </footer>
      </body>
    </html>
  );
}
