import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Comment ça marche — FaceInsight",
  description:
    "Modèles, datasets, limites et engagements éthiques de FaceInsight.",
};

const SECTIONS = [
  {
    id: "pipeline",
    title: "Le pipeline d'analyse",
    body: (
      <>
        <p>
          Chaque image passe par quatre étapes : <strong>détection</strong> du
          visage (MediaPipe BlazeFace, modèle pré-entraîné non modifié),{" "}
          <strong>alignement</strong> (rotation pour mettre les yeux à
          l&apos;horizontale, recadrage 224×224),{" "}
          <strong>inférence multi-tâches</strong> (une seule passe dans un
          ResNet-18 fine-tuné avec cinq têtes de sortie), puis{" "}
          <strong>heuristique couleur des yeux</strong> (landmarks de
          l&apos;iris + analyse HSV, sans apprentissage).
        </p>
        <p>
          Le multi-tâches n&apos;est pas un détail : les représentations
          visuelles utiles à une tâche (détecter des cheveux gris) recoupent
          celles d&apos;une autre (estimer l&apos;âge). Un backbone partagé
          apprend ces features une seule fois, et l&apos;inférence est cinq fois
          moins coûteuse que cinq modèles séparés.
        </p>
      </>
    ),
  },
  {
    id: "donnees",
    title: "Les données d'entraînement",
    body: (
      <>
        <p>
          Trois datasets publics de recherche : <strong>FER2013</strong>{" "}
          (35 000 visages, 7 émotions), <strong>UTKFace</strong> (23 000
          visages annotés en âge et genre) et <strong>CelebA</strong> (200 000
          visages, attributs pilosité et cheveux). UTKFace contient aussi un
          label d&apos;origine ethnique : il est <strong>exclu du projet</strong>{" "}
          — jamais chargé, jamais appris, jamais prédit. Cette exclusion est
          faite au niveau du code de lecture des fichiers et vérifiée par un
          test automatisé.
        </p>
      </>
    ),
  },
  {
    id: "explicabilite",
    title: "L'explicabilité (Grad-CAM)",
    body: (
      <p>
        Le bouton « Pourquoi ? » calcule une carte Grad-CAM : elle met en
        évidence les zones de l&apos;image qui ont le plus contribué à la
        prédiction, en remontant les gradients jusqu&apos;aux dernières couches
        convolutives du réseau. Une prédiction « moustache » qui s&apos;appuie
        sur la zone de la bouche est plus digne de confiance qu&apos;une
        prédiction qui s&apos;appuierait sur l&apos;arrière-plan — c&apos;est
        exactement ce que cet outil permet de vérifier.
      </p>
    ),
  },
  {
    id: "limites",
    title: "Les limites connues",
    body: (
      <ul className="list-disc space-y-1.5 pl-5">
        <li>
          <strong>Couleur des yeux</strong> : heuristique sans apprentissage,
          précision indicative — peu fiable sur images basse résolution ou mal
          éclairées. Choix assumé : pas de dataset fiable, donc pas de modèle.
        </li>
        <li>
          <strong>Longueur des cheveux</strong> : CelebA n&apos;a pas de label
          court/long ; la valeur affichée est approximée à partir
          d&apos;attributs visibles.
        </li>
        <li>
          <strong>Genre</strong> : classification binaire imposée par les
          datasets disponibles. C&apos;est une perception du modèle, pas
          l&apos;identité de la personne — l&apos;interface le rappelle sur
          chaque résultat.
        </li>
        <li>
          <strong>Biais des données</strong> : les datasets publics
          sur-représentent certaines populations. Les métriques par sous-groupe
          (tranches d&apos;âge, genre) sont mesurées et publiées dans le README
          du projet.
        </li>
        <li>
          <strong>Conditions difficiles</strong> : contre-jour, occlusions et
          angles extrêmes dégradent la détection et les prédictions.
        </li>
      </ul>
    ),
  },
  {
    id: "vie-privee",
    title: "Vie privée : ce que ce projet refuse de faire",
    body: (
      <>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>
            <strong>Pas d&apos;identification.</strong> Aucune base de visages,
            aucun embedding lié à une identité, aucune réponse à « qui est
            cette personne ».
          </li>
          <li>
            <strong>Pas de stockage.</strong> Les images sont traitées en
            mémoire et supprimées sitôt la réponse envoyée. Un test automatisé
            inspecte le disque après chaque appel d&apos;API pour le garantir.
          </li>
          <li>
            <strong>Pas de classification ethnique.</strong> Ni entraînée, ni
            prédite, ni affichée.
          </li>
          <li>
            <strong>Pas de traçage.</strong> Aucun log nominatif, aucun
            identifiant de session dans les statistiques.
          </li>
        </ul>
        <p className="mt-3">
          Le code complet (entraînement, API, cette interface) est public :{" "}
          <a
            href="https://github.com/mrSvet0zar/faceinsight"
            className="text-iris underline underline-offset-2"
            target="_blank"
            rel="noreferrer"
          >
            github.com/mrSvet0zar/faceinsight
          </a>
          .
        </p>
      </>
    ),
  },
];

export default function MethodologiePage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-10">
      <p className="font-mono text-[11px] uppercase tracking-widest text-signal">
        Transparence · méthodologie
      </p>
      <h1 className="mt-2 font-display text-3xl font-semibold">
        Comment ça marche
      </h1>
      <p className="mt-3 text-sm leading-relaxed text-muted">
        Un système qui analyse des visages doit expliquer ce qu&apos;il fait,
        avec quelles données, et où il s&apos;arrête. Cette page documente les
        choix techniques et éthiques du projet — les deux sont indissociables.
      </p>

      <div className="mt-8 space-y-8">
        {SECTIONS.map((section) => (
          <section key={section.id} aria-labelledby={section.id}>
            <h2
              id={section.id}
              className="border-b border-line pb-2 font-display text-lg font-semibold"
            >
              {section.title}
            </h2>
            <div className="mt-3 space-y-3 text-sm leading-relaxed">
              {section.body}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
