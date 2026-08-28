# FaceInsight — Analyse Faciale Multi-Attributs

Application d'analyse faciale (webcam temps réel, photo ou vidéo) qui prédit
plusieurs attributs via un modèle de deep learning **multi-tâches fine-tuné
maison** : émotion, âge, genre perçu, pilosité faciale, cheveux — plus une
heuristique de vision classique pour la couleur des yeux.

> ⚠️ **Ce n'est pas un système d'identification de personnes.** Le modèle
> analyse uniquement des attributs visuels génériques sur un visage détecté.
> Aucune reconnaissance faciale, aucune classification d'origine ethnique,
> aucune image conservée après traitement. Voir [Note éthique](#note-éthique).

## Architecture

```
Image visage alignée (224×224)
        │
        ▼
  Backbone partagé (ResNet-18 fine-tuné, pré-entraîné ImageNet)
        │
   ┌────┼────┬─────────┬─────────────┐
   ▼    ▼    ▼         ▼             ▼
Émotion Âge  Genre   Pilosité     Cheveux
(7-way) (régr.) (binaire) (multi-label) (multi-label)
```

La couleur des yeux n'utilise **pas** de deep learning : faute de dataset
large et fiable, elle est estimée par une heuristique classique (landmarks
MediaPipe → région iris → clustering HSV). Ce choix est documenté comme une
décision d'architecture assumée : savoir quand ne pas utiliser du deep
learning fait partie du projet.

## Datasets

| Tâche | Dataset | Notes |
|---|---|---|
| Émotion | FER2013 | 7 classes |
| Âge + genre | UTKFace | label ethnicité **exclu** (jamais chargé) |
| Pilosité + cheveux | CelebA | 13 attributs multi-label |

## Structure du repo

```
backend/
├── app/
│   ├── config.py              # chemins, classes, constantes des tâches
│   ├── inference/             # détection MediaPipe, alignement, préprocessing
│   ├── training/              # téléchargement datasets, loaders, exploration
│   ├── models/                # architecture multi-tâches (Phase 2)
│   └── api/                   # endpoints FastAPI (Phase 3)
└── tests/                     # tests unitaires (sans datasets requis)
frontend/                      # Next.js (Phase 4)
```

## Démarrage (Phase 1)

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate          # Windows — source .venv/bin/activate sur Linux/Mac
pip install -r requirements.txt

# Credentials Kaggle requis (~/.kaggle/kaggle.json) : https://www.kaggle.com/docs/api
python -m app.training.download_datasets      # FER2013 + UTKFace + CelebA (~3 Go)
python -m app.training.explore_datasets       # statistiques par dataset

pytest                                        # tests unitaires (sans datasets)
```

## Résultats

Deux runs comparés dans W&B, test sets jamais vus à l'entraînement. Rapports
complets : [`reports/eval_baseline.json`](reports/eval_baseline.json) et
[`reports/eval_v2.json`](reports/eval_v2.json) (modèle retenu en production).

| Tâche | Métrique | Baseline (20 ep.) | **v2 (30 ep.)** | Cible |
|---|---|---|---|---|
| Émotion | accuracy | 55,8 % | **64,6 %** | > 65 % 🟡 |
| Âge | MAE | 6,2 ans | **6,0 ans** | < 6 ans 🟡 |
| Genre | accuracy | 92,4 % | **93,3 %** | > 90 % ✅ |
| Pilosité faciale | macro F1 | 0,66 | **0,74** | — |
| Cheveux | macro F1 | 0,71 | **0,73** | — |

**La leçon du run 1 (documentée volontairement)** : le meilleur checkpoint
était sélectionné sur la loss de validation pondérée, qui remontait par excès
de confiance dès l'epoch 6 alors que toutes les métriques continuaient de
progresser — figeant un modèle sous-entraîné (émotion 55,8 %). Le run 2
sélectionne le best sur un score composite de métriques et ajoute du label
smoothing sur l'émotion : +8,8 points d'accuracy émotion, toutes les tâches
en progrès, best à l'epoch 28/30. C'est exactement le genre d'écart
loss/métriques qu'un suivi W&B rigoureux permet d'attraper.

Point faible restant : la classe `sad` (38,6 % de rappel, confondue avec
`fear`/`angry`/`neutral`) — limite connue de FER2013, où ces émotions sont
visuellement proches sur des images 48×48.

### Analyse de biais par sous-groupe (genre/âge, UTKFace test, modèle v2)

| Sous-groupe | n | MAE âge | Accuracy genre |
|---|---|---|---|
| 0-18 ans | 456 | 2,3 | **79,6 %** ⚠️ |
| 19-35 ans | 1 032 | 3,6 | 96,3 % |
| 36-60 ans | 623 | 8,4 | 97,6 % |
| 61+ ans | 260 | **16,7** ⚠️ | 95,0 % |
| Hommes | 1 273 | 6,1 | 91,3 % |
| Femmes | 1 098 | 5,9 | 95,6 % |

Deux dégradations nettes, publiées plutôt que masquées : le genre est peu
fiable sur les visages d'enfants (traits peu genrés + sous-représentation),
et l'âge est fortement sous-estimé chez les 61+ (260 exemples de test
seulement — le modèle régresse vers la moyenne du dataset).

## Roadmap

- [x] **Phase 1** — Datasets, loaders, preprocessing/alignement MediaPipe
- [x] **Phase 2** — Entraînement multi-task (Colab), suivi W&B, analyse de biais — itération sur les pondérations en cours
- [x] **Phase 3** — Backend FastAPI (upload image/vidéo, WebSocket temps réel, Grad-CAM, test de non-persistance)
- [x] **Phase 4** — Frontend Next.js (webcam + consentement, upload, explicabilité, page méthodologie)
- [ ] **Phase 5** — Déploiement public (HF Hub + Railway + Vercel), rapport d'évaluation final

## Note éthique

1. **Pas d'identification de personnes** — aucun matching contre une base de
   visages, aucun embedding lié à une identité.
2. **Pas de classification ethnique** — le label ethnicité d'UTKFace est
   exclu au niveau du parsing des fichiers (testé unitairement).
3. **Pas de stockage d'images** — traitement en mémoire uniquement, vérifié
   par test automatisé (Phase 3).
4. **Consentement explicite** avant toute activation caméra.
5. **Prédictions présentées comme des estimations statistiques** — jamais
   comme des faits (ex. « genre perçu par le modèle », avec confiance).
6. **Aucun log d'usage nominatif.**

## Licence & crédits

Datasets : FER2013 (Kaggle), UTKFace, CelebA (usage recherche/non commercial).
Détection de visage : MediaPipe (Apache 2.0), non réentraînée.
