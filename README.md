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

## Roadmap

- [x] **Phase 1** — Datasets, loaders, preprocessing/alignement MediaPipe
- [ ] **Phase 2** — Entraînement multi-task (Colab Pro+), suivi W&B, analyse de biais
- [ ] **Phase 3** — Backend FastAPI (upload image/vidéo, WebSocket temps réel, Grad-CAM)
- [ ] **Phase 4** — Frontend Next.js (webcam + consentement, upload, explicabilité)

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
