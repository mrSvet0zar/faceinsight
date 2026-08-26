# CLAUDE.md — FaceInsight (Analyse Faciale Multi-Attributs)

## 🎯 Vision du Projet

**FaceInsight** est une application d'analyse faciale qui détecte un visage (webcam
temps réel, photo ou vidéo uploadée) et prédit plusieurs attributs via des modèles
de deep learning **fine-tunés maison** : émotion, âge, genre, et description physique
(pilosité faciale, longueur/couleur des cheveux, couleur des yeux).

**Positionnement portfolio :** contrairement aux 5 premiers projets qui exploitent
l'API Claude, celui-ci démontre des compétences **computer vision & deep learning
"from scratch"** — entraînement de modèles, apprentissage multi-tâches, gestion de
datasets, métriques d'évaluation par tâche. C'est le projet qui répond le mieux à un
recruteur qui demanderait "as-tu déjà entraîné un modèle toi-même, pas juste appelé
une API ?".

---

## ⚠️ Contraintes Non-Négociables (Éthique & Vie Privée)

Ces règles structurent toute l'architecture, au même titre que le respect du copyright
structure Prisme-IA. À ne jamais contourner, même pour "aller plus vite" :

1. **Ce n'est PAS un système d'identification de personnes.** Aucune reconnaissance
   faciale au sens "qui est cette personne" (pas de matching contre une base de
   visages connus, pas de stockage d'embeddings liés à une identité). Le système
   analyse uniquement des **attributs visuels génériques** sur un visage détecté,
   sans jamais chercher à savoir de qui il s'agit.

2. **Aucune classification d'origine ethnique/raciale.** Même si certains datasets
   publics (UTKFace notamment) incluent un label d'ethnicité, ce label est **exclu
   du projet** — non entraîné, non prédit, non affiché. Seuls âge, genre, émotion,
   et attributs physiques neutres (pilosité, couleur/longueur cheveux, couleur yeux)
   sont dans le périmètre.

3. **Pas de stockage des images/frames.** Traitement en mémoire uniquement, résultat
   retourné puis frame/image immédiatement jetée. Aucune image uploadée ou capturée
   via webcam n'est persistée sur disque ou en base au-delà du temps de traitement.
   C'est aussi un argument produit fort ("privacy-first") pour la démo publique.

4. **Consentement explicite avant activation caméra.** Écran de consentement clair
   avant toute demande de permission navigateur (`getUserMedia`), qui explique ce
   qui est analysé et confirme qu'aucune image n'est conservée.

5. **Disclaimer visible sur les résultats.** Chaque prédiction (genre, âge, émotion)
   doit être présentée comme une **estimation statistique du modèle**, pas un fait :
   ex. "Genre perçu par le modèle : Femme (confiance 82%)" plutôt qu'une affirmation
   catégorique. Le genre est traité comme une classification binaire du modèle par
   simplicité technique (contrainte des datasets disponibles), avec ce disclaimer
   explicite plutôt qu'une prétention à catégoriser l'identité réelle de la personne.

6. **Pas de logs d'usage nominatifs.** Si des statistiques d'usage agrégées sont
   conservées (nombre d'analyses, répartition des émotions détectées), elles doivent
   être totalement anonymes et non traçables à une session/personne individuelle.

---

## 🏗️ Stack Technique

| Composant | Choix | Justification |
|---|---|---|
| Entraînement | PyTorch + torchvision | Standard, bon support des architectures pré-entraînées à fine-tuner |
| Détection de visage | MediaPipe Face Detection (pré-entraîné, non réentraîné) | Tâche solvée, pas de valeur ajoutée à la réentraîner — la profondeur technique du projet est sur la classification d'attributs, pas la détection |
| Backbone classification | ResNet-18 ou EfficientNet-B0 (pré-entraîné ImageNet, fine-tuné) | Bon compromis précision/vitesse pour inference temps réel |
| Tracking d'expériences | Weights & Biases | Comparaison rigoureuse des runs (cf. bonnes pratiques production-ready) |
| Backend | FastAPI (Python) | Cohérent avec l'écosystème PyTorch, WebSocket natif pour le flux temps réel |
| Frontend | Next.js 14+, TypeScript, Tailwind | Cohérent avec le reste du portfolio |
| Communication temps réel | WebSocket (frames périodiques, pas chaque frame) | Évite de saturer le réseau/l'inférence |
| Déploiement backend | Railway ou Fly.io (GPU si besoin pour l'inférence, sinon CPU suffisant en quantized) | Modèles légers, inférence CPU viable pour une démo |
| Déploiement frontend | Vercel | Standard du portfolio |

---

## ☁️ Infrastructure d'Entraînement Distant

Le fine-tuning (multi-task, plusieurs datasets dont CelebA qui pèse plusieurs Go) ne
tourne pas en local. Entraînement et inférence sont **découplés** : l'entraînement
produit des checkpoints versionnés, l'inférence en backend ne fait que les charger.

### Option recommandée : Google Colab Pro+
Cohérent avec l'approche déjà utilisée sur le Projet 2 (fine-tuning LLM). Points d'attention
spécifiques au multi-task learning (sessions plus longues qu'un simple LoRA) :
- **Sauvegarde de checkpoint toutes les N epochs vers Google Drive** (pas seulement en fin
  de run) — une session Colab peut se couper, un run de plusieurs heures doit survivre à une
  déconnexion
- Reprise d'entraînement (`resume_from_checkpoint`) implémentée dès le départ dans `train.py`,
  pas ajoutée après coup
- Logs W&B en parallèle (pas seulement en local) pour ne rien perdre même si la session meurt

### Alternative si les runs deviennent longs/fréquents : RunPod ou Vast.ai
Location de GPU à l'heure avec stockage persistant (pas de limite de session comme Colab).
Pertinent si les itérations sur les pondérations multi-tâches nécessitent beaucoup de runs
successifs — évite de re-télécharger les datasets à chaque session.

### Flux de Déploiement des Poids
```
Entraînement (Colab/RunPod) 
   → Checkpoint final validé (meilleur run selon métriques W&B)
   → Upload vers Hugging Face Hub (repo privé) ou cloud storage (S3/GCS)
   → Backend FastAPI télécharge le checkpoint au démarrage/déploiement
   → Inférence en production (CPU suffisant pour ResNet-18 quantisé)
```

Ne jamais committer les poids (`.pth`) dans le repo Git — versionner uniquement le code
d'entraînement et la référence (URL/tag) du checkpoint utilisé en production.

---

| Attribut | Dataset | Notes |
|---|---|---|
| Émotion | FER2013 ou AffectNet | 7-8 classes (colère, dégoût, peur, joie, tristesse, surprise, neutre) |
| Âge | UTKFace | Label ethnicité du dataset **ignoré/non chargé** (cf. contrainte éthique #2) |
| Genre | UTKFace ou CelebA (attribut `Male`) | Classification binaire, avec disclaimer (cf. contrainte #5) |
| Pilosité faciale | CelebA (`No_Beard`, `Goatee`, `Mustache`, `5_o_Clock_Shadow`, `Sideburns`) | Multi-label |
| Couleur/longueur cheveux | CelebA (`Black_Hair`, `Blond_Hair`, `Brown_Hair`, `Gray_Hair`, `Bald`, `Bangs`, `Straight_Hair`, `Wavy_Hair`) | Pas de label "court/long" direct — approximer via `Bald`/`Bangs`/longueur visible, documenter comme limite si imprécis |
| Couleur des yeux | Pas de dataset large labellisé disponible | Voir approche hybride ci-dessous |

### Couleur des Yeux — Approche Hybride (Décision d'Architecture à Documenter)
Contrairement aux autres attributs, il n'existe pas de dataset large et fiable pour la
couleur des yeux. Plutôt que d'entraîner un modèle peu fiable sur des données limitées,
utiliser une **heuristique de vision classique** : après détection des landmarks
oculaires (MediaPipe Face Mesh), extraire la région de l'iris et classifier la couleur
dominante par clustering HSV simple (marron / bleu / vert / gris / noisette). Documenter
explicitement ce choix dans le README comme "précision indicative, pas un modèle appris"
— c'est un excellent point de discussion en entretien (savoir quand NE PAS utiliser du
deep learning faute de données suffisantes est un vrai signal de maturité).

---

## 🧠 Architecture — Multi-Task Learning

Plutôt que d'entraîner un modèle séparé par attribut, utiliser un **backbone partagé**
avec plusieurs têtes de sortie — plus impressionnant techniquement et plus efficace
à l'inférence (une seule forward pass pour tous les attributs sauf couleur des yeux) :

```
Image visage alignée (224x224)
        │
        ▼
  Backbone partagé (ResNet-18 fine-tuné)
        │
   ┌────┼────┬────────┬─────────────┐
   ▼    ▼    ▼         ▼             ▼
 Tête  Tête  Tête    Tête          Tête
Émotion Âge  Genre  Pilosité    Cheveux
(8-way (regr.(binaire (multi-     (multi-
 CE)   MSE)   CE)     label BCE) label BCE)
```

**Fonction de perte combinée :**
```
loss_total = w1*loss_emotion + w2*loss_age + w3*loss_genre 
           + w4*loss_pilosite + w5*loss_cheveux
```
Pondérations (`w1..w5`) à ajuster empiriquement — documenter les valeurs testées et le
choix final dans le README (comparaison de runs W&B).

**Justification pédagogique à inclure dans le README :** le multi-task learning permet
au modèle de partager des représentations visuelles utiles entre tâches connexes (ex:
les features utiles pour détecter l'âge recoupent en partie celles utiles pour les
cheveux gris), ce qui peut améliorer chaque tâche individuellement par rapport à des
modèles entraînés séparément — à vérifier empiriquement et documenter (comparaison
multi-task vs modèles séparés = excellent contenu pour un post technique).

---

## ✨ Feature Différenciante : Explicabilité des Prédictions (Grad-CAM)

**Le problème que ça résout :** des dizaines de projets "face analysis" existent déjà
en portfolio/GitHub, quasiment tous en boîte noire (upload photo → résultat, sans
explication). Ajouter de l'explicabilité différencie FaceInsight et démontre une
compétence ML plus rare que la classification pure.

**Principe :** pour chaque tête de prédiction (émotion, genre, pilosité, cheveux),
générer une carte de chaleur (Grad-CAM ou Score-CAM) sur les dernières couches
convolutives du backbone partagé, montrant quelles régions du visage ont le plus
contribué à la décision. Superposée en transparence sur l'image/le flux webcam.

**Implémentation :**
- Utiliser une librairie existante (`pytorch-grad-cam`) plutôt que ré-implémenter
  Grad-CAM from scratch — la valeur du projet est dans l'intégration au produit et le
  multi-task learning, pas dans la réécriture d'un algorithme déjà standard
- Une heatmap par tête activée (l'utilisateur peut choisir "montrer pourquoi le modèle
  pense X" pour chaque attribut affiché)
- En mode temps réel (webcam), calculer la heatmap seulement sur demande ponctuelle
  (bouton "Pourquoi ?") plutôt qu'en continu — Grad-CAM ajoute un coût de calcul non
  négligeable, pas nécessaire à chaque frame

**Impact sur l'architecture existante :**
- Nouveau module `inference/explainability.py` (wrapper autour de `pytorch-grad-cam`
  branché sur le backbone partagé)
- Nouvel endpoint `POST /api/analyze/explain` — prend l'image + l'attribut ciblé, retourne
  l'image originale + la heatmap superposée (base64) en plus du JSON de prédiction
- Nouveau composant frontend `AttributeHeatmap` (overlay sur `FaceOverlay`, toggle par attribut)

**Bonus contenu :** ce composant se prête très bien à un post LinkedIn/blog dédié
("Pourquoi mon modèle pense que vous souriez — explicabilité en Computer Vision"), qui
recoupe directement la stratégie de contenu déjà évoquée pour ton profil.

---

## 🔄 Pipeline d'Inférence

1. **Détection** : MediaPipe détecte le/les visage(s) dans la frame/image
2. **Alignement** : recadrage + rotation basée sur les landmarks (yeux alignés horizontalement)
3. **Inférence multi-tâches** : une forward pass dans le modèle fine-tuné → émotion, âge, genre, pilosité, cheveux
4. **Heuristique couleur des yeux** : extraction région iris + clustering HSV (indépendant du modèle principal)
5. **Agrégation** : format de réponse unifié avec scores de confiance par attribut
6. **Nettoyage mémoire** : frame/image jetée immédiatement après inférence (cf. contrainte #3)

### Format de Réponse API
```json
{
  "faces": [
    {
      "bounding_box": {"x": 120, "y": 80, "width": 200, "height": 200},
      "emotion": {"label": "joie", "confidence": 0.87},
      "age_estimate": {"value": 28, "range": "24-32"},
      "gender": {"label": "femme", "confidence": 0.82},
      "facial_hair": {"barbe": false, "moustache": false, "confidence": 0.91},
      "hair": {"couleur": "brun", "longueur_estimee": "long", "confidence": 0.79},
      "eye_color": {"label": "marron", "method": "heuristique_hsv", "confidence": "indicative"}
    }
  ],
  "disclaimer": "Estimations statistiques du modèle, à but démonstratif uniquement.",
  "explainability_available": true
}
```

**Endpoint dédié pour l'explicabilité** (calcul à la demande, séparé de l'analyse standard
pour ne pas alourdir chaque requête) :
```json
// POST /api/analyze/explain { "image": "...", "attribute": "emotion" }
{
  "attribute": "emotion",
  "predicted_label": "joie",
  "heatmap_overlay_base64": "...",
  "explanation_note": "Zones du visage ayant le plus influencé cette prédiction"
}
```

---

## 🖥️ Backend FastAPI

```
app/
├── main.py
├── models/
│   ├── multitask_model.py       # Architecture PyTorch (backbone + têtes)
│   ├── checkpoints/              # Poids entraînés (versionnés, pas dans git — LFS ou storage externe)
│   └── eye_color_heuristic.py    # Approche classique (non-ML)
├── inference/
│   ├── face_detector.py          # Wrapper MediaPipe
│   ├── preprocessing.py          # Alignement, normalisation
│   ├── predictor.py              # Orchestration inférence complète
│   └── explainability.py         # Grad-CAM (pytorch-grad-cam), à la demande
├── api/
│   ├── analyze_image.py          # POST /api/analyze/image (upload photo)
│   ├── analyze_video.py          # POST /api/analyze/video (upload vidéo, échantillonnage frames)
│   ├── analyze_explain.py        # POST /api/analyze/explain (heatmap Grad-CAM à la demande)
│   └── ws_realtime.py            # WebSocket /ws/realtime (webcam live)
└── training/
    ├── train.py                  # Script d'entraînement multi-tâches
    ├── dataset_loaders.py        # Chargement FER2013, UTKFace, CelebA (sans label ethnicité)
    └── evaluate.py                # Métriques par tâche sur test set
```

### Endpoints
- `POST /api/analyze/image` — upload photo, retourne le JSON d'analyse
- `POST /api/analyze/video` — upload vidéo, échantillonne ~1 frame/seconde, retourne une analyse par frame échantillonnée + agrégat (émotion dominante sur la vidéo, etc.)
- `WS /ws/realtime` — le frontend envoie une frame webcam encodée toutes les ~500ms-1s, le serveur répond avec le JSON d'analyse (pas plus fréquent, pour ne pas saturer l'inférence)

---

## 🎨 Frontend Next.js

- **Mode Webcam** : accès `getUserMedia`, canvas overlay affichant bounding box + attributs
  en temps réel par-dessus le flux vidéo, écran de consentement obligatoire avant activation
- **Mode Upload** : dropzone photo/vidéo, mêmes composants de résultat que le mode webcam
- **Composants** : `FaceOverlay` (bounding box + labels sur canvas), `AttributeCard`
  (affichage résultat détaillé avec scores de confiance), `AttributeHeatmap` (overlay
  Grad-CAM à la demande, toggle "Pourquoi ?" par attribut), `ConsentModal`
- **Page "Comment ça marche"** : transparence sur les modèles utilisés, les datasets,
  les limites (cohérent avec l'approche méthodologie de Prisme-IA) — importante ici
  encore plus qu'ailleurs vu la sensibilité du sujet

---

## 📈 Entraînement & Évaluation

- Split train/val/test strict par dataset (80/10/10), aucune fuite entre splits
- Data augmentation : flip horizontal, variations de luminosité/contraste, léger crop aléatoire
  (attention : ne pas augmenter de façon à fausser l'âge/genre perçu, ex. pas de distorsion extrême)
- Suivi W&B : loss par tête, métriques par tâche (accuracy pour émotion/genre/pilosité,
  MAE pour âge, F1 multi-label pour cheveux), comparaison de plusieurs configs de pondération
- Évaluation finale sur test set jamais vu : matrice de confusion par tâche, analyse des
  erreurs (ex: quelles émotions sont confondues entre elles)
- **Test de biais à documenter** : évaluer les métriques du modèle genre/âge séparément
  par sous-groupes visibles dans le test set (ex: par tranche d'âge) pour vérifier
  l'absence de dégradation disproportionnée sur certains groupes — même à petite échelle,
  cette démarche de vérification de biais est très valorisée et rare chez les candidats

---

## 🧪 Tests

- Tests unitaires : preprocessing (alignement produit bien une image centrée), format
  de sortie API (schema JSON valide)
- Tests d'intégration : pipeline complet sur images de test connues avec résultats de
  référence attendus (tolérance sur l'âge, exact match sur émotion dominante)
- Test de confidentialité : vérifier qu'aucune image n'est écrite sur disque après un
  appel API (test automatisé qui inspecte le filesystem/tmp après inférence)

---

## 🚀 Roadmap de Développement

### Phase 1 — Datasets & Préparation (semaine 1)
- [ ] Téléchargement et exploration FER2013, UTKFace, CelebA
- [ ] Scripts de chargement (en excluant explicitement le label ethnicité d'UTKFace)
- [ ] Pipeline de préprocessing/alignement (MediaPipe)

### Phase 2 — Entraînement Multi-Task (semaines 2-3, sur Colab Pro+/RunPod)
- [ ] Architecture backbone partagé + têtes multiples
- [ ] Setup checkpointing robuste (sauvegarde périodique + reprise sur coupure de session)
- [ ] Premiers runs d'entraînement, suivi W&B
- [ ] Itération sur les pondérations de loss, comparaison multi-task vs modèles séparés
- [ ] Heuristique couleur des yeux (approche classique)
- [ ] Évaluation finale + analyse de biais par sous-groupe
- [ ] Upload du meilleur checkpoint vers Hugging Face Hub

### Phase 3 — Backend API (semaine 4)
- [ ] Endpoints upload image/vidéo
- [ ] WebSocket temps réel
- [ ] Téléchargement automatique du checkpoint depuis HF Hub au démarrage
- [ ] Module Grad-CAM + endpoint `/api/analyze/explain`
- [ ] Garantie de non-persistance des images (+ test automatisé)

### Phase 4 — Frontend & Démo Publique (semaine 5)
- [ ] Mode webcam avec overlay temps réel + écran de consentement
- [ ] Mode upload
- [ ] Bouton "Pourquoi ?" avec overlay heatmap Grad-CAM par attribut
- [ ] Page "Comment ça marche" (transparence méthodologie)
- [ ] Responsive, déploiement

### Phase 5 — Polish (optionnel)
- [ ] Comparaison de plusieurs architectures backbone (ResNet vs EfficientNet), documentée
- [ ] Export d'un rapport d'évaluation complet (façon mini-paper, réutilisable pour un post technique)

---

## 📝 README à Produire (Documentation Transversale)

- **Décisions d'architecture** : pourquoi multi-task plutôt que modèles séparés, pourquoi
  heuristique classique pour la couleur des yeux plutôt que du deep learning
- **Limites connues** : précision réduite sur la couleur/longueur des cheveux (pas de
  label direct dans CelebA), biais potentiels des datasets sources (à documenter après
  l'analyse de biais de la Phase 2), performance en conditions d'éclairage difficiles
- **Métriques** : accuracy/MAE/F1 par tâche sur le test set, résultats de l'analyse de biais
- **Note éthique** : rappel explicite des contraintes de la section "Contraintes
  Non-Négociables" ci-dessus, pour que quiconque lit le repo comprenne le périmètre
  volontairement limité du projet (pas d'identification de personnes, pas de stockage)

---

## ✅ Critères de Succès

- Accuracy émotion > 65% sur test set (FER2013 est un dataset difficile, c'est un score
  déjà correct comparé à la littérature)
- MAE âge < 6 ans sur test set
- Accuracy genre > 90%
- Démo webcam fonctionnelle avec latence perçue < 1s entre capture et affichage résultat
- Zéro image persistée après traitement (vérifié par test automatisé)
- Heatmap Grad-CAM générée en < 2s à la demande, cohérente visuellement (zones activées
  plausibles par rapport à l'attribut prédit)
- Page "Comment ça marche" et disclaimers visibles et clairs sur la démo publique
