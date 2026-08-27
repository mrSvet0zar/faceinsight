# Déploiement — FaceInsight

Trois étapes : poids sur Hugging Face Hub → backend sur Railway → frontend
sur Vercel. Aucun poids ni secret ne transite par le repo Git.

## 1. Upload du checkpoint validé vers HF Hub

Prérequis : compte Hugging Face, token **write** (Settings → Access Tokens),
ajouté dans `backend/.env` :

```
HF_TOKEN=hf_xxx
```

Depuis `backend/`, avec le `best.pth` du meilleur run (téléchargé depuis
Google Drive) :

```bash
python -m app.training.upload_checkpoint --checkpoint chemin/vers/best.pth --repo <user>/faceinsight-weights --eval-report ../reports/eval_baseline.json
```

Le script retire l'état optimizer/scheduler (fichier ~2× plus léger) et crée
le repo HF en **privé** par défaut.

Test local avec ce checkpoint distant :

```
# backend/.env
FACEINSIGHT_HF_REPO=<user>/faceinsight-weights
```

puis démarrer l'API — le log de démarrage doit afficher
`loaded checkpoint ... (epoch N)` et `/api/health` doit répondre
`"model_trained": true`.

## 2. Backend sur Railway

1. railway.app → New Project → Deploy from GitHub repo → `faceinsight`.
2. Settings du service : **Root Directory = `backend`** (Railway détecte le
   `Dockerfile`).
3. Variables d'environnement :
   | Variable | Valeur |
   |---|---|
   | `FACEINSIGHT_HF_REPO` | `<user>/faceinsight-weights` |
   | `HF_TOKEN` | token HF **read** (repo privé) |
   | `ALLOWED_ORIGINS` | `https://<projet>.vercel.app` (ajoutable après l'étape 3) |
4. Déployer, puis vérifier `https://<service>.up.railway.app/api/health`.

Notes :
- Inférence CPU : ResNet-18 tourne confortablement sur le plan de base ;
  prévoir ~1 Go de RAM (torch + mediapipe).
- Le checkpoint est téléchargé au premier démarrage puis mis en cache dans
  l'image du conteneur en cours d'exécution.

## 3. Frontend sur Vercel

1. vercel.com → Add New Project → importer le repo → **Root Directory =
   `frontend`** (framework Next.js auto-détecté).
2. Variable d'environnement :
   | Variable | Valeur |
   |---|---|
   | `NEXT_PUBLIC_API_URL` | `https://<service>.up.railway.app` |
3. Déployer. Le WebSocket passe automatiquement en `wss://` (dérivé de
   l'URL API).
4. Revenir sur Railway et fixer `ALLOWED_ORIGINS` sur l'URL Vercel exacte.

## Vérifications post-déploiement

- `/api/health` → `{"status": "ok", "model_trained": true}`
- Mode Photo : upload d'un portrait → attributs + « Pourquoi ? » (heatmap < 2 s)
- Mode Webcam : consentement → analyse en direct, latence affichée < 1 s
- La page « Comment ça marche » est accessible publiquement
