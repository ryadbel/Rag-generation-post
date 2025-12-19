# 🧠 ASphere — RAG Generator

Générateur de posts pour réseaux sociaux avec IA et RAG (Retrieval Augmented Generation)

## ✨ Fonctionnalités

- 🎨 **Génération automatique** de posts avec texte et images
- 📚 **RAG optionnel** : enrichissez vos posts avec du contexte depuis des URLs ou fichiers PDF
- 🔄 **Régénération d'images** avec conservation du style
- 🎯 **Multi-posts** : générez plusieurs posts d'un coup (ex: "génère 3 posts")
- 🖼️ **Images DALL-E 3** haute qualité

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone https://github.com/ryadbel/Rag-generation-post.git
cd asphere
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Configuration

Créez un fichier `.env` à la racine :

```bash
cp .env.example .env
```

Éditez `.env` et ajoutez votre clé OpenAI :

```
OPENAI_API_KEY=sk-...votre_clé_ici
```

**Comment obtenir votre clé API :**
1. Allez sur https://platform.openai.com/api-keys
2. Créez une nouvelle clé API
3. Copiez-la dans votre `.env`

## 🎮 Utilisation

### Démarrer le serveur backend

```bash
uvicorn asphere_backend:app --reload
```

Le serveur démarre sur `http://127.0.0.1:8000`

### Ouvrir l'interface web

Ouvrez simplement `asphere_frontend.html` dans votre navigateur.

## 📖 Guide d'utilisation

### Sans RAG (mode simple)

1. Allez directement à la section "3️⃣ Générer des posts"
2. Entrez votre prompt (ex: "Génère 3 posts LinkedIn sur l'IA")
3. Laissez la case RAG **décochée**
4. Cliquez sur "⚡ Générer les posts"

### Avec RAG (mode enrichi)

1. **Section 1** : Indexer vos sources
   - Ajoutez des URLs (une par ligne)
   - Et/ou uploadez des fichiers PDF
   - Cliquez sur "📚 Indexer les sources"
   
2. **Section 2** (optionnel) : Tester le retrieval
   - Entrez une requête
   - Voyez quels chunks sont récupérés
   
3. **Section 3** : Générer avec contexte
   - Entrez votre prompt
   - **Cochez** "Utiliser le contexte RAG"
   - Cliquez sur "⚡ Générer les posts"

### Régénérer une image

1. Dans un post généré, scrollez vers le bas
2. Entrez une nouvelle description dans le champ de texte
3. Cliquez sur "🔁 Régénérer l'image"

## 💡 Exemples de prompts

### Simples
```
Génère 1 post LinkedIn sur la cybersécurité
```

### Multi-posts
```
Génère 3 posts Instagram sur le développement durable
```

### Avec RAG
```
Génère 2 posts basés sur les articles indexés à propos de notre nouveau produit
```

## 🛠️ Architecture

```
asphere/
├── asphere_backend.py      # API FastAPI
├── asphere_frontend.html   # Interface web
├── requirements.txt        # Dépendances Python
├── .env                    # Variables d'environnement (à créer)
└── data/                   # Dossier de données (auto-créé)
    ├── uploads/            # Fichiers uploadés
    ├── vectorstore/        # Index FAISS
    └── media_history.json  # Historique des générations
```

## 🔧 Endpoints API

### Santé
```
GET /health
```

### RAG
```
POST /rag/ingest        # Indexer des sources
GET  /rag/debug         # Tester le retrieval
GET  /rag/status        # Vérifier si RAG initialisé
```

### Génération
```
POST /generate-with-media  # Générer posts + images
POST /image/regenerate     # Régénérer une image
```

### Historique
```
GET    /history           # Récupérer l'historique
DELETE /history/{id}      # Marquer comme oublié
```

## ⚠️ Corrections principales

### Problème résolu : 400 Bad Request

**Causes identifiées :**
1. ✅ RAG activé par défaut sans index → Changé à `use_rag: false` par défaut
2. ✅ Pas de gestion d'erreur frontend → Ajout de messages d'erreur clairs
3. ✅ Pas de vérification du statut RAG → Ajout de `/rag/status`

**Améliorations apportées :**
- Toggle RAG désactivé si index absent
- Badge de statut RAG en temps réel
- Messages d'erreur explicites
- Meilleure gestion des exceptions
- Interface plus intuitive

## 📝 Notes techniques

### Génération d'images

- Modèle : DALL-E 3
- Format : 1024x1024
- Prompt optimisé pour éviter le texte dans l'image
- Upload via tmpfiles.org (temporaire)

### RAG

- Embeddings : OpenAI `text-embedding-3-small`
- Vector store : FAISS
- Chunking : 1000 tokens avec overlap de 150
- Par défaut : top 5 chunks

### LLM

- Modèle : GPT-4o-mini
- Format : JSON structuré
- Détection automatique du nombre de posts

## 🐛 Dépannage

### "RAG activé mais index absent"
→ Indexez d'abord des sources dans la section 1

### "Le prompt est vide"
→ Entrez un texte dans le champ de génération

### Images ne s'affichent pas
→ Vérifiez votre connexion internet (upload tmpfiles.org)

### Erreur OpenAI
→ Vérifiez votre clé API et vos crédits OpenAI

## 📊 Limitations

- Images temporaires (tmpfiles.org expire après quelques heures)
- Pas d'authentification
- Pas de persistance des sessions
- Max 5 posts par génération

## 🔮 Améliorations futures

- [ ] Support vidéo
- [ ] Stockage permanent des images
- [ ] Multi-utilisateurs avec auth
- [ ] Templates de posts personnalisables
- [ ] Export direct vers réseaux sociaux
- [ ] Analytics des posts générés

## 📄 Licence

MIT

## 🤝 Contribution

Les contributions sont bienvenues ! N'hésitez pas à ouvrir une issue ou une PR.