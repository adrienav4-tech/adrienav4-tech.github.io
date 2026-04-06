# Projet Interface RAG

Ce dépôt contient le code source du projet P3A-5. L'objectif de ce projet est de fournir une interface d'interrogation en langage naturel basée sur une architecture RAG (Retrieval-Augmented Generation) fonctionnant intégralement en local. 

Le système permet d'analyser et d'extraire des informations de bases documentaires spécifiques tout en garantissant la confidentialité des données via l'utilisation d'Ollama.

## Caractéristiques Techniques

- Traitement du Langage Naturel (NLP) : Utilisation de modèles de langage exécutés localement pour la compréhension et la génération de réponses.
- Recherche Vectorielle : Indexation et recherche de similarité des documents via la base de données vectorielle ChromaDB.
- Pipeline d'Ingestion : Scripts automatisés pour le prétraitement, le découpage et la vectorisation des jeux de données bruts (ex: fichiers CSV).
- Interface Utilisateur : Client web léger assurant la communication avec le backend de traitement.

## Pile Technologique

- Langage principal : Python
- Modèles d'IA locaux : Ollama (support pour Mistral, Llama, etc.)
- Base de données vectorielle : ChromaDB
- Frameworks d'orchestration LLM : LangChain
- Interface : HTML, CSS, JavaScript

## Architecture du Dépôt

Le projet est structuré de la manière suivante :

```text
├── chroma_db/            # Répertoire persistant de la base de données vectorielle principale
├── db_centrale_lyon/     # Répertoire persistant spécifique au contexte de Centrale Lyon
├── backend.py            # Serveur d'application gérant l'API et la communication avec le LLM
├── emissions.csv         # Jeu de données source pour l'analyse des bilans d'émissions
├── index.html            # Point d'entrée de l'interface client
├── ingest.py             # Script de traitement et d'indexation des données dans ChromaDB
└── rag.py                # Module central implémentant la logique de récupération et de génération
