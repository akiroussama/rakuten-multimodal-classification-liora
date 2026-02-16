# Rakuten Product Classifier

Application de classification automatique de produits e-commerce en 27 catégories.

## Installation

```bash
cd src/streamlit
pip install -r requirements.txt
```

## Lancer l'app

```bash
streamlit run app.py
```

L'app tourne sur `http://localhost:8501`

## Structure

```
├── app.py              # Page d'accueil
├── pages/
│   ├── 1_Données       # Stats du dataset (84K produits, distribution)
│   ├── 2_Preprocessing # Pipeline NLP et image
│   ├── 3_Modèles       # Comparaison des 6 modèles (3 texte, 3 image)
│   ├── 4_Démo          # Tester la classification
│   ├── 5_Performance   # Métriques (accuracy, F1, confusion matrix)
│   ├── 6_Conclusions   # Résultats et perspectives
│   ├── 7_Qualité       # Tests automatisés et couverture
│   └── 8_Explicabilité # SHAP, Grad-CAM
├── utils/              # Code métier
└── tests/              # Tests pytest
```

## Fonctionnalités

- Classification texte ou image (pas de multimodal)
- 3 modèles texte : TF-IDF+SVM, TF-IDF+RF, CamemBERT
- 3 modèles image : ResNet50+SVM, ResNet50+RF, VGG16+SVM
- Comparaison côte à côte des modèles
- Visualisations interactives (Plotly)

## Tests

```bash
pip install -r requirements-test.txt
pytest
```

## Mode démo

L'app fonctionne en mode simulation (mock) sans les vrais modèles ML.
Pour utiliser les vrais modèles, placer les fichiers `.joblib` dans `models/`.

## Stack

- Streamlit
- Plotly
- scikit-learn
- Pillow
