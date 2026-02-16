# Rakuten Product Classifier

Multimodal e-commerce product classification into 27 categories using text and image pipelines, fused via late fusion.

| Pipeline | Model | F1-Score |
|----------|-------|----------|
| Text | TF-IDF + LinearSVC | 83% |
| Image | Voting (DINOv3 + EfficientNet + XGBoost) | ~79% |
| **Fusion** | **Late Fusion (60/40)** | **~85%** |

**Team**: Johan Frachon, Liviu Andronic, Hery M. Ralaimanantsoa, Oussama Akir
**Program**: Machine Learning Engineer (RNCP) -- DataScientest x Mines Paris-PSL (Oct 2025)

**Live Demo**: [akiroussama-rakuten-classifier.hf.space](https://akiroussama-rakuten-classifier.hf.space)
**Hugging Face Space**: [akiroussama/rakuten-classifier](https://huggingface.co/spaces/akiroussama/rakuten-classifier)

---

## Quick Start

```bash
git clone [https://github.com/akiroussama/rakuten-multimodal-classification-liora](https://github.com/akiroussama/rakuten-multimodal-classification-liora)
cd rakuten-multimodal-classification-liora
pip install -r requirements.txt
```

Configure model downloads (one of):
- Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in `repo_id`
- Or set env var: `export HF_REPO_ID=akiroussama/rakuten-models`

```bash
streamlit run src/streamlit/app.py
```

Models (~2.5 GB) are downloaded automatically from [Hugging Face Hub](https://huggingface.co/akiroussama/rakuten-models) on first launch.

---

## Architecture

```
Product Input (text + image)
         |
    +----+----+
    v         v
  TEXT       IMAGE
    |         |
 TF-IDF    +--+----------+
    |       v  v          v
 LinearSVC  DINOv3  EfficientNet  XGBoost
  (83%)     (w=4/7)  (w=2/7)     (w=1/7)
    |         +------+------+
    |            VOTING (~79%)
    |                |
    +----- LATE FUSION (60% image / 40% text) --- ~85%
```

---

## Project Structure

```
.
├── src/
│   ├── features/
│   │   └── build_features.py        # Image preprocessing (DINOv3 518px / standard 224px)
│   ├── models/
│   │   └── predict_model.py         # VotingPredictor — 3-model ensemble with sharpening
│   └── streamlit/
│       ├── app.py                   # Home page & entry point
│       ├── config.py                # Paths, fusion weights, HF repo ID
│       ├── pages/                   # 5 interactive pages
│       ├── utils/                   # Classifiers, data loading, preprocessing
│       ├── assets/                  # CSS + model charts
│       └── tests/                   # Unit, integration, robustness tests
├── models/                          # Downloaded at runtime from HF Hub (gitignored)
├── .streamlit/                      # Streamlit config + secrets template
├── requirements.txt
└── LICENSE
```

---

## Models

All model weights are hosted on [Hugging Face Hub](https://huggingface.co/akiroussama/rakuten-models) and downloaded automatically at runtime.

| File | Size | Role |
|------|------|------|
| `M1_IMAGE_DeepLearning_DINOv3.pth` | 1.2 GB | DINOv3 ViT-Large (79.43% standalone) |
| `M3_IMAGE_Classic_EfficientNetB0.pth` | 16 MB | EfficientNet-B0 CNN (66.63%) |
| `M2_IMAGE_XGBoost_Encoder.pkl` | <1 KB | Label encoder for XGBoost |
| `text_classifier.joblib` | 32 MB | TF-IDF FeatureUnion + LinearSVC (83%) |
| `category_mapping.json` | 4 KB | 27-category code-to-name mapping |

XGBoost model (`M2_IMAGE_Classic_XGBoost.json`) is optional. The Voting System falls back to DINOv3 + EfficientNet (weights 4:2) when unavailable.

---

## Tests

```bash
cd src/streamlit
python -m pytest tests/test_robustness.py -v
```

38 tests covering: model download fallbacks, graceful degradation with missing models, voting weight correctness, error message clarity, prediction format regression, 27-class integrity, and app structure validation.

---

## License

MIT
