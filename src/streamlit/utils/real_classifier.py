"""
Production multimodal classifier — uses real trained models.

Supports three prediction modes:
  - Text only:  TF-IDF + LinearSVC pipeline (F1=0.83)
  - Image only: Voting System with 3 models (F1~0.79)
  - Fusion:     Weighted average of both (60% image + 40% text, F1~0.85)

The text model uses decision_function + softmax to produce probabilities
from LinearSVC (which doesn't natively support predict_proba).
"""
import sys
import os
import joblib
import json
import numpy as np
import scipy.sparse as sp
from pathlib import Path

# Add project root to path for cross-module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config import MODELS_DIR, TEXT_MODEL_PATH, CATEGORY_MAPPING_PATH, FUSION_W_IMAGE, FUSION_W_TEXT
from src.models.predict_model import VotingPredictor


class MultimodalClassifier:
    """Loads all models once, exposes predict_text / predict_image / predict_fusion."""

    def __init__(self):
        # Fusion weights from config (single source of truth)
        self.w_text = FUSION_W_TEXT
        self.w_image = FUSION_W_IMAGE

        # 1. Category mapping (code -> human-readable name)
        try:
            with open(CATEGORY_MAPPING_PATH, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            # Handle nested JSON format: {"categories": {"2583": {"name": "Piscine", ...}}}
            if "categories" in raw and isinstance(raw["categories"], dict):
                self.mapping = {
                    code: f"{cat.get('emoji', '')} {cat['name']}".strip()
                    for code, cat in raw["categories"].items()
                }
            else:
                self.mapping = raw
        except Exception:
            self.mapping = {}

        # 2. Image model — Voting System (DINOv3 + XGBoost + EfficientNet)
        try:
            self.voting = VotingPredictor(MODELS_DIR)
            self.voting.load_models()
        except Exception as e:
            print(f"Image model error: {e}")
            self.voting = None

        # 3. Text model — TF-IDF FeatureUnion + LinearSVC
        try:
            self.text_model = joblib.load(TEXT_MODEL_PATH)
        except Exception as e:
            print(f"Text model error: {e}")
            self.text_model = None

    def _format_result(self, label, score):
        """Format a single prediction as {label, name, confidence}."""
        return {
            "label": str(label),
            "name": self.mapping.get(str(label), f"Produit Type {label}"),
            "confidence": float(score)
        }

    def predict_image(self, image_path):
        """Run image-only classification through the Voting System."""
        if not self.voting:
            return []
        try:
            raw_res = self.voting.predict(image_path)
            return [self._format_result(r['label'], r['confidence']) for r in raw_res]
        except Exception as e:
            print(f"Image prediction error: {e}")
            return []

    def predict_text(self, text):
        """
        Run text-only classification through LinearSVC.

        LinearSVC uses decision_function (not predict_proba), so we convert
        raw scores to probabilities via tempered softmax with T=0.3.
        Standard softmax (T=1) spreads probability too uniformly across 27
        classes because LinearSVC raw scores have small spread. T=0.3 produces
        confidence levels that reflect actual prediction quality.
        """
        if not self.text_model:
            return []
        try:
            if isinstance(text, str):
                text = [text]

            # Get probabilities from the sklearn pipeline
            if hasattr(self.text_model, "predict_proba"):
                probs = self.text_model.predict_proba(text)[0]
            elif hasattr(self.text_model, "decision_function"):
                scores = self.text_model.decision_function(text)[0]
                # Tempered softmax: T=0.3 sharpens the distribution so that
                # the winning class gets meaningful confidence (not ~4% uniform)
                T = 0.3
                scaled = (scores - np.max(scores)) / T
                exp_scores = np.exp(scaled)
                probs = exp_scores / exp_scores.sum()
            else:
                return []

            # Build results for all 27 classes, sorted by confidence
            results = []
            for i, class_id in enumerate(self.text_model.classes_):
                results.append(self._format_result(class_id, probs[i]))
            return sorted(results, key=lambda x: x['confidence'], reverse=True)

        except Exception as e:
            print(f"Text prediction error: {e}")
            return []

    def predict_fusion(self, text, image_path):
        """
        Late fusion: combine text and image scores with configurable weights.

        For each class, the fused score = w_text * text_score + w_image * image_score.
        This allows classes missed by one modality to be rescued by the other.
        """
        res_text = self.predict_text(text)
        res_image = self.predict_image(image_path)

        # Merge scores by label
        fusion_scores = {}
        for item in res_text:
            fusion_scores[item['label']] = item['confidence'] * self.w_text
        for item in res_image:
            label = item['label']
            fusion_scores[label] = fusion_scores.get(label, 0.0) + (item['confidence'] * self.w_image)

        # Sort and return
        final_results = [self._format_result(label, score) for label, score in fusion_scores.items()]
        return sorted(final_results, key=lambda x: x['confidence'], reverse=True)

    def explain_text(self, text):
        """Extract top TF-IDF features contributing to the predicted class."""
        if not self.text_model:
            return []
        try:
            if isinstance(text, str):
                text = [text]

            # Get the predicted class index
            scores = self.text_model.decision_function(text)[0]
            predicted_idx = int(np.argmax(scores))

            # Get TF-IDF feature vector for input text
            feats_step = self.text_model.named_steps.get('feats')
            clf_step = self.text_model.named_steps.get('clf')
            if feats_step is None or clf_step is None:
                return []

            tfidf_vec = feats_step.transform(text)
            if sp.issparse(tfidf_vec):
                tfidf_arr = tfidf_vec.toarray()[0]
            else:
                tfidf_arr = np.array(tfidf_vec[0])

            # Get coefficients for the predicted class
            coefs = clf_step.coef_[predicted_idx]

            # Contribution = tfidf_value * coefficient
            contributions = tfidf_arr * coefs
            nonzero = np.nonzero(contributions)[0]

            if len(nonzero) == 0:
                return []

            # Get feature names
            feature_names = feats_step.get_feature_names_out()

            # Sort by absolute contribution, take top 10
            top_indices = nonzero[np.argsort(np.abs(contributions[nonzero]))[-10:][::-1]]

            return [
                {"feature": str(feature_names[i]), "contribution": float(contributions[i])}
                for i in top_indices
            ]
        except Exception as e:
            print(f"Text explain error: {e}")
            return []

    def predict_image_detailed(self, image_path):
        """Run image classification with per-model breakdown."""
        if not self.voting:
            return {"results": [], "per_model": {}}
        try:
            detailed = self.voting.predict_detailed(image_path)
            results = [self._format_result(r['label'], r['confidence'])
                       for r in detailed['predictions']]
            # Add human-readable names to per-model
            per_model = {}
            for model_name, info in detailed['per_model'].items():
                per_model[model_name] = {
                    "label": info['top_label'],
                    "name": self.mapping.get(str(info['top_label']),
                                             f"Produit {info['top_label']}"),
                    "confidence": info['top_conf'],
                }
            return {"results": results, "per_model": per_model}
        except Exception as e:
            print(f"Image detailed error: {e}")
            return {"results": [], "per_model": {}}

    def predict_fusion_detailed(self, text, image_path):
        """Fusion with full explainability: text features + per-model image breakdown."""
        res_text = self.predict_text(text)
        img_detail = self.predict_image_detailed(image_path)
        text_explain = self.explain_text(text)

        # Merge scores for fusion
        fusion_scores = {}
        for item in res_text:
            fusion_scores[item['label']] = item['confidence'] * self.w_text
        for item in img_detail['results']:
            label = item['label']
            fusion_scores[label] = fusion_scores.get(label, 0.0) + (item['confidence'] * self.w_image)

        fusion_results = [self._format_result(label, score) for label, score in fusion_scores.items()]
        fusion_results = sorted(fusion_results, key=lambda x: x['confidence'], reverse=True)

        return {
            "fusion": fusion_results,
            "text_results": res_text,
            "image_results": img_detail['results'],
            "per_model": img_detail['per_model'],
            "text_explain": text_explain,
            "text_top": res_text[0] if res_text else None,
            "image_top": img_detail['results'][0] if img_detail['results'] else None,
        }
