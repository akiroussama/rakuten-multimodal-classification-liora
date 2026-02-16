"""
Voting Predictor — Ensemble inference engine for image classification.

Combines three models via weighted soft voting:
  - M1: DINOv3 ViT-Large (weight 4/7) — self-supervised features, 79.43%
  - M2: XGBoost on ResNet50 features (weight 1/7) — best single model at 85.32%
  - M3: EfficientNet-B0 (weight 2/7) — lightweight CNN for diversity, 66.63%

XGBoost probabilities are "sharpened" (p^3 / sum(p^3)) to prevent
its flat distributions from diluting confident predictions from DINOv3.
"""
import torch
import torch.nn.functional as F
import xgboost as xgb
import timm, joblib, numpy as np
from pathlib import Path
from torchvision import models
import torch.nn as nn
from src.features.build_features import preprocess_image


class VotingPredictor:
    """Loads 3 image models and returns fused top-5 predictions."""

    def __init__(self, models_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mdir = Path(models_dir)
        self.loaded = False

    def load_models(self):
        """Load all three models + label encoder into memory (lazy, once)."""
        if self.loaded:
            return

        # M1 — DINOv3 ViT-Large fine-tuned on 27 Rakuten classes
        # Use mmap + assign to avoid 3x memory spike on large models
        self.m1 = timm.create_model(
            'vit_large_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=27
        )
        _sd = torch.load(
            self.mdir / "M1_IMAGE_DeepLearning_DINOv3.pth",
            map_location=self.device, mmap=True, weights_only=True,
        )
        self.m1.load_state_dict(_sd, assign=True)

        # M2 — XGBoost classifier on ResNet50 features (2048-dim)
        self.has_xgboost = False
        xgb_path = self.mdir / "M2_IMAGE_Classic_XGBoost.json"
        if xgb_path.exists():
            self.m2 = xgb.XGBClassifier()
            self.m2.load_model(str(xgb_path))
            self.le = joblib.load(self.mdir / "M2_IMAGE_XGBoost_Encoder.pkl")

            # ResNet50 feature extractor (headless) — feeds M2
            res = models.resnet50(weights=None)
            self.ext = nn.Sequential(*list(res.children())[:-1])
            self.ext.to(self.device).eval()
            self.has_xgboost = True

        # M3 — EfficientNet-B0 with custom 27-class head
        self.m3 = models.efficientnet_b0(weights=None)
        self.m3.classifier[1] = nn.Linear(1280, 27)
        self.m3.load_state_dict(
            torch.load(self.mdir / "M3_IMAGE_Classic_EfficientNetB0.pth", map_location=self.device)
        )

        # Move all torch models to device and set eval mode
        for m in [self.m1, self.m3]:
            m.to(self.device).eval()
        self.loaded = True

    def predict(self, img_p):
        """
        Run voting inference on a single image.

        Args:
            img_p: Path to the image file.

        Returns:
            List of top-5 dicts: [{"label": str, "confidence": float}, ...]
        """
        if not self.loaded:
            self.load_models()

        with torch.no_grad():
            # M1 — DINOv3 (518x518 input)
            p1 = F.softmax(
                self.m1(preprocess_image(img_p, "dino").to(self.device)), dim=1
            ).cpu().numpy()[0]

            # Shared 224x224 input for M2 and M3
            i2 = preprocess_image(img_p, "standard").to(self.device)

            # M3 — EfficientNet-B0
            p3 = F.softmax(self.m3(i2), dim=1).cpu().numpy()[0]

            if self.has_xgboost:
                # M2 — XGBoost on ResNet50 features
                f = self.ext(i2).squeeze().cpu().numpy().reshape(1, -1)
                raw_p2 = self.m2.predict_proba(f)[0]

                # Sharpening: raise to power 3, then renormalize.
                # This forces XGBoost to commit to a class instead of
                # spreading probability uniformly across all 27 classes.
                sharp_p2 = np.power(raw_p2, 3)
                p2 = sharp_p2 / sharp_p2.sum()

                # Weighted soft vote: DINOv3=4, EfficientNet=2, XGBoost=1
                f_p = (4.0 * p1 + 1.0 * p2 + 2.0 * p3) / 7.0
            else:
                # Fallback: DINOv3 + EfficientNet only (weights 4:2)
                f_p = (4.0 * p1 + 2.0 * p3) / 6.0

        # Return top-5 predictions with original Rakuten category codes
        ids = np.argsort(f_p)[-5:][::-1]

        # Use label encoder if available (XGBoost), otherwise use class index
        if self.has_xgboost:
            labels = [str(self.le.inverse_transform([i])[0]) for i in ids]
        else:
            # Load category mapping to get actual Rakuten codes
            import json
            mapping_path = self.mdir / "category_mapping.json"
            if mapping_path.exists():
                with open(mapping_path, 'r', encoding='utf-8') as mf:
                    cat_map = json.load(mf)
                if "categories" in cat_map:
                    code_list = sorted(cat_map["categories"].keys(), key=int)
                else:
                    code_list = sorted(cat_map.keys(), key=int)
                labels = [code_list[i] if i < len(code_list) else str(i) for i in ids]
            else:
                labels = [str(i) for i in ids]

        return [
            {"label": labels[j], "confidence": float(f_p[ids[j]])}
            for j in range(len(ids))
        ]

    def predict_detailed(self, img_p):
        """Run voting inference and return per-model breakdown for explainability."""
        if not self.loaded:
            self.load_models()

        with torch.no_grad():
            p1 = F.softmax(
                self.m1(preprocess_image(img_p, "dino").to(self.device)), dim=1
            ).cpu().numpy()[0]

            i2 = preprocess_image(img_p, "standard").to(self.device)
            p3 = F.softmax(self.m3(i2), dim=1).cpu().numpy()[0]

            p2 = None
            if self.has_xgboost:
                f = self.ext(i2).squeeze().cpu().numpy().reshape(1, -1)
                raw_p2 = self.m2.predict_proba(f)[0]
                sharp_p2 = np.power(raw_p2, 3)
                p2 = sharp_p2 / sharp_p2.sum()
                f_p = (4.0 * p1 + 1.0 * p2 + 2.0 * p3) / 7.0
            else:
                f_p = (4.0 * p1 + 2.0 * p3) / 6.0

        # Build label mapping (reuse same logic as predict())
        import json
        if self.has_xgboost:
            code_list = [str(self.le.inverse_transform([i])[0]) for i in range(27)]
        else:
            mapping_path = self.mdir / "category_mapping.json"
            if mapping_path.exists():
                with open(mapping_path, 'r', encoding='utf-8') as mf:
                    cat_map = json.load(mf)
                if "categories" in cat_map:
                    code_list = sorted(cat_map["categories"].keys(), key=int)
                else:
                    code_list = sorted(cat_map.keys(), key=int)
            else:
                code_list = [str(i) for i in range(27)]

        # Top-5 final predictions
        ids = np.argsort(f_p)[-5:][::-1]
        predictions = [
            {"label": code_list[ids[j]] if ids[j] < len(code_list) else str(ids[j]),
             "confidence": float(f_p[ids[j]])}
            for j in range(len(ids))
        ]

        # Per-model top prediction
        def _top(proba):
            idx = int(np.argmax(proba))
            lbl = code_list[idx] if idx < len(code_list) else str(idx)
            return {"top_label": lbl, "top_conf": float(proba[idx])}

        per_model = {
            "DINOv3": _top(p1),
            "EfficientNet": _top(p3),
        }
        if p2 is not None:
            per_model["XGBoost"] = _top(p2)

        return {
            "predictions": predictions,
            "per_model": per_model,
        }
