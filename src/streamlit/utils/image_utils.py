"""
Utilitaires de traitement d'images pour l'application Rakuten.

Ce module fournit des fonctions pour:
- Charger et valider des images uploadées
- Redimensionner les images pour le modèle
- Prétraiter les images pour l'extraction de features ResNet50
"""
from typing import Tuple, Optional
from pathlib import Path
import io
import sys

from PIL import Image
import numpy as np

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import IMAGE_CONFIG


def load_image_from_upload(uploaded_file) -> Image.Image:
    """
    Charge une image depuis un fichier uploadé Streamlit.

    Args:
        uploaded_file: Objet UploadedFile de Streamlit

    Returns:
        Image PIL en mode RGB

    Raises:
        ValueError: Si le fichier n'est pas une image valide
    """
    try:
        image = Image.open(uploaded_file)
        # Convertir en RGB si nécessaire (gère PNG avec transparence, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Impossible de charger l'image: {e}")


def validate_image(image: Image.Image) -> Tuple[bool, str]:
    """
    Valide une image selon les critères de l'application.

    Args:
        image: Image PIL à valider

    Returns:
        Tuple (is_valid, message)
    """
    # Vérifier les dimensions minimales
    min_size = 32
    if image.width < min_size or image.height < min_size:
        return False, f"Image trop petite (minimum {min_size}x{min_size})"

    # Vérifier les dimensions maximales
    max_size = 10000
    if image.width > max_size or image.height > max_size:
        return False, f"Image trop grande (maximum {max_size}x{max_size})"

    return True, "Image valide"


def resize_image(
    image: Image.Image,
    target_size: Tuple[int, int] = None,
    maintain_aspect_ratio: bool = True
) -> Image.Image:
    """
    Redimensionne une image vers la taille cible.

    Args:
        image: Image PIL à redimensionner
        target_size: Taille cible (width, height). Par défaut: config.IMAGE_CONFIG
        maintain_aspect_ratio: Si True, conserve le ratio et ajoute du padding

    Returns:
        Image PIL redimensionnée
    """
    if target_size is None:
        target_size = IMAGE_CONFIG["target_size"]

    if maintain_aspect_ratio:
        return _resize_with_padding(image, target_size)
    else:
        return image.resize(target_size, Image.Resampling.LANCZOS)


def _resize_with_padding(
    image: Image.Image,
    target_size: Tuple[int, int],
    fill_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Redimensionne une image en conservant le ratio et en ajoutant du padding.

    Args:
        image: Image PIL source
        target_size: Taille cible (width, height)
        fill_color: Couleur de remplissage RGB pour le padding

    Returns:
        Image PIL avec padding
    """
    target_w, target_h = target_size
    original_w, original_h = image.size

    # Calculer le ratio de redimensionnement
    ratio = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * ratio)
    new_h = int(original_h * ratio)

    # Redimensionner l'image
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Créer l'image avec padding
    padded = Image.new("RGB", target_size, fill_color)

    # Centrer l'image redimensionnée
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))

    return padded


def preprocess_for_resnet(image: Image.Image) -> np.ndarray:
    """
    Prétraite une image pour l'extraction de features ResNet50.

    Applique les transformations standard ImageNet:
    - Redimensionnement à 224x224
    - Normalisation des pixels
    - Expansion des dimensions pour le batch

    Args:
        image: Image PIL en RGB

    Returns:
        Array numpy de shape (1, 224, 224, 3) prêt pour ResNet50
    """
    # Redimensionner à 224x224
    resized = resize_image(image, (224, 224), maintain_aspect_ratio=True)

    # Convertir en array numpy
    img_array = np.array(resized, dtype=np.float32)

    # Normalisation ImageNet (preprocessing Keras)
    # Mode 'caffe': BGR, centré sur ImageNet mean
    img_array = img_array[..., ::-1]  # RGB -> BGR
    img_array[..., 0] -= 103.939
    img_array[..., 1] -= 116.779
    img_array[..., 2] -= 123.68

    # Ajouter la dimension batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """
    Convertit une image PIL en bytes.

    Args:
        image: Image PIL
        format: Format de sortie (PNG, JPEG, etc.)

    Returns:
        Bytes de l'image
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def create_thumbnail(
    image: Image.Image,
    max_size: Tuple[int, int] = (150, 150)
) -> Image.Image:
    """
    Crée une miniature d'une image.

    Args:
        image: Image PIL source
        max_size: Taille maximale de la miniature

    Returns:
        Image PIL miniature
    """
    thumbnail = image.copy()
    thumbnail.thumbnail(max_size, Image.Resampling.LANCZOS)
    return thumbnail


def get_image_info(image: Image.Image) -> dict:
    """
    Retourne les informations d'une image.

    Args:
        image: Image PIL

    Returns:
        Dict avec les informations de l'image
    """
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
        "size_str": f"{image.width}x{image.height}",
    }
