import cv2
import numpy as np
from PIL import Image


def rank_windows(query_image, reference_image, windows, describe_image):
    windows = list(windows)
    if not windows:
        return []
    height, width = query_image.shape[:2]
    half = min(height, width) // 2
    if not half:
        raise ValueError("The query must be at least two pixels wide and high.")
    top, left = height // 2 - half, width // 2 - half
    square = query_image[top:top + 2 * half, left:left + 2 * half]
    query = np.asarray(Image.fromarray(square).resize((384, 384)))
    query_feature = np.asarray(describe_image(query), dtype=float)
    features = []
    for left, top, right, bottom in windows:
        tile = cv2.resize(reference_image[top:bottom, left:right], (384, 384))
        features.append(describe_image(tile))
    gallery = np.asarray(features, dtype=float)
    if (query_feature.ndim != 1 or not query_feature.size
            or gallery.shape != (len(windows), len(query_feature))
            or not np.isfinite(query_feature).all() or not np.isfinite(gallery).all()):
        raise ValueError("The retrieval callback must return finite vectors of equal length.")
    query_feature = query_feature / max(np.linalg.norm(query_feature), 1e-12)
    gallery = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
    order = np.argsort(-(gallery @ query_feature), kind="stable")
    return [windows[index] for index in order[:5]]
