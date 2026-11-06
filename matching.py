import numpy as np


def match_features(query_points, query_descriptors, tile_points, tile_descriptors, *,
                   ratio_threshold=0.8):
    query_points, tile_points = np.asarray(query_points), np.asarray(tile_points)
    query_descriptors = np.asarray(query_descriptors, dtype=np.float64)
    tile_descriptors = np.asarray(tile_descriptors, dtype=np.float64)
    if (query_descriptors.ndim != 2 or tile_descriptors.ndim != 2
            or query_points.shape != (len(query_descriptors), 2)
            or tile_points.shape != (len(tile_descriptors), 2)
            or query_descriptors.shape[1] != tile_descriptors.shape[1]):
        raise ValueError("Each extractor must return aligned keypoints and descriptors.")
    if not 0 < ratio_threshold <= 1:
        raise ValueError("The ratio threshold must be in (0, 1].")
    if not len(query_points) or len(tile_points) < 2:
        return query_points[:0], tile_points[:0]
    if not np.isfinite(query_descriptors).all() or not np.isfinite(tile_descriptors).all():
        raise ValueError("Descriptors must be finite.")
    squared = np.sum(query_descriptors**2, axis=1)[:, None]
    squared = squared + np.sum(tile_descriptors**2, axis=1) - 2 * query_descriptors @ tile_descriptors.T
    np.maximum(squared, 0, out=squared)
    nearest = squared.argmin(axis=1)
    reverse = squared.argmin(axis=0)
    indices = np.arange(len(query_points))
    first = np.sqrt(squared[indices, nearest])
    second = np.sqrt(np.partition(squared, 1, axis=1)[:, 1])
    keep = (reverse[nearest] == indices) & (first / (second + 1e-8) < ratio_threshold)
    return query_points[keep], tile_points[nearest[keep]]
