import cv2
import numpy as np

from map_tiles import align_reference, tile_windows
from matching import match_features
from pose import estimate_pose
from retrieval import rank_windows


def localize_frame(query_image, reference_image, elevation, pixel_to_xy, camera_matrix, *,
                   query_gsd, reference_gsd, overlap_percent, describe_image, extract_features,
                   yaw_prior_deg=None, resize_ratio=1.0, reference_to_elevation=None,
                   distortion=None, reprojection_error=8.0):
    if elevation.ndim != 2:
        raise ValueError("The elevation raster must be two-dimensional.")
    affine = np.asarray(pixel_to_xy, dtype=float)
    if (affine.shape != (2, 3) or not np.isfinite(affine).all()
            or np.linalg.matrix_rank(affine[:, :2]) < 2):
        raise ValueError("pixel_to_xy must map map pixels to two independent world axes.")
    if reference_to_elevation is None:
        if elevation.shape != reference_image.shape[:2]:
            raise ValueError("Supply reference_to_elevation for a different elevation grid.")
    else:
        reference_to_elevation = np.asarray(reference_to_elevation, dtype=float)
        if (reference_to_elevation.shape != (2, 3)
                or not np.isfinite(reference_to_elevation).all()):
            raise ValueError("reference_to_elevation must be a finite 2-by-3 transform.")
    if not 0 < resize_ratio <= 1:
        raise ValueError("The query resize ratio must be in (0, 1].")

    aligned, inverse = reference_image, None
    if yaw_prior_deg is not None:
        aligned, inverse = align_reference(reference_image, yaw_prior_deg)
    windows = tile_windows(aligned, query_image.shape, query_gsd, reference_gsd, overlap_percent)
    candidates = rank_windows(query_image, aligned, windows, describe_image)
    if not candidates:
        return None

    fine_scale = query_gsd / reference_gsd
    reference_resize = resize_ratio / fine_scale
    query = query_image
    if resize_ratio < 1:
        query = cv2.resize(query, None, fx=resize_ratio, fy=resize_ratio)
    query_points, query_descriptors = extract_features(query)
    best = None
    for left, top, right, bottom in candidates:
        tile = cv2.resize(aligned[top:bottom, left:right], None,
                          fx=reference_resize, fy=reference_resize)
        tile_points, tile_descriptors = extract_features(tile)
        query_xy, tile_xy = match_features(query_points, query_descriptors,
                                          tile_points, tile_descriptors)
        query_limit = np.array(query.shape[:2][::-1])
        tile_limit = np.array(tile.shape[:2][::-1])
        valid = ((query_xy >= 0) & (query_xy < query_limit)).all(axis=1)
        valid &= ((tile_xy >= 0) & (tile_xy < tile_limit)).all(axis=1)
        image_xy = query_xy[valid] / resize_ratio
        map_xy = tile_xy[valid] / reference_resize + [left, top]
        if inverse is not None:
            map_xy = map_xy @ inverse[:, :2].T + inverse[:, 2]
        inside = ((map_xy >= 0) & (map_xy < reference_image.shape[:2][::-1])).all(axis=1)
        sample_xy = map_xy
        if reference_to_elevation is not None:
            sample_xy = map_xy @ reference_to_elevation[:, :2].T + reference_to_elevation[:, 2]
        inside &= ((sample_xy >= 0) & (sample_xy < elevation.shape[::-1])).all(axis=1)
        indices = sample_xy[inside].astype(int)
        z = elevation[indices[:, 1], indices[:, 0]]
        xy = map_xy[inside] @ affine[:, :2].T + affine[:, 2]
        pose = estimate_pose(np.column_stack((xy, z)), image_xy[inside], camera_matrix,
                             distortion, reprojection_error=reprojection_error)
        if pose is not None and (best is None or pose["inliers"] > best["inliers"]):
            best = pose
    return best
