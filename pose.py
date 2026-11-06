import cv2
import numpy as np


def estimate_pose(world_points, image_points, camera_matrix, distortion=None,
                  *, reprojection_error=8.0):
    world_points = np.asarray(world_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        raise ValueError("world_points must have shape (N, 3).")
    if image_points.shape != (len(world_points), 2):
        raise ValueError("image_points must have shape (N, 2) and align with world_points.")
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
        raise ValueError("The camera matrix must be a finite 3-by-3 matrix.")
    if not np.isfinite(reprojection_error) or reprojection_error <= 0:
        raise ValueError("The reprojection threshold must be positive and finite.")
    distortion = None if distortion is None else np.asarray(distortion, dtype=np.float64)
    valid = np.isfinite(world_points).all(axis=1) & np.isfinite(image_points).all(axis=1)
    world_points, image_points = world_points[valid], image_points[valid]
    if len(world_points) < 5:
        return None

    origin = world_points.mean(axis=0)
    local_points = world_points - origin
    if np.linalg.matrix_rank(local_points) < 2:
        return None
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.ascontiguousarray(local_points),
        np.ascontiguousarray(image_points),
        camera_matrix,
        distortion,
        reprojectionError=reprojection_error,
        flags=cv2.SOLVEPNP_P3P,
    )
    if not success or inliers is None or len(inliers) < 4:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    translation = tvec.reshape(3)
    centre = origin - rotation.T @ translation
    if not np.isfinite(centre).all() or not np.isfinite(rotation).all():
        return None
    return {"centre": centre.tolist(), "rotation": rotation.tolist(), "inliers": len(inliers)}
