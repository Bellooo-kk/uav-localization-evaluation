import numpy as np


def evaluate(records):
    position_errors, rotation_errors = [], []
    for record in records:
        prediction = record["prediction"]
        if prediction is None:
            position_errors.append(np.inf)
            rotation_errors.append(np.inf)
            continue
        delta = np.asarray(prediction["centre"]) - record["reference_centre"]
        position_errors.append(np.linalg.norm(delta[:2]))
        relative = np.asarray(prediction["rotation"]) @ np.asarray(record["reference_rotation"]).T
        cosine = np.clip((np.trace(relative) - 1) / 2, -1, 1)
        rotation_errors.append(np.degrees(np.arccos(cosine)))
    if not position_errors:
        raise ValueError("At least one query is required.")

    position = np.asarray(position_errors)
    rotation = np.asarray(rotation_errors)
    success = np.isfinite(position) & (position <= 50)
    finite_position = position[np.isfinite(position)]
    finite_rotation = rotation[np.isfinite(rotation)]
    metrics = {
        "R": float(success.mean() * 100),
        "PE": float(np.where(success, position, 50).mean()),
        "MTE": float(np.median(finite_position)) if finite_position.size else None,
        "MRE": float(np.median(finite_rotation)) if finite_rotation.size else None,
    }
    for distance in (1, 3, 5):
        metrics[f"SR@{distance}m"] = float((position <= distance).mean() * 100)
    for threshold in (5, 7, 10):
        joint = (position <= threshold) & (rotation <= threshold)
        metrics[f"SR@({threshold}m,{threshold}deg)"] = float(joint.mean() * 100)
    return metrics
