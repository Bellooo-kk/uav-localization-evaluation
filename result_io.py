import pickle

import numpy as np


def frame_record(prediction, reference_centre, reference_rotation):
    reference_centre = np.asarray(reference_centre, dtype=float)
    reference_rotation = np.asarray(reference_rotation, dtype=float)
    if reference_centre.shape != (3,) or not np.isfinite(reference_centre).all():
        raise ValueError("The reference camera centre must be a finite 3-vector.")
    if reference_rotation.shape != (3, 3) or not np.isfinite(reference_rotation).all():
        raise ValueError("The reference rotation must be a finite 3-by-3 matrix.")
    if (not np.allclose(reference_rotation @ reference_rotation.T, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(reference_rotation), 1.0, atol=1e-6)):
        raise ValueError("The reference rotation must be a proper rotation matrix.")
    return {
        "prediction": prediction,
        "reference_centre": reference_centre.tolist(),
        "reference_rotation": reference_rotation.tolist(),
    }


def save_results(records, stream):
    pickle.dump(list(records), stream)


def load_results(stream):
    return pickle.load(stream)
