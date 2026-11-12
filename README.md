# UAV Visual Localization Evaluation

Python tools for evaluating UAV visual localization with imagery and reference maps. The workflow builds on the AnyVisLoc benchmark, with additional rotation-error and joint pose metrics, as well as input and boundary checks. [Ye et al. (2025)](https://arxiv.org/abs/2503.10692v1)

## Workflow

| Step | What it does | Code |
| --- | --- | --- |
| 1. Map tiling and retrieval | Generate and rank candidate reference tiles. | [`map_tiles.py`](map_tiles.py), [`retrieval.py`](retrieval.py) |
| 2. Matching and pose estimation | Match local features and estimate a camera pose for the query image. | [`pipeline.py`](pipeline.py), [`matching.py`](matching.py), [`pose.py`](pose.py) |
| 3. Result recording | Prepare and store per-query evaluation records in PKL format. | [`result_io.py`](result_io.py) |
| 4. Evaluation | Compute localization metrics from the evaluation records. | [`metrics.py`](metrics.py) |

## Quick start

Install the dependencies.

```bash
python -m pip install -r requirements.txt
```

Use your own imagery, reference-map inputs, and feature extraction functions.

1. Call [`localize_frame`](pipeline.py) to estimate a pose for each query image.
2. Use [`frame_record`](result_io.py) to pair each prediction with its reference pose.
3. Pass the resulting records to [`evaluate`](metrics.py) to compute the metrics.
