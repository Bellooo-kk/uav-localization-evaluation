# Semantic-guided Local Feature Learning with Stability-Aware Detection for UAV Visual Localization

**Evaluation code for the accompanying manuscript**

The manuscript studies how scene semantics can guide local feature learning and help identify stable image regions for UAV visual localization. This repository provides Python tools for evaluating localization with user-provided imagery and reference maps.

Further information about the dataset used in this study will be provided after the paper is published.

The evaluation workflow builds on the AnyVisLoc benchmark, with additional metrics for rotation error and joint pose success, as well as input and boundary checks. [Ye et al. (2025)](https://arxiv.org/abs/2503.10692v1)

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
