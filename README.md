This repository contains evaluation code accompanying the manuscript *Semantic-guided Local Feature Learning with Stability-Aware Detection for UAV Visual Localization*. The localization workflow was adapted from the AnyVisLoc benchmark of Ye et al. (2025), with additions for rotation error, joint pose success rates, and input and boundary checks. [Benchmark paper](https://arxiv.org/abs/2503.10692v1)

| Module | Purpose |
| --- | --- |
| `map_tiles.py` | Generate reference map tiles based on query ground coverage. |
| `retrieval.py` | Rank candidate tiles using global image descriptors. |
| `matching.py` | Match local features using mutual nearest neighbours and a ratio test. |
| `pose.py` | Estimate camera pose using PnP with RANSAC. |
| `pipeline.py` | Run localization and select the candidate pose with the most inliers. |
| `result_io.py` | Read and write evaluation records in PKL format. |
| `metrics.py` | Compute localization metrics. |

Install the dependencies with `python -m pip install -r requirements.txt`. Call `localize_frame` with your own images, map inputs, and feature extraction functions. Use `frame_record` to prepare one record per query and `evaluate` to compute the metrics.

After the paper is published, readers may contact the authors to inquire about access to the ZJ-UAV dataset. Access requests are subject to review.
