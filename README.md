# Surface distance metrics

## Summary
When comparing multiple image segmentations, performance metrics that assess how closely the surfaces align can be a useful difference measure. This group of surface distance based measures computes the closest distances from all surface points on one segmentation to the points on another surface, and returns performance metrics between the two. This distance can be used alongside other metrics to compare segmented regions against a ground truth.

Surfaces are represented using surface elements with corresponding area, allowing for more consistent approximation of surface measures.

## Metrics included
This library computes the following performance metrics for segmentation:

- Average surface distance (see `compute_average_surface_distance`)
- Hausdorff distance (see `compute_robust_hausdorff`)
- Surface overlap (see `compute_surface_overlap_at_tolerance`)
- Surface dice (see `compute_surface_dice_at_tolerance`)
- Volumetric dice (see `compute_dice_coefficient`)

## Installation
First clone the repo, then install the dependencies and `surface-distance`
package via pip:

```shell
$ git clone https://github.com/deepmind/surface-distance.git
$ pip install surface-distance/
```

## Usage
For simple usage examples, see `surface_distance_test.py`.
