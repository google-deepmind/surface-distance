# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple tests for surface metric computations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import google3
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import surface_distance
from surface_distance.surface_distance import metrics


class SurfaceDistanceTest(parameterized.TestCase, absltest.TestCase):

  def _assert_almost_equal(self, expected, actual, places):
    """Assertion wrapper correctly handling NaN equality."""
    if np.isnan(expected) and np.isnan(actual):
      return
    self.assertAlmostEqual(expected, actual, places)

  def _assert_metrics(self,
                      surface_distances, mask_gt, mask_pred,
                      expected_average_surface_distance,
                      expected_hausdorff_100,
                      expected_hausdorff_95,
                      expected_surface_overlap_at_1mm,
                      expected_surface_dice_at_1mm,
                      expected_volumetric_dice,
                      places=3):
    actual_average_surface_distance = (
        surface_distance.compute_average_surface_distance(surface_distances))
    for i in range(2):
      self._assert_almost_equal(
          expected_average_surface_distance[i],
          actual_average_surface_distance[i],
          places=places)

    self._assert_almost_equal(
        expected_hausdorff_100,
        surface_distance.compute_robust_hausdorff(surface_distances, 100),
        places=places)

    self._assert_almost_equal(
        expected_hausdorff_95,
        surface_distance.compute_robust_hausdorff(surface_distances, 95),
        places=places)

    actual_surface_overlap_at_1mm = (
        surface_distance.compute_surface_overlap_at_tolerance(
            surface_distances, tolerance_mm=1))
    for i in range(2):
      self._assert_almost_equal(
          expected_surface_overlap_at_1mm[i],
          actual_surface_overlap_at_1mm[i],
          places=places)

    self._assert_almost_equal(
        expected_surface_dice_at_1mm,
        surface_distance.compute_surface_dice_at_tolerance(
            surface_distances, tolerance_mm=1),
        places=places)

    self._assert_almost_equal(
        expected_volumetric_dice,
        surface_distance.compute_dice_coefficient(mask_gt, mask_pred),
        places=places)

  @parameterized.parameters((
      np.zeros([2, 2, 2], dtype=bool),
      np.zeros([2, 2], dtype=bool),
      [1, 1],
  ), (
      np.zeros([2, 2], dtype=bool),
      np.zeros([2, 2, 2], dtype=bool),
      [1, 1],
  ), (
      np.zeros([2, 2], dtype=bool),
      np.zeros([2, 2], dtype=bool),
      [1, 1, 1],
  ))
  def test_compute_surface_distances_raises_on_incompatible_shapes(
      self, mask_gt, mask_pred, spacing_mm):
    with self.assertRaisesRegex(ValueError,
                                'The arguments must be of compatible shape'):
      surface_distance.compute_surface_distances(mask_gt, mask_pred, spacing_mm)

  @parameterized.parameters((
      np.zeros([2], dtype=bool),
      np.zeros([2], dtype=bool),
      [1],
  ), (
      np.zeros([2, 2, 2, 2], dtype=bool),
      np.zeros([2, 2, 2, 2], dtype=bool),
      [1, 1, 1, 1],
  ))
  def test_compute_surface_distances_raises_on_invalid_shapes(
      self, mask_gt, mask_pred, spacing_mm):
    with self.assertRaisesRegex(ValueError,
                                'Only 2D and 3D masks are supported'):
      surface_distance.compute_surface_distances(mask_gt, mask_pred, spacing_mm)


class SurfaceDistance2DTest(SurfaceDistanceTest, parameterized.TestCase):

  def test_on_2_pixels_2mm_away(self):
    mask_gt = np.zeros((128, 128), bool)
    mask_pred = np.zeros((128, 128), bool)
    mask_gt[50, 70] = 1
    mask_pred[50, 72] = 1
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(2, 1))

    diag = 0.5 * math.sqrt(2**2 + 1**2)
    expected_distances = {
        'surfel_areas_gt': np.asarray([diag, diag, diag, diag]),
        'surfel_areas_pred': np.asarray([diag, diag, diag, diag]),
        'distances_gt_to_pred': np.asarray([1., 1., 2., 2.]),
        'distances_pred_to_gt': np.asarray([1., 1., 2., 2.]),
    }
    self.assertEqual(len(expected_distances), len(surface_distances))
    for key, expected_value in expected_distances.items():
      np.testing.assert_array_equal(expected_value, surface_distances[key])

    self._assert_metrics(
        surface_distances,
        mask_gt,
        mask_pred,
        expected_average_surface_distance=(1.5, 1.5),
        expected_hausdorff_100=2.0,
        expected_hausdorff_95=2.0,
        expected_surface_overlap_at_1mm=(0.5, 0.5),
        expected_surface_dice_at_1mm=0.5,
        expected_volumetric_dice=0.0)

  def test_two_squares_shifted_by_one_pixel(self):
    # We make sure we do not have active pixels on the border of the image,
    # because this will add additional 2D surfaces on the border of the image
    # because the image is padded with background.
    mask_gt = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=bool)

    mask_pred = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=bool)

    vertical = 2
    horizontal = 1
    diag = 0.5 * math.sqrt(horizontal**2 + vertical**2)
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(vertical, horizontal))

    # We go from top left corner, clockwise to describe the surfaces and
    # distances. The 2 surfaces are:
    #
    #  /-\  /-\
    #  | |  | |
    #  \-/  | |
    #       \-/
    expected_surfel_areas_gt = np.asarray(
        [diag, horizontal, diag, vertical, diag, horizontal, diag, vertical])
    expected_surfel_areas_pred = np.asarray([
        diag, horizontal, diag, vertical, vertical, diag, horizontal, diag,
        vertical, vertical
    ])
    expected_distances_gt_to_pred = np.asarray([0] * 5 + [horizontal] + [0] * 2)
    expected_distances_pred_to_gt = np.asarray([0] * 5 + [vertical] * 3 +
                                               [0] * 2)

    # We sort these using the same sorting algorithm
    (expected_distances_gt_to_pred, expected_surfel_areas_gt) = (
        metrics._sort_distances_surfels(expected_distances_gt_to_pred,
                                        expected_surfel_areas_gt))
    (expected_distances_pred_to_gt, expected_surfel_areas_pred) = (
        metrics._sort_distances_surfels(expected_distances_pred_to_gt,
                                        expected_surfel_areas_pred))

    expected_distances = {
        'surfel_areas_gt': expected_surfel_areas_gt,
        'surfel_areas_pred': expected_surfel_areas_pred,
        'distances_gt_to_pred': expected_distances_gt_to_pred,
        'distances_pred_to_gt': expected_distances_pred_to_gt,
    }

    self.assertEqual(len(expected_distances), len(surface_distances))
    for key, expected_value in expected_distances.items():
      np.testing.assert_array_equal(expected_value, surface_distances[key])

    self._assert_metrics(
        surface_distances,
        mask_gt,
        mask_pred,
        expected_average_surface_distance=(
            surface_distance.compute_average_surface_distance(
                expected_distances)),
        expected_hausdorff_100=(surface_distance.compute_robust_hausdorff(
            expected_distances, 100)),
        expected_hausdorff_95=surface_distance.compute_robust_hausdorff(
            expected_distances, 95),
        expected_surface_overlap_at_1mm=(
            surface_distance.compute_surface_overlap_at_tolerance(
                expected_distances, tolerance_mm=1)),
        expected_surface_dice_at_1mm=(
            surface_distance.compute_surface_dice_at_tolerance(
                surface_distances, tolerance_mm=1)),
        expected_volumetric_dice=(surface_distance.compute_dice_coefficient(
            mask_gt, mask_pred)))

  def test_empty_prediction_mask(self):
    mask_gt = np.zeros((128, 128), bool)
    mask_pred = np.zeros((128, 128), bool)
    mask_gt[50, 60] = 1
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2))
    self._assert_metrics(
        surface_distances,
        mask_gt,
        mask_pred,
        expected_average_surface_distance=(np.inf, np.nan),
        expected_hausdorff_100=np.inf,
        expected_hausdorff_95=np.inf,
        expected_surface_overlap_at_1mm=(0.0, np.nan),
        expected_surface_dice_at_1mm=0.0,
        expected_volumetric_dice=0.0)

  def test_empty_ground_truth_mask(self):
    mask_gt = np.zeros((128, 128), bool)
    mask_pred = np.zeros((128, 128), bool)
    mask_pred[50, 60] = 1
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2))
    self._assert_metrics(
        surface_distances,
        mask_gt,
        mask_pred,
        expected_average_surface_distance=(np.nan, np.inf),
        expected_hausdorff_100=np.inf,
        expected_hausdorff_95=np.inf,
        expected_surface_overlap_at_1mm=(np.nan, 0.0),
        expected_surface_dice_at_1mm=0.0,
        expected_volumetric_dice=0.0)

  def test_both_empty_masks(self):
    mask_gt = np.zeros((128, 128), bool)
    mask_pred = np.zeros((128, 128), bool)
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2))
    self._assert_metrics(
        surface_distances,
        mask_gt,
        mask_pred,
        expected_average_surface_distance=(np.nan, np.nan),
        expected_hausdorff_100=np.inf,
        expected_hausdorff_95=np.inf,
        expected_surface_overlap_at_1mm=(np.nan, np.nan),
        expected_surface_dice_at_1mm=np.nan,
        expected_volumetric_dice=np.nan)


class SurfaceDistance3DTest(SurfaceDistanceTest):

  def test_on_2_pixels_2mm_away(self):
    mask_gt = np.zeros((128, 128, 128), bool)
    mask_pred = np.zeros((128, 128, 128), bool)
    mask_gt[50, 60, 70] = 1
    mask_pred[50, 60, 72] = 1
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    self._assert_metrics(surface_distances, mask_gt, mask_pred,
                         expected_average_surface_distance=(1.5, 1.5),
                         expected_hausdorff_100=2.0,
                         expected_hausdorff_95=2.0,
                         expected_surface_overlap_at_1mm=(0.5, 0.5),
                         expected_surface_dice_at_1mm=0.5,
                         expected_volumetric_dice=0.0)

  def test_two_cubes_shifted_by_one_pixel(self):
    mask_gt = np.zeros((100, 100, 100), bool)
    mask_pred = np.zeros((100, 100, 100), bool)
    mask_gt[0:50, :, :] = 1
    mask_pred[0:51, :, :] = 1
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(2, 1, 1))
    self._assert_metrics(
        surface_distances, mask_gt, mask_pred,
        expected_average_surface_distance=(0.322, 0.339),
        expected_hausdorff_100=2.0,
        expected_hausdorff_95=2.0,
        expected_surface_overlap_at_1mm=(0.842, 0.830),
        expected_surface_dice_at_1mm=0.836,
        expected_volumetric_dice=0.990)

  def test_empty_prediction_mask(self):
    mask_gt = np.zeros((128, 128, 128), bool)
    mask_pred = np.zeros((128, 128, 128), bool)
    mask_gt[50, 60, 70] = 1
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    self._assert_metrics(
        surface_distances, mask_gt, mask_pred,
        expected_average_surface_distance=(np.inf, np.nan),
        expected_hausdorff_100=np.inf,
        expected_hausdorff_95=np.inf,
        expected_surface_overlap_at_1mm=(0.0, np.nan),
        expected_surface_dice_at_1mm=0.0,
        expected_volumetric_dice=0.0)

  def test_empty_ground_truth_mask(self):
    mask_gt = np.zeros((128, 128, 128), bool)
    mask_pred = np.zeros((128, 128, 128), bool)
    mask_pred[50, 60, 72] = 1
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    self._assert_metrics(
        surface_distances, mask_gt, mask_pred,
        expected_average_surface_distance=(np.nan, np.inf),
        expected_hausdorff_100=np.inf,
        expected_hausdorff_95=np.inf,
        expected_surface_overlap_at_1mm=(np.nan, 0.0),
        expected_surface_dice_at_1mm=0.0,
        expected_volumetric_dice=0.0)

  def test_both_empty_masks(self):
    mask_gt = np.zeros((128, 128, 128), bool)
    mask_pred = np.zeros((128, 128, 128), bool)
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    self._assert_metrics(
        surface_distances, mask_gt, mask_pred,
        expected_average_surface_distance=(np.nan, np.nan),
        expected_hausdorff_100=np.inf,
        expected_hausdorff_95=np.inf,
        expected_surface_overlap_at_1mm=(np.nan, np.nan),
        expected_surface_dice_at_1mm=np.nan,
        expected_volumetric_dice=np.nan)


if __name__ == '__main__':
  absltest.main()
