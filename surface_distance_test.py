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

from absl.testing import absltest
import numpy as np
import surface_distance


class SurfaceDistanceTest(absltest.TestCase):

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
            surface_distances, 1))
    for i in range(2):
      self._assert_almost_equal(
          expected_surface_overlap_at_1mm[i],
          actual_surface_overlap_at_1mm[i],
          places=places)

    self._assert_almost_equal(
        expected_surface_dice_at_1mm,
        surface_distance.compute_surface_dice_at_tolerance(
            surface_distances, 1),
        places=places)

    self._assert_almost_equal(
        expected_volumetric_dice,
        surface_distance.compute_dice_coefficient(mask_gt, mask_pred),
        places=places)

  def testSinglePixels2mmAway(self):
    mask_gt = np.zeros((128, 128, 128), np.uint8)
    mask_pred = np.zeros((128, 128, 128), np.uint8)
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

  def testTwoCubes(self):
    mask_gt = np.zeros((100, 100, 100), np.uint8)
    mask_pred = np.zeros((100, 100, 100), np.uint8)
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

  def testEmptyPredictionMask(self):
    mask_gt = np.zeros((128, 128, 128), np.uint8)
    mask_pred = np.zeros((128, 128, 128), np.uint8)
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

  def testEmptyGroundTruthMask(self):
    mask_gt = np.zeros((128, 128, 128), np.uint8)
    mask_pred = np.zeros((128, 128, 128), np.uint8)
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

  def testEmptyBothMasks(self):
    mask_gt = np.zeros((128, 128, 128), np.uint8)
    mask_pred = np.zeros((128, 128, 128), np.uint8)
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


if __name__ == "__main__":
  absltest.main()
