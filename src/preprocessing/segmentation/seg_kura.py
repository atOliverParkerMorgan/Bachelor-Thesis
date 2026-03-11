import cv2
import numpy as np
from .seg_common import segment_using_superpixels_and_kmeans, to_binary


KURA_MIN_COMPONENT_AREA = 80
KURA_MIN_BOUNDARY_OVERLAP_RATIO = 0.01
KURA_MIN_BOUNDARY_OVERLAP_PIXELS = 40

def segment_crust(img, k=10, attempts=10, seed=42):
   return segment_using_superpixels_and_kmeans(img, k, attempts, seed, [1,2,3])


def refine_kura_outer_crust(raw_kura_mask, log_mask, crust_band, outer_ring, trhlina_and_hniloba_mask=None):
   """Keep bark components attached to the outer boundary and remove inner crack spill."""
   kura_candidate = cv2.bitwise_and(to_binary(raw_kura_mask), to_binary(log_mask))
   labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(kura_candidate, connectivity=8)

   outer_ring_dilated = cv2.dilate(
      to_binary(outer_ring),
      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
      iterations=1,
   )

   kept = np.zeros_like(kura_candidate)
   for label in range(1, labels_count):
      area = int(stats[label, cv2.CC_STAT_AREA])
      if area < KURA_MIN_COMPONENT_AREA:
         continue

      component = np.where(labels == label, 255, 0).astype(np.uint8)
      boundary_overlap = int(cv2.countNonZero(cv2.bitwise_and(component, outer_ring_dilated)))
      overlap_ratio = boundary_overlap / float(area)

      if boundary_overlap >= KURA_MIN_BOUNDARY_OVERLAP_PIXELS or overlap_ratio >= KURA_MIN_BOUNDARY_OVERLAP_RATIO:
         kept = cv2.bitwise_or(kept, component)

   kept = cv2.bitwise_and(kept, to_binary(crust_band))

   if trhlina_and_hniloba_mask is not None:
      # Suppress crack detections in inner bark while preserving the outer bark edge.
      outer_guard = cv2.dilate(
         to_binary(outer_ring),
         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
         iterations=1,
      )
      inner_crust_zone = cv2.bitwise_and(to_binary(crust_band), cv2.bitwise_not(outer_guard))
      crack_suppression = cv2.bitwise_or(trhlina_and_hniloba_mask, inner_crust_zone)
      kept = cv2.bitwise_and(kept, cv2.bitwise_not(crack_suppression))

   close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
   kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, close_kernel)
   return cv2.bitwise_and(kept, to_binary(crust_band))
