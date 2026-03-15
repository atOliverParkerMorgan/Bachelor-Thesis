from .seg_common import segment_using_superpixels_and_kmeans, has_outliers
import numpy as np
import improutils as iu

SUK_SUPERPIXEL_REGION_SIZE = 12
SUK_SUPERPIXEL_RULER = 1.0


def segment_suk(img, k=22, attempts=1, seed=42):
   if has_outliers(img):
      mask = segment_using_superpixels_and_kmeans(
         img,
         k,
         attempts,
         seed,
         [k - 1],
         region_size=SUK_SUPERPIXEL_REGION_SIZE,
         ruler=SUK_SUPERPIXEL_RULER,
      )
      final_mask, _, _ = iu.find_contours(mask, min_area=32)
      return final_mask
   
   return np.zeros(img.shape[:2], dtype=np.uint8)
