from .seg_common import segment_using_superpixels_and_kmeans, has_outliers
import numpy as np


SUK_SUPERPIXEL_REGION_SIZE = 12
SUK_SUPERPIXEL_RULER = 1.0


def segment_suk(img, k=20, attempts=1, seed=42):
   if has_outliers(img):
      return segment_using_superpixels_and_kmeans(
         img,
         k,
         attempts,
         seed,
         [k - 1],
         region_size=SUK_SUPERPIXEL_REGION_SIZE,
         ruler=SUK_SUPERPIXEL_RULER,
      )
   
   return np.zeros(img.shape[:2], dtype=np.uint8)
