import cv2
import numpy as np
from .seg_common import segment_using_superpixels_and_kmeans

def segment_crust(img, k=4, attempts=10, seed=42, k_label=1):
   return segment_using_superpixels_and_kmeans(img, k, attempts, seed, k_label)
