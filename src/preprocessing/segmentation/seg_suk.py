from .seg_common import segment_using_superpixels_and_kmeans


def segment_suk(img, k=16, attempts=10, seed=42):
   return segment_using_superpixels_and_kmeans(img, k, attempts, seed, [k-1 ])
