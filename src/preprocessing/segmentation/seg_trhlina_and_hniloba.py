import cv2
import numpy as np
import improutils as iu
from .seg_common import kmeans_brightness_labels, mask_from_cluster_ids
from skimage.segmentation import morphological_chan_vese

def segment_trhlina(img, k=5):
    # 1. Grayscale & Enhancement (The "Black Hat" Trick)
    # A Black Hat transform specifically isolates dark, thin structures (cracks)
    # and ignores the large, gradual changes in the wood grain.
    gray = iu.to_gray(img)
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)
    
    # Add the blackhat result back to the inverted gray image to heavily emphasize cracks
    # This makes the cracks POP for the K-means algorithm
    enhanced = cv2.addWeighted(cv2.bitwise_not(gray), 0.5, blackhat, 0.5, 0)
    
    # We use Bilateral on the enhanced image to smooth noise but keep crack edges sharp
    filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=120, sigmaSpace=75)
    
    # 2. Run K-means (using 3 channels artificially so your kmeans function still works)
    filtered_3c = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    
    # Note: Because we used Blackhat and inverted, cracks are now the BRIGHTEST things.
    # So we grab the highest tiers (k-1, k-2) instead of 0.
    k_labels, k_centers = kmeans_brightness_labels(filtered_3c, k=k)
    initial_mask = mask_from_cluster_ids(k_labels, [k-1, k-2])
    
    # 3. Clean up the mask: Use CLOSE instead of OPEN!
    # CLOSE will bridge small gaps in broken thin cracks without deleting the crack itself.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_init_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 4. Prepare for scikit-image
    # skimage Chan-Vese behaves best when the input image is scaled to [0.0, 1.0]
    gray_float = gray.astype(np.float32) / 255.0
    init_level_set = (clean_init_mask > 0).astype(np.float32)
    
    # 5. Run Morphological Snake
    # Set smoothing=0 so the snake doesn't shrink and crush the thin cracks!
    # Increased num_iter slightly to let it travel down the narrow paths.
    snake_mask_float = morphological_chan_vese(gray_float, 
                                               num_iter=25, 
                                               init_level_set=init_level_set,
                                               smoothing=0)
    
    # 6. Convert back to OpenCV format
    final_crack_mask = (snake_mask_float * 255).astype(np.uint8)
    
    return final_crack_mask