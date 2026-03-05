import cv2
import numpy as np
import improutils as iu


def aspect_ratio(contour):
	"""Calculate aspect ratio of bounding rectangle (width/height)."""
	_, _, w, h = cv2.boundingRect(contour)
	if h == 0:
		return 0.0
	return float(w) / h


def roundness(contour):
	"""Calculate roundness (4*pi*area/perimeter^2). Circle = 1.0."""
	area = cv2.contourArea(contour)
	perimeter = cv2.arcLength(contour, True)
	if perimeter == 0:
		return 0.0
	return (4 * np.pi * area) / (perimeter ** 2)


def _enhance_cracks(gray, clip_limit=2.5, tile_grid_size=(8, 8), gamma=1.35):
	"""Enhance local contrast and brightness so cracks are easier to detect."""
	gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

	clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
	clahe_img = clahe.apply(gray_norm)

	gamma = max(0.1, float(gamma))
	if abs(gamma - 1.0) < 1e-6:
		return clahe_img

	lut = np.array([
		((value / 255.0) ** (1.0 / gamma)) * 255 for value in range(256)
	], dtype=np.uint8)
	return cv2.LUT(clahe_img, lut)


def segment_cracks(
	inner_log_img,
	crust_mask,
	crack_threshold,
	edge_exclude_kernel,
	crust_exclude_kernel,
	min_crack_area,
	max_aspect_ratio,
	max_roundness,
	gauss_kernel_size,
):
	"""Detect cracks in the inner log using CLAHE + gamma + Scharr gradients."""
	gray = iu.to_gray(inner_log_img)
	gray_enhanced = _enhance_cracks(gray)

	scharr_x = cv2.Scharr(gray_enhanced, cv2.CV_64F, 1, 0)
	scharr_y = cv2.Scharr(gray_enhanced, cv2.CV_64F, 0, 1)
	gradient_magnitude = cv2.magnitude(scharr_x, scharr_y)

	gradient_norm = cv2.normalize(
		gradient_magnitude,
		None,
		0,
		255,
		cv2.NORM_MINMAX,
		dtype=cv2.CV_8U,
	)

	edge_mask = iu.segmentation_one_threshold(gradient_norm, 100)
	edge_contours_mask, _, _ = iu.find_contours(edge_mask, min_area=0, fill=True)

	edge_kernel = max(1, int(edge_exclude_kernel))
	edge_exclusion_mask = cv2.dilate(
		edge_contours_mask,
		np.ones((edge_kernel, edge_kernel), np.uint8),
		iterations=1,
	)

	crust_exclusion_mask = crust_mask
	if crust_exclude_kernel and crust_exclude_kernel > 1:
		crust_kernel = int(crust_exclude_kernel)
		crust_exclusion_mask = cv2.dilate(
			crust_exclusion_mask,
			np.ones((crust_kernel, crust_kernel), np.uint8),
			iterations=1,
		)

	exclusion_mask = cv2.bitwise_or(edge_exclusion_mask, crust_exclusion_mask)
	gradient_magnitude[exclusion_mask == 255] = 0

	kernel_size = max(1, int(gauss_kernel_size)) | 1
	gradient_magnitude = cv2.GaussianBlur(
		gradient_magnitude,
		(kernel_size, kernel_size),
		0,
	)

	gradient_norm = cv2.normalize(
		gradient_magnitude,
		None,
		0,
		255,
		cv2.NORM_MINMAX,
		dtype=cv2.CV_8U,
	)
	crack_candidates = iu.segmentation_one_threshold(gradient_norm, crack_threshold)

	_, _, contours = iu.find_contours(
		crack_candidates,
		min_area=min_crack_area,
		fill=False,
	)

	crack_mask = np.zeros_like(crack_candidates)
	selected_contours = [
		contour
		for contour in contours
		if aspect_ratio(contour) <= max_aspect_ratio and roundness(contour) <= max_roundness
	]

	if selected_contours:
		cv2.drawContours(crack_mask, selected_contours, -1, 255, thickness=cv2.FILLED)

	return crack_mask
