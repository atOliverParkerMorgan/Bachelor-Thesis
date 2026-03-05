import improutils as iu


def segment_decay_in_knots(
    log,
    knots_mask,
    min_area,
    lower_threshold=70,
    upper_threshold=255,
):
    """Detect decay/rot inside knot regions."""
    log_knots = iu.apply_mask(iu.negative(log), knots_mask)
    log_knots_gray = iu.to_gray(log_knots)
    decay_mask = iu.segmentation_two_thresholds(
        log_knots_gray,
        lower=lower_threshold,
        higher=upper_threshold,
    )
    clean_mask, _, _ = iu.find_contours(
        decay_mask,
        min_area=min_area,
        fill=True,
        external=True,
    )
    return clean_mask
