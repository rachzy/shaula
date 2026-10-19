import numpy as np

# 4 - Compute duration-matched noise and CDPP

# Baselines tried by ``flank_durations="auto"``, in box widths per side; None
# is the series-median baseline. Synthetic injections (experiments/
# mes_accuracy) put white-noise MES at 0.78/0.88/0.91/0.94/1.0 of ideal for
# these, while 1 box width recovers spotted stars from 0.14 to 0.65 of ideal.
ADAPTIVE_FLANK_DURATIONS = (1.0, 2.0, 3.0, 5.0, None)


def _empty_statistics():
    empty_float = np.array([], dtype=float)
    empty_bool = np.array([], dtype=bool)
    return {
        "time": empty_float,
        "depth": empty_float,
        "raw_depth": empty_float,
        "uncertainty": empty_float,
        "N": empty_float,
        "D": empty_float,
        "SES": empty_float,
        "coverage": empty_float,
        "valid": empty_bool,
        "noise_eligible": empty_bool,
        "cadence_days": np.nan,
        "duration_days": np.nan,
    }


def _robust_location_scale(values):
    """Return the median and Gaussian-equivalent MAD, without RMS fallback."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan, np.nan
    location = float(np.median(values))
    mad = float(np.median(np.abs(values - location)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nan
    return location, scale


def _sample_anchor_indices(time, valid, segment_id, step_days):
    """Select valid statistic samples separated by at least ``step_days``."""
    selected = []
    for segment in np.unique(segment_id):
        candidates = np.flatnonzero(valid & (segment_id == segment))
        if candidates.size == 0:
            continue
        last_time = -np.inf
        for index in candidates:
            if time[index] - last_time >= step_days * (1.0 - 1e-12):
                selected.append(int(index))
                last_time = time[index]
    return np.asarray(selected, dtype=int)


def _independent_span_count(time, segment_id, duration_days):
    """Approximate the number of independent duration-sized noise samples."""
    count = 0.0
    for segment in np.unique(segment_id):
        values = time[segment_id == segment]
        if values.size:
            count += max(1.0, (values[-1] - values[0]) / duration_days)
    return count


def _local_anchor_noise(
    time,
    raw_depth,
    segment_id,
    anchor_indices,
    duration_days,
    noise_window_durations,
    min_local_anchors,
    min_independent_samples,
):
    """Estimate robust local depth location and scale at sparse anchors."""
    if anchor_indices.size == 0:
        return np.array([]), np.array([])

    anchor_time = time[anchor_indices]
    anchor_depth = raw_depth[anchor_indices]
    anchor_segment = segment_id[anchor_indices]
    global_location, global_scale = _robust_location_scale(anchor_depth)
    independent_count = _independent_span_count(
        anchor_time, anchor_segment, duration_days
    )
    if independent_count < min_independent_samples:
        global_location = np.nan
        global_scale = np.nan

    locations = np.full(anchor_indices.size, np.nan, dtype=float)
    scales = np.full(anchor_indices.size, np.nan, dtype=float)
    half_window = 0.5 * float(noise_window_durations) * duration_days

    for segment in np.unique(anchor_segment):
        positions = np.flatnonzero(anchor_segment == segment)
        segment_times = anchor_time[positions]
        segment_depths = anchor_depth[positions]
        left = np.searchsorted(segment_times, segment_times - half_window, side="left")
        right = np.searchsorted(
            segment_times, segment_times + half_window, side="right"
        )
        # Every window as one NaN-padded row, so the medians run vectorised.
        # Same result as calling _robust_location_scale per window.
        width = int(np.max(right - left))
        offsets = np.arange(width)
        index = left[:, None] + offsets[None, :]
        inside = index < right[:, None]
        windows = np.where(
            inside, segment_depths[np.minimum(index, positions.size - 1)], np.nan
        )
        windows[~np.isfinite(windows)] = np.nan
        n_finite = np.sum(np.isfinite(windows), axis=1)
        enough = (right - left >= min_local_anchors) & (n_finite >= 2)
        location = np.full(positions.size, np.nan)
        scale = np.full(positions.size, np.nan)
        if np.any(enough):
            rows = windows[enough]
            location[enough] = np.nanmedian(rows, axis=1)
            scale[enough] = 1.4826 * np.nanmedian(
                np.abs(rows - location[enough][:, None]), axis=1
            )
        fallback = ~enough | ~np.isfinite(scale) | (scale <= 0)
        location[fallback] = global_location
        scale[fallback] = global_scale
        locations[positions] = location
        scales[positions] = scale

    return locations, scales


def _flank_relative_depth(
    time, deficit, left, right, box_depth, segment_id, usable, flank_days, min_points
):
    """Box depth relative to the mean of the flux on either side of the box.

    Each flank spans ``flank_days`` next to the box, stays inside the box's
    segment and uses only ``usable`` cadences (known transits withheld). The
    two flank means are averaged with equal weight so a linear trend cancels.
    Every box, in transit or not, is treated identically, so the noise model
    sees exactly the filter the signal sees.
    """
    n = time.size
    seg_start = np.zeros(n, dtype=int)
    seg_end = np.full(n, n, dtype=int)
    boundaries = np.flatnonzero(np.diff(segment_id)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))
    for start, end in zip(starts, ends):
        seg_start[start:end] = start
        seg_end[start:end] = end

    left_lo = np.maximum(
        np.searchsorted(time, time[np.minimum(left, n - 1)] - flank_days, side="left"),
        seg_start,
    )
    left_hi = np.maximum(left, left_lo)
    right_hi = np.minimum(
        np.searchsorted(
            time, time[np.maximum(right - 1, 0)] + flank_days, side="right"
        ),
        seg_end,
    )
    right_lo = np.minimum(right, right_hi)

    weights = usable.astype(float)
    prefix_count = np.concatenate(([0.0], np.cumsum(weights)))
    prefix_sum = np.concatenate(([0.0], np.cumsum(deficit * weights)))
    n_left = prefix_count[left_hi] - prefix_count[left_lo]
    n_right = prefix_count[right_hi] - prefix_count[right_lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_left = (prefix_sum[left_hi] - prefix_sum[left_lo]) / n_left
        mean_right = (prefix_sum[right_hi] - prefix_sum[right_lo]) / n_right
    has_left = n_left >= min_points
    has_right = n_right >= min_points
    baseline = np.where(
        has_left & has_right,
        0.5 * (mean_left + mean_right),
        np.where(has_left, mean_left, mean_right),
    )
    ok = has_left | has_right
    depth = np.where(ok, box_depth - baseline, np.nan)
    return depth, ok


def duration_matched_statistics(
    time,
    flux,
    duration_hours,
    cadence_hours=None,
    exclude_mask=None,
    gap_mask=None,
    min_coverage=0.8,
    gap_cadences=5.0,
    noise_window_durations=30.0,
    min_local_anchors=20,
    min_independent_samples=5,
    flank_durations="auto",
    _scale_only=False,
):
    """Build a local, duration-matched transit detection statistic.

    A positive ``depth`` represents a flux deficit. The box-template amplitude
    is evaluated at every observed cadence. Its local median and MAD are
    estimated from half-duration-spaced anchors, excluding known transit
    windows from the noise sample only. The returned ``N`` and ``D`` components
    can be combined coherently across predicted transit events.

    ``exclude_mask`` marks the candidate's own transit windows: those cadences
    stay foldable but are kept out of the noise sample. ``gap_mask`` marks
    cadences to treat as unobserved (Kepler TPS "gapping" of already-detected
    transits) - any box touching one is invalidated, so it contributes to
    neither the noise model nor the fold. Passing ``gap_mask`` is strongly
    preferred over deleting those cadences from ``time``/``flux``: deletion
    leaves sub-``gap_cadences`` holes that fragment ``segment_id``, truncating
    the local noise windows and biasing the MAD low. Note ``gap_cadences``
    (segment-splitting threshold) is unrelated to ``gap_mask``.

    ``flank_durations`` sets the box's baseline. ``None`` measures depth
    against the series median, so any stellar variability slower than the box
    stays in the noise. A number measures it against the mean flux in that
    many box widths on either side (``_flank_relative_depth``), a local
    high-pass that removes spot modulation at the cost of the flanks' own
    noise. ``"auto"`` ranks every option in ``ADAPTIVE_FLANK_DURATIONS`` by the
    pooled robust scale of its noise anchors and builds the lowest. The choice looks only at
    noise-eligible boxes, never at the candidate's transits, and every option
    preserves the full transit depth, so the lowest noise is the highest
    expected MES without biasing MES on pure noise.

    This is a time-domain approximation to Kepler TPS. It intentionally does
    not reproduce TPS's adaptive wavelet whitening; ``"auto"`` stands in for
    its adaptivity.
    """
    if isinstance(flank_durations, str):
        if flank_durations != "auto":
            raise ValueError("flank_durations must be None, a number or 'auto'")
        # Rank the options on the pooled scale of their noise anchors, which
        # is cheap, then build the full local noise model only for the winner.
        chosen, chosen_scale = None, np.inf
        for option in ADAPTIVE_FLANK_DURATIONS:
            scale = duration_matched_statistics(
                time,
                flux,
                duration_hours,
                cadence_hours=cadence_hours,
                exclude_mask=exclude_mask,
                gap_mask=gap_mask,
                min_coverage=min_coverage,
                gap_cadences=gap_cadences,
                noise_window_durations=noise_window_durations,
                min_local_anchors=min_local_anchors,
                min_independent_samples=min_independent_samples,
                flank_durations=option,
                _scale_only=True,
            )
            if isinstance(scale, dict):  # an early return: nothing to rank
                continue
            if np.isfinite(scale) and scale < chosen_scale:
                chosen, chosen_scale = option, scale
        flank_durations = chosen

    result = _empty_statistics()
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if time.ndim != 1 or flux.ndim != 1:
        raise ValueError("time and flux must be one-dimensional arrays")
    if time.shape != flux.shape:
        raise ValueError("time and flux must have the same shape")
    if not np.isfinite(duration_hours) or duration_hours <= 0:
        return result
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in the interval (0, 1]")
    if not np.isfinite(gap_cadences) or gap_cadences <= 1:
        raise ValueError("gap_cadences must be greater than one")

    if exclude_mask is None:
        exclude_mask = np.zeros(time.shape, dtype=bool)
    else:
        exclude_mask = np.asarray(exclude_mask, dtype=bool)
        if exclude_mask.shape != time.shape:
            raise ValueError("exclude_mask, time, and flux must have the same shape")

    if gap_mask is None:
        gap_mask = np.zeros(time.shape, dtype=bool)
    else:
        gap_mask = np.asarray(gap_mask, dtype=bool)
        if gap_mask.shape != time.shape:
            raise ValueError("gap_mask, time, and flux must have the same shape")

    finite = np.isfinite(time) & np.isfinite(flux)
    if np.sum(finite) < 2:
        return result
    order = np.argsort(time[finite], kind="stable")
    time = time[finite][order]
    flux = flux[finite][order]
    exclude_mask = exclude_mask[finite][order]
    gap_mask = gap_mask[finite][order]

    positive_steps = np.diff(time)
    positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0)]
    measured_cadence = (
        float(np.median(positive_steps)) if positive_steps.size else np.nan
    )
    supplied_cadence = (
        float(cadence_hours) / 24.0
        if cadence_hours is not None
        and np.isfinite(cadence_hours)
        and cadence_hours > 0
        else np.nan
    )
    cadence_days = (
        supplied_cadence if np.isfinite(supplied_cadence) else measured_cadence
    )
    duration_days = float(duration_hours) / 24.0
    if not np.isfinite(cadence_days) or cadence_days <= 0:
        return result

    result["cadence_days"] = cadence_days
    result["duration_days"] = duration_days

    segment_id = np.zeros(time.size, dtype=int)
    if time.size > 1:
        segment_id[1:] = np.cumsum(np.diff(time) > gap_cadences * cadence_days)

    baseline_values = flux[~exclude_mask & ~gap_mask]
    if baseline_values.size < 2:
        return result
    baseline = float(np.median(baseline_values))
    if not np.isfinite(baseline) or baseline == 0:
        return result
    deficit = 1.0 - flux / baseline

    half_duration = 0.5 * duration_days
    left = np.searchsorted(time, time - half_duration, side="left")
    right = np.searchsorted(time, time + half_duration, side="right")
    counts = right - left
    expected_points = max(1.0, duration_days / cadence_days)
    minimum_points = max(2, int(np.ceil(min_coverage * expected_points)))

    prefix_deficit = np.concatenate(([0.0], np.cumsum(deficit)))
    prefix_excluded = np.concatenate(
        ([0], np.cumsum(exclude_mask.astype(np.int64)))
    )
    prefix_gapped = np.concatenate(([0], np.cumsum(gap_mask.astype(np.int64))))
    raw_depth = (prefix_deficit[right] - prefix_deficit[left]) / np.maximum(counts, 1)
    overlaps_excluded = (prefix_excluded[right] - prefix_excluded[left]) > 0
    # Any overlap with a gapped cadence invalidates the box outright. Gapped
    # windows hold another signal's transits, so a partially overlapping box
    # would carry that depth into this candidate's fold.
    overlaps_gap = (prefix_gapped[right] - prefix_gapped[left]) > 0

    right_index = np.maximum(right - 1, 0)
    same_segment = segment_id[left] == segment_id[right_index]
    box_valid = (counts >= minimum_points) & same_segment & ~overlaps_gap

    if flank_durations is not None:
        raw_depth, flank_ok = _flank_relative_depth(
            time,
            deficit,
            left,
            right,
            raw_depth,
            segment_id,
            ~exclude_mask & ~gap_mask,
            float(flank_durations) * duration_days,
            max(1, int(np.ceil(0.25 * expected_points))),
        )
        box_valid &= flank_ok
    raw_depth[~box_valid] = np.nan
    coverage = counts.astype(float) / expected_points
    coverage[~box_valid] = np.nan

    noise_candidate = box_valid & ~overlaps_excluded & np.isfinite(raw_depth)
    anchor_indices = _sample_anchor_indices(
        time,
        noise_candidate,
        segment_id,
        step_days=max(cadence_days, half_duration),
    )
    if _scale_only:
        return _robust_location_scale(raw_depth[anchor_indices])[1]
    anchor_locations, anchor_scales = _local_anchor_noise(
        time,
        raw_depth,
        segment_id,
        anchor_indices,
        duration_days,
        noise_window_durations=float(noise_window_durations),
        min_local_anchors=int(min_local_anchors),
        min_independent_samples=int(min_independent_samples),
    )

    local_location = np.full(time.size, np.nan, dtype=float)
    local_scale = np.full(time.size, np.nan, dtype=float)
    if anchor_indices.size:
        anchor_segment = segment_id[anchor_indices]
        finite_anchor_noise = np.isfinite(anchor_locations) & np.isfinite(anchor_scales)
        global_location, global_scale = _robust_location_scale(
            raw_depth[anchor_indices]
        )
        independent_count = _independent_span_count(
            time[anchor_indices], anchor_segment, duration_days
        )
        if independent_count < min_independent_samples:
            global_location = np.nan
            global_scale = np.nan

        for segment in np.unique(segment_id):
            output_positions = np.flatnonzero(segment_id == segment)
            source_positions = np.flatnonzero(
                (anchor_segment == segment) & finite_anchor_noise
            )
            if source_positions.size:
                source_indices = anchor_indices[source_positions]
                local_location[output_positions] = np.interp(
                    time[output_positions],
                    time[source_indices],
                    anchor_locations[source_positions],
                )
                local_scale[output_positions] = np.interp(
                    time[output_positions],
                    time[source_indices],
                    anchor_scales[source_positions],
                )
            elif np.isfinite(global_scale):
                local_location[output_positions] = global_location
                local_scale[output_positions] = global_scale

    # The local MAD is calibrated on normally covered boxes. Only incomplete
    # boxes need extra uncertainty; inclusive endpoints can make regular boxes
    # contain one point more than the duration/cadence ratio and must not make
    # them look artificially more precise than the local noise model.
    effective_points = np.minimum(np.maximum(counts, 1), expected_points)
    coverage_factor = np.sqrt(expected_points / effective_points)
    uncertainty = local_scale * coverage_factor
    depth = raw_depth - local_location
    valid = (
        box_valid
        & np.isfinite(depth)
        & np.isfinite(uncertainty)
        & (uncertainty > 0)
    )
    depth[~valid] = np.nan
    uncertainty[~valid] = np.nan

    denominator = np.full(time.size, np.nan, dtype=float)
    numerator = np.full(time.size, np.nan, dtype=float)
    ses = np.full(time.size, np.nan, dtype=float)
    denominator[valid] = 1.0 / np.square(uncertainty[valid])
    numerator[valid] = depth[valid] * denominator[valid]
    ses[valid] = numerator[valid] / np.sqrt(denominator[valid])

    result.update(
        {
            "time": time,
            "depth": depth,
            "raw_depth": raw_depth,
            "uncertainty": uncertainty,
            "N": numerator,
            "D": denominator,
            "SES": ses,
            "coverage": coverage,
            "valid": valid,
            "noise_eligible": valid & ~overlaps_excluded,
        }
    )
    return result


def calculate_cdpp(
    flux,
    cadence_hours,
    durations=(3.0, 6.0, 12.0),
    time=None,
    exclude_mask=None,
    gap_mask=None,
    min_coverage=0.8,
    min_windows=5,
):
    """Return median local duration-matched uncertainty in ppm.

    CDPP is summarized from the same time-dependent uncertainty used by the
    SES/MES estimator. Known transit windows are excluded from the noise model
    and from the summary, but the statistic remains a time-domain approximation
    rather than a reproduction of Kepler TPS wavelet CDPP.

    ``gap_mask`` marks already-detected transits to treat as unobserved; see
    ``duration_matched_statistics`` for why gapping beats deleting cadences.
    """
    flux = np.asarray(flux, dtype=float)
    if flux.ndim != 1:
        raise ValueError("flux must be a one-dimensional array")
    if time is None:
        if not np.isfinite(cadence_hours) or cadence_hours <= 0:
            return {f"cdpp_{int(dur)}h": np.nan for dur in durations}
        time = np.arange(flux.size, dtype=float) * float(cadence_hours) / 24.0

    results = {}
    for duration in durations:
        duration = float(duration)
        series = duration_matched_statistics(
            time,
            flux,
            duration_hours=duration,
            cadence_hours=cadence_hours,
            exclude_mask=exclude_mask,
            gap_mask=gap_mask,
            min_coverage=min_coverage,
            min_independent_samples=min_windows,
        )
        eligible = series["noise_eligible"]
        uncertainty = series["uncertainty"][eligible]
        value = float(np.median(uncertainty) * 1e6) if uncertainty.size else np.nan
        results[f"cdpp_{int(duration)}h"] = value
    return results
