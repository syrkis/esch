"""Validate point timelines shared by mesh, line and Voronoi layers."""

import numpy as np

from .values import values_array


def point_timelines(e, positions, values):
    positions = values_array(positions)
    if values.ndim not in (2, 3) or values.shape[0] != len(e.gs):
        raise ValueError("values must have shape (panels, points[, time])")
    if positions.ndim not in (3, 4) or positions.shape[:2] != values.shape[:2] or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (panels, points[, time], 2)")
    if values.ndim == 2:
        values = values[..., None]
    if positions.ndim == 3:
        positions = positions[..., None, :]
    frames = max(values.shape[-1], positions.shape[-2])
    if values.shape[-1] not in (1, frames) or positions.shape[-2] not in (1, frames):
        raise ValueError("positions and values must have matching time lengths (or one frame)")
    return (
        np.broadcast_to(positions, (*values.shape[:2], frames, 2)),
        np.broadcast_to(values, (*values.shape[:2], frames)),
    )


def trajectories(e, positions):
    """Positions alone, as (panel, point, frame, 2), for layers without values."""
    positions = values_array(positions)
    if positions.ndim not in (3, 4) or len(positions) != len(e.gs):
        raise ValueError("positions must have shape (panels, points[, time], 2)")
    return point_timelines(e, positions, np.ones(positions.shape[:2]))[0]
