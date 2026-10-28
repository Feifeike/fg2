from __future__ import annotations

import math

import torch


def make_metric_grids(
    grid_size_m: float,
    resolution: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FG2 satellite and front-half point-plane coordinates.

    Both grids use (row, column) coordinates in meters.  Ground rows cover the
    front half of the camera-centered BEV, matching FG2's KITTI convention.
    """

    satellite_rows = torch.linspace(-grid_size_m / 2, grid_size_m / 2, resolution, device=device)
    ground_rows = torch.linspace(
        -grid_size_m / 2,
        0.0,
        math.floor(resolution / 2) + 1,
        device=device,
    )
    columns = torch.linspace(-grid_size_m / 2, grid_size_m / 2, resolution, device=device)
    sat_row, sat_col = torch.meshgrid(satellite_rows, columns, indexing="ij")
    grd_row, grd_col = torch.meshgrid(ground_rows, columns, indexing="ij")
    return (
        torch.stack((sat_row, sat_col), dim=-1).reshape(-1, 2),
        torch.stack((grd_row, grd_col), dim=-1).reshape(-1, 2),
    )


def make_index_grids(resolution: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return homogeneous integer-grid coordinates used by FG2's InfoNCE loss."""

    center = (resolution - 1) / 2
    satellite_axis = torch.arange(resolution, device=device, dtype=torch.float32) - center
    ground_axis = torch.arange(math.floor(resolution / 2) + 1, device=device, dtype=torch.float32) - center
    columns = torch.arange(resolution, device=device, dtype=torch.float32) - center
    sat_row, sat_col = torch.meshgrid(satellite_axis, satellite_axis, indexing="ij")
    grd_row, grd_col = torch.meshgrid(ground_axis, columns, indexing="ij")
    sat = torch.stack((sat_row, sat_col, torch.ones_like(sat_row)), dim=-1).reshape(-1, 3)
    grd = torch.stack((grd_row, grd_col, torch.ones_like(grd_row)), dim=-1).reshape(-1, 3)
    return sat, grd


def _deduplicate_first(keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    first = torch.full((unique.numel(),), keys.numel(), device=keys.device, dtype=torch.long)
    first.scatter_reduce_(0, inverse, torch.arange(keys.numel(), device=keys.device), reduce="amin", include_self=True)
    return unique, values[first]


def sampled_bidirectional_infonce(
    logits: torch.Tensor,
    rotation_gt: torch.Tensor,
    translation_gt: torch.Tensor,
    sampled_sat_indices: torch.Tensor,
    sampled_ground_indices: torch.Tensor,
    ground_valid: torch.Tensor,
    *,
    grid_size_m: float,
    resolution: int,
) -> torch.Tensor:
    """FG2-style pose-supervised InfoNCE on the sampled correspondences.

    ``logits`` is the temperature-scaled score matrix ``[B, N_sat, N_ground]``.
    GT validity only limits supervised positives and denominator columns; it is
    not fed into the model's reverse normalization or matching selection.
    """

    batch_size, num_sat, num_ground = logits.shape
    device = logits.device
    center = (resolution - 1) / 2
    losses: list[torch.Tensor] = []
    for batch_idx in range(batch_size):
        valid_ground = ground_valid[batch_idx]
        if not bool(valid_ground.any()):
            continue

        rotation = rotation_gt[batch_idx]
        translation = translation_gt[batch_idx, 0]
        translation_index = translation / grid_size_m * resolution
        transform = torch.eye(3, dtype=logits.dtype, device=device)
        transform[:2, :2] = rotation
        transform[:2, 2] = translation_index

        sat_h = sampled_sat_indices[batch_idx]
        grd_h = sampled_ground_indices[batch_idx]

        # Satellite -> ground positives, deduplicated by satellite point.
        gt_ground_xy = (transform @ sat_h.transpose(0, 1)).transpose(0, 1)[:, :2].round()
        gt_ground_row = gt_ground_xy[:, 0] + center
        gt_ground_col = gt_ground_xy[:, 1] + center
        ground_in_bounds = (
            (gt_ground_row >= 0)
            & (gt_ground_row < math.floor(resolution / 2) + 1)
            & (gt_ground_col >= 0)
            & (gt_ground_col < resolution)
        )
        ground_flat = gt_ground_row.long() * resolution + gt_ground_col.long()
        ground_in_bounds &= valid_ground[ground_flat.clamp(0, num_ground - 1)]
        sat_flat = ((sat_h[:, 0] + center).long() * resolution + (sat_h[:, 1] + center).long())
        keep = ground_in_bounds & (sat_flat >= 0) & (sat_flat < num_sat)
        if bool(keep.any()):
            sat_keys, positive_ground = _deduplicate_first(sat_flat[keep], ground_flat[keep])
            log_denominator = torch.logsumexp(logits[batch_idx, sat_keys][:, valid_ground], dim=-1)
            log_positive = logits[batch_idx, sat_keys, positive_ground]
            losses.append((log_denominator - log_positive).mean())

        # Ground -> satellite positives, deduplicated by ground point.
        inverse_rotation = rotation.transpose(0, 1)
        gt_sat_xy = (grd_h[:, :2] - translation_index) @ inverse_rotation.transpose(0, 1)
        gt_sat_xy = gt_sat_xy.round()
        sat_row = gt_sat_xy[:, 0] + center
        sat_col = gt_sat_xy[:, 1] + center
        sat_in_bounds = (
            (sat_row >= 0)
            & (sat_row < resolution)
            & (sat_col >= 0)
            & (sat_col < resolution)
        )
        sat_flat_gt = sat_row.long() * resolution + sat_col.long()
        ground_flat_sampled = ((grd_h[:, 0] + center).long() * resolution + (grd_h[:, 1] + center).long())
        ground_query_valid = valid_ground[ground_flat_sampled.clamp(0, num_ground - 1)]
        keep = sat_in_bounds & ground_query_valid & (ground_flat_sampled >= 0) & (ground_flat_sampled < num_ground)
        if bool(keep.any()):
            ground_keys, positive_sat = _deduplicate_first(ground_flat_sampled[keep], sat_flat_gt[keep])
            selected_ground_logits = logits[batch_idx].index_select(1, ground_keys)
            log_denominator = torch.logsumexp(selected_ground_logits, dim=0)
            log_positive = logits[batch_idx, positive_sat, ground_keys]
            losses.append((log_denominator - log_positive).mean())

    if not losses:
        finite_logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
        return finite_logits.sum() * 0.0
    return torch.stack(losses).mean()


def weighted_procrustes_2d(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Estimate a differentiable 2-D rigid transform from weighted matches."""

    if source.shape[0] < 3:
        return None, None
    weights = weights.clamp_min(0)
    weight_sum = weights.sum()
    if not bool(torch.isfinite(weight_sum)) or float(weight_sum.detach()) <= 1e-12:
        return None, None
    normalized = weights / (weight_sum + 1e-12)
    source_mean = (normalized[:, None] * source).sum(dim=0, keepdim=True)
    target_mean = (normalized[:, None] * target).sum(dim=0, keepdim=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.transpose(0, 1) @ (weights[:, None] * target_centered)
    try:
        u, singular_values, vh = torch.linalg.svd(covariance)
    except RuntimeError:
        return None, None
    if float(singular_values[-1].detach()) <= 1e-8 or not bool(torch.isfinite(singular_values).all()):
        return None, None
    correction = torch.eye(2, dtype=source.dtype, device=source.device)
    correction[-1, -1] = torch.sign(torch.linalg.det(u @ vh))
    rotation = vh.transpose(-1, -2) @ correction @ u.transpose(-1, -2)
    translation = target_mean - source_mean @ rotation.transpose(-1, -2)
    return rotation, translation


def virtual_correspondence_error(
    rotation_gt: torch.Tensor,
    translation_gt: torch.Tensor,
    rotation_pred: torch.Tensor,
    translation_pred: torch.Tensor,
    grid_size_m: float = 5.0,
    resolution: int = 10,
) -> torch.Tensor:
    """FG2's virtual-correspondence transform loss in metric coordinates."""

    axis = torch.linspace(-grid_size_m / 2, grid_size_m / 2, resolution, device=rotation_gt.device)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((x, y), dim=-1).reshape(-1, 2)
    gt = points @ rotation_gt.transpose(-1, -2) + translation_gt.reshape(1, 2)
    pred = points @ rotation_pred.transpose(-1, -2) + translation_pred.reshape(1, 2)
    return torch.linalg.vector_norm(gt - pred, dim=-1).mean()


def pose_errors(
    rotation_pred: torch.Tensor,
    translation_pred: torch.Tensor,
    rotation_gt: torch.Tensor,
    translation_gt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return camera-origin translation error (m) and wrapped yaw error (deg)."""

    pred_camera_sat = -rotation_pred.transpose(-1, -2) @ translation_pred.reshape(2, 1)
    gt_camera_sat = -rotation_gt.transpose(-1, -2) @ translation_gt.reshape(2, 1)
    translation_error = torch.linalg.vector_norm(pred_camera_sat - gt_camera_sat)
    relative = rotation_pred @ rotation_gt.transpose(-1, -2)
    yaw_error = torch.atan2(relative[1, 0], relative[0, 0]).abs() * (180.0 / math.pi)
    yaw_error = torch.minimum(yaw_error, 360.0 - yaw_error)
    return translation_error, yaw_error

