from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _rgb_tensor(path: Path, size_hw: tuple[int, int]) -> torch.Tensor:
    height, width = size_hw
    with Image.open(path) as image:
        image = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _rotation_2d(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def _augment_satellite(
    image: torch.Tensor,
    *,
    angle_deg: float,
    shift_uv_px: tuple[float, float],
) -> torch.Tensor:
    """Warp the satellite crop and keep the transform convention explicit.

    The output image samples the original image at
    ``center + R(-angle) @ (output - center - shift)``.  The matching
    satellite-to-ground pose is updated from this same inverse mapping.
    """

    _, height, width = image.shape
    device = image.device
    dtype = image.dtype
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    center_x = (width - 1) * 0.5
    center_y = (height - 1) * 0.5
    dx = xx - center_x - float(shift_uv_px[0])
    dy = yy - center_y - float(shift_uv_px[1])
    angle = math.radians(-float(angle_deg))
    source_x = center_x + math.cos(angle) * dx - math.sin(angle) * dy
    source_y = center_y + math.sin(angle) * dx + math.cos(angle) * dy
    grid = torch.stack(
        (
            source_x * (2.0 / max(width - 1, 1)) - 1.0,
            source_y * (2.0 / max(height - 1, 1)) - 1.0,
        ),
        dim=-1,
    ).unsqueeze(0)
    return F.grid_sample(
        image.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).squeeze(0)


class FG2Kitti360Dataset(Dataset):
    """Read KITTI-360 samples and derive FG2's relative BEV pose labels.

    The returned pose maps satellite point-plane coordinates to ground point-plane
    coordinates.  Ground coordinates follow FG2's KITTI convention: x is
    negative in front of the camera, y points right.  Satellite coordinates are
    (image row, image column), in meters, on a north-up crop.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        front_size_hw: tuple[int, int] = (364, 1232),
        satellite_size_hw: tuple[int, int] = (504, 504),
        training: bool = False,
        fixed_eval_augmentation: bool = False,
        augmentation_seed: int = 0,
        shift_range_m: float = 20.0,
        rotation_range_deg: float = 10.0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in manifest: {self.manifest_path}")
        self.records: list[dict[str, Any]] = payload
        self.front_size_hw = tuple(int(value) for value in front_size_hw)
        self.satellite_size_hw = tuple(int(value) for value in satellite_size_hw)
        self.training = bool(training)
        self.fixed_eval_augmentation = bool(fixed_eval_augmentation)
        self.augmentation_seed = int(augmentation_seed)
        self.shift_range_m = float(shift_range_m)
        self.rotation_range_deg = float(rotation_range_deg)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def record_drive_id(record: dict[str, Any]) -> str:
        image_path = str(record.get("front_rgb_path", ""))
        for component in Path(image_path).parts:
            if component.startswith("2013_") and "_drive_" in component and component.endswith("_sync"):
                return component
        return ""

    def retain_drives(
        self,
        *,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> None:
        include = include or set()
        exclude = exclude or set()
        if include:
            self.records = [record for record in self.records if self.record_drive_id(record) in include]
        if exclude:
            self.records = [record for record in self.records if self.record_drive_id(record) not in exclude]

    def drive_ids(self) -> set[str]:
        return {self.record_drive_id(record) for record in self.records}

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.records[index]
        sample_id = str(item.get("sample_id") or index)
        for key in ("front_rgb_path", "satellite_rgb_path", "intrinsics", "camera_to_vehicle", "vehicle_to_world"):
            if key not in item:
                raise KeyError(f"{sample_id}: manifest entry is missing {key!r}")

        front_path = Path(item["front_rgb_path"])
        sat_path = Path(item["satellite_rgb_path"])
        with Image.open(front_path) as image:
            original_front_width, original_front_height = image.size
        with Image.open(sat_path) as image:
            original_sat_width, original_sat_height = image.size

        front = _rgb_tensor(front_path, self.front_size_hw)
        satellite = _rgb_tensor(sat_path, self.satellite_size_hw)

        front_height, front_width = self.front_size_hw
        scale_x = front_width / original_front_width
        scale_y = front_height / original_front_height
        intrinsics = np.asarray(item["intrinsics"], dtype=np.float64).copy()
        intrinsics[0, :] *= scale_x
        intrinsics[1, :] *= scale_y

        camera_to_vehicle = np.asarray(item["camera_to_vehicle"], dtype=np.float64)
        vehicle_to_world = np.asarray(item["vehicle_to_world"], dtype=np.float64)
        camera_to_world = vehicle_to_world @ camera_to_vehicle
        camera_xy = camera_to_world[:2, 3]
        forward = camera_to_world[:2, 2]
        forward_norm = np.linalg.norm(forward)
        if not np.isfinite(forward_norm) or forward_norm < 1e-8:
            raise ValueError(f"{sample_id}: camera forward axis has no finite world-XY projection")
        forward = forward / forward_norm
        # World x/y are east/north for these north-up KITTI-360 crops.
        right = np.asarray([forward[1], -forward[0]], dtype=np.float64)
        ground_from_world = np.stack((-forward, right), axis=0)

        sat_center = np.asarray(item.get("satellite_center_world_xy", vehicle_to_world[:2, 3]), dtype=np.float64)
        source_mpp = float(item.get("satellite_meters_per_pixel", 0.2))
        sat_height, sat_width = self.satellite_size_hw
        mpp_u = source_mpp * original_sat_width / sat_width
        mpp_v = source_mpp * original_sat_height / sat_height
        if abs(mpp_u - mpp_v) > max(mpp_u, mpp_v) * 1e-3:
            raise ValueError(
                f"{sample_id}: resizing would make satellite pixels anisotropic "
                f"(mpp_u={mpp_u:.6f}, mpp_v={mpp_v:.6f}); use square, square-scale crops"
            )
        mpp = 0.5 * (mpp_u + mpp_v)

        if self.training or self.fixed_eval_augmentation:
            rng = np.random if self.training else np.random.default_rng(self.augmentation_seed + index)
            angle_deg = float(rng.uniform(-self.rotation_range_deg, self.rotation_range_deg))
            shift_world_px = rng.uniform(-self.shift_range_m, self.shift_range_m, size=2) / mpp
        else:
            angle_deg = 0.0
            shift_world_px = np.zeros(2, dtype=np.float64)
        shift_uv = (float(shift_world_px[0]), float(shift_world_px[1]))
        satellite = _augment_satellite(satellite, angle_deg=angle_deg, shift_uv_px=shift_uv)

        # Pixel -> world for the north-up crop: u is east, v is south.
        pixel_to_world = np.asarray([[mpp, 0.0], [0.0, -mpp]], dtype=np.float64)
        inverse_rotation = _rotation_2d(math.radians(-angle_deg))
        output_center_to_source = inverse_rotation @ (-np.asarray(shift_uv, dtype=np.float64))
        augmented_sat_center_world = sat_center + pixel_to_world @ output_center_to_source

        # FG2 point coordinates are (row, col) in meters; convert them to
        # output-image pixel offsets, then through the exact augmentation warp.
        metric_to_pixel_uv = np.asarray([[0.0, 1.0 / mpp], [1.0 / mpp, 0.0]], dtype=np.float64)
        satellite_to_world = pixel_to_world @ inverse_rotation @ metric_to_pixel_uv
        rotation_gt = ground_from_world @ satellite_to_world
        translation_gt = ground_from_world @ (augmented_sat_center_world - camera_xy)

        return {
            "sample_id": sample_id,
            "front": front,
            "satellite": satellite,
            "camera_k": torch.tensor(intrinsics, dtype=torch.float32),
            "rotation_gt": torch.tensor(rotation_gt, dtype=torch.float32),
            "translation_gt": torch.tensor(translation_gt[None], dtype=torch.float32),
            "satellite_meters_per_pixel": torch.tensor(mpp, dtype=torch.float32),
        }
