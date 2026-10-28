from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT.parent
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from stage1.fg2_kitti360.dataset import FG2Kitti360Dataset
from stage1.fg2_kitti360.matching import (
    make_index_grids,
    make_metric_grids,
    pose_errors,
    sampled_bidirectional_infonce,
    virtual_correspondence_error,
    weighted_procrustes_2d,
)
from stage1.fg2_kitti360.runtime import load_dinov2_weights, load_fg2_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate FG2's point matcher on KITTI-360.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path, default=None, help="Resume an FG2-KITTI-360 checkpoint.")
    parser.add_argument("--init_checkpoint", type=Path, default=None, help="Initialize the FG2 matcher from a checkpoint.")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_split", choices=("val", "test"), default="val")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    return parser.parse_args()


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _make_tensorboard_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError(
            "TensorBoard logging requires the tensorboard package. Install it in the active environment with: "
            "pip install tensorboard"
        ) from error
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir), flush_secs=30)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_model_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must be a state-dict or contain a model/state_dict entry.")
    state = {key.removeprefix("module."): value for key, value in payload.items() if torch.is_tensor(value)}
    if not state:
        raise ValueError("No tensor weights were found in the matcher checkpoint.")
    return state


def _load_checkpoint(path: Path, model, optimizer=None, *, strict: bool = True) -> tuple[int, int, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = _extract_model_state(payload)
    model_keys = set(model.state_dict())
    matched_keys = model_keys.intersection(state)
    if not matched_keys:
        raise RuntimeError(f"Checkpoint has no keys compatible with the FG2 matcher: {path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if not strict and (missing or unexpected):
        print(f"initialized {len(matched_keys)} matcher tensors; missing={len(missing)}, unexpected={len(unexpected)}")
    if optimizer is not None and isinstance(payload, dict) and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    epoch = int(payload.get("epoch", 0)) if isinstance(payload, dict) else 0
    step = int(payload.get("step", 0)) if isinstance(payload, dict) else 0
    best_metric = float(payload.get("best_top1_accuracy_2_5m", -math.inf)) if isinstance(payload, dict) else -math.inf
    return epoch, step, best_metric


def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_id": [sample["sample_id"] for sample in samples],
        **{
            key: torch.stack([sample[key] for sample in samples])
            for key in (
                "front",
                "satellite",
                "camera_k",
                "rotation_gt",
                "translation_gt",
                "satellite_meters_per_pixel",
            )
        },
    }


def _make_loader(dataset, batch_size: int, workers: int, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=_collate,
        generator=generator,
        worker_init_fn=seed_worker,
    )


def _sample_matches(
    score: torch.Tensor,
    num_samples: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Sample point correspondences from FG2's dustbin-normalized score matrix."""

    weights = torch.nan_to_num(score.detach().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
    positive_count = int((weights > 0).sum().item())
    if positive_count < 3 or float(weights.sum().item()) <= 0:
        return None
    count = min(int(num_samples), positive_count)
    selected = torch.multinomial(weights, count, replacement=False, generator=generator)
    num_ground = score.shape[1]
    return torch.div(selected, num_ground, rounding_mode="floor"), selected.remainder(num_ground)


def _top1_metrics(
    score: torch.Tensor,
    raw_logits: torch.Tensor,
    rotation_gt: torch.Tensor,
    translation_gt: torch.Tensor,
    ground_metric: torch.Tensor,
    satellite_metric: torch.Tensor,
    grid_size_m: float,
    thresholds_m: list[float],
) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {"errors": [], "coverage": []}
    for threshold in thresholds_m:
        metrics[f"correct_{threshold:g}m"] = []

    for batch_idx in range(score.shape[0]):
        valid_ground = torch.isfinite(raw_logits[batch_idx, 0])
        if not bool(valid_ground.any()):
            continue
        gt_satellite = (ground_metric - translation_gt[batch_idx, 0]) @ rotation_gt[batch_idx]
        gt_in_map = (gt_satellite.abs() <= grid_size_m / 2).all(dim=-1)
        evaluable = valid_ground & gt_in_map
        valid_count = int(valid_ground.sum().item())
        if valid_count == 0 or not bool(evaluable.any()):
            continue
        top1 = score[batch_idx].argmax(dim=0)
        predicted_satellite = satellite_metric[top1]
        errors = torch.linalg.vector_norm(predicted_satellite[evaluable] - gt_satellite[evaluable], dim=-1)
        metrics["errors"].extend(errors.detach().cpu().tolist())
        metrics["coverage"].append(float(evaluable.sum().item() / valid_count))
        for threshold in thresholds_m:
            metrics[f"correct_{threshold:g}m"].extend((errors <= threshold).float().detach().cpu().tolist())
    return metrics


@torch.no_grad()
def evaluate(
    *,
    model,
    feature_extractor,
    loader,
    device: torch.device,
    sat_metric: torch.Tensor,
    grd_metric: torch.Tensor,
    config: dict,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    model.eval()
    feature_extractor.eval()
    correspondence_thresholds = [float(value) for value in config["evaluation"]["correspondence_thresholds_m"]]
    all_errors: list[float] = []
    all_correct = {threshold: [] for threshold in correspondence_thresholds}
    all_coverage: list[float] = []
    translation_errors: list[float] = []
    yaw_errors: list[float] = []
    sample_count = 0
    grid_size_m = float(config["model"]["grid_size_h_m"])
    num_pose_samples = int(config["model"]["num_samples_matches"])

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        front = batch["front"].to(device, non_blocking=True)
        satellite = batch["satellite"].to(device, non_blocking=True)
        camera_k = batch["camera_k"].to(device, non_blocking=True)
        rotation_gt = batch["rotation_gt"].to(device, non_blocking=True)
        translation_gt = batch["translation_gt"].to(device, non_blocking=True)
        with torch.no_grad():
            front_feature = feature_extractor(front)
            satellite_feature = feature_extractor(satellite)
        score, raw_logits = model(front_feature, satellite_feature, camera_k)

        top1 = _top1_metrics(
            score,
            raw_logits,
            rotation_gt,
            translation_gt,
            grd_metric,
            sat_metric,
            grid_size_m,
            correspondence_thresholds,
        )
        all_errors.extend(top1["errors"])
        all_coverage.extend(top1["coverage"])
        for threshold in correspondence_thresholds:
            all_correct[threshold].extend(top1[f"correct_{threshold:g}m"])

        for b in range(score.shape[0]):
            eval_seed = int(config["evaluation"].get("seed", 0)) + batch_idx * score.shape[0] + b
            match_generator = torch.Generator(device=device)
            match_generator.manual_seed(eval_seed)
            sampled = _sample_matches(score[b], num_pose_samples, generator=match_generator)
            if sampled is None:
                continue
            sat_ids, grd_ids = sampled
            weights = score[b].reshape(-1)[sat_ids * score.shape[2] + grd_ids]
            rotation_pred, translation_pred = weighted_procrustes_2d(
                sat_metric[sat_ids], grd_metric[grd_ids], weights
            )
            if rotation_pred is None or translation_pred is None:
                continue
            trans_error, yaw_error = pose_errors(
                rotation_pred,
                translation_pred,
                rotation_gt[b],
                translation_gt[b],
            )
            translation_errors.append(float(trans_error.cpu()))
            yaw_errors.append(float(yaw_error.cpu()))
            sample_count += 1

    result: dict[str, float | int] = {
        "num_evaluable_correspondences": len(all_errors),
        "num_pose_samples": sample_count,
        "top1_error_mean_m": float(np.mean(all_errors)) if all_errors else math.nan,
        "top1_error_median_m": float(np.median(all_errors)) if all_errors else math.nan,
        "evaluable_coverage_mean": float(np.mean(all_coverage)) if all_coverage else math.nan,
        "pose_translation_mean_m": float(np.mean(translation_errors)) if translation_errors else math.nan,
        "pose_translation_median_m": float(np.median(translation_errors)) if translation_errors else math.nan,
        "pose_yaw_mean_deg": float(np.mean(yaw_errors)) if yaw_errors else math.nan,
        "pose_yaw_median_deg": float(np.median(yaw_errors)) if yaw_errors else math.nan,
    }
    for threshold in correspondence_thresholds:
        values = all_correct[threshold]
        result[f"top1_accuracy_{threshold:g}m"] = float(np.mean(values)) if values else math.nan
    return result


def _save_epoch_checkpoint(
    output_dir: Path,
    model,
    optimizer,
    *,
    epoch: int,
    step: int,
    config: dict,
    best_metric: float,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"fg2_kitti360_epoch_{epoch:03d}_step_{step:07d}.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_top1_accuracy_2_5m": best_metric,
            "config": config,
        },
        path,
    )
    return path


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if tuple(config["data"]["front_size_hw"]) != (364, 1232):
        raise ValueError("Released FG2 KITTI cross-attention requires front_size_hw=[364, 1232] (26x88 DINO patches).")
    if tuple(config["data"]["satellite_size_hw"]) != (504, 504):
        raise ValueError("This FG2 KITTI-360 adapter expects satellite_size_hw=[504, 504].")
    if config["model"]["ground_bev_resolution"] != config["model"]["satellite_bev_resolution"]:
        raise ValueError("The current FG2 KITTI-360 loss path requires equal ground/satellite BEV resolutions.")
    seed = int(config["training"].get("seed", 0))
    _seed_everything(seed)

    components = load_fg2_components(config["external"]["fg2_repository"])
    # model_kitti.py has a non-persistent fixed query buffer. Re-seed directly
    # before construction so its initialization is stable across resume/eval.
    _seed_everything(seed)
    device = torch.device(args.device)

    model_cfg = config["model"]
    model = components["CVM"](
        device,
        grd_bev_res=int(model_cfg["ground_bev_resolution"]),
        grd_height_res=int(model_cfg["ground_height_resolution"]),
        sat_bev_res=int(model_cfg["satellite_bev_resolution"]),
        grid_size_h=float(model_cfg["grid_size_h_m"]),
        grid_size_v=float(model_cfg["grid_size_v_m"]),
        temperature=float(model_cfg["temperature"]),
        embed_dim=1024,
        desc_dim=128,
    ).to(device)
    dinov2_weights = load_dinov2_weights(config["external"].get("dinov2_weights"))
    feature_extractor = components["DinoExtractor"](dinov2_weights=dinov2_weights).to(device)
    feature_extractor.eval()
    feature_extractor.requires_grad_(False)

    data_cfg = config["data"]
    train_dataset = FG2Kitti360Dataset(
        _resolve_path(data_cfg["train_manifest"], MODELS_ROOT),
        front_size_hw=tuple(data_cfg["front_size_hw"]),
        satellite_size_hw=tuple(data_cfg["satellite_size_hw"]),
        training=True,
        shift_range_m=float(data_cfg["shift_range_m"]),
        rotation_range_deg=float(data_cfg["rotation_range_deg"]),
    )
    val_dataset = FG2Kitti360Dataset(
        _resolve_path(data_cfg["val_manifest"], MODELS_ROOT),
        front_size_hw=tuple(data_cfg["front_size_hw"]),
        satellite_size_hw=tuple(data_cfg["satellite_size_hw"]),
        training=False,
        fixed_eval_augmentation=bool(config["evaluation"].get("synthetic_perturbation", True)),
        augmentation_seed=int(config["evaluation"].get("seed", seed + 1)),
        shift_range_m=float(data_cfg["shift_range_m"]),
        rotation_range_deg=float(data_cfg["rotation_range_deg"]),
    )
    validation_holdout_drives = set(data_cfg.get("validation_holdout_drives", []))
    if validation_holdout_drives:
        train_dataset.retain_drives(exclude=validation_holdout_drives)
        val_dataset.retain_drives(include=validation_holdout_drives)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(
            "Train/validation split is empty after drive filtering. Check "
            "data.validation_holdout_drives and the manifest sequence IDs."
        )
    train_val_overlap = train_dataset.drive_ids() & val_dataset.drive_ids()
    if train_val_overlap:
        raise ValueError(f"Drive-level train/validation leakage detected: {sorted(train_val_overlap)}")

    test_loader = None
    test_dataset = None
    if data_cfg.get("test_manifest"):
        test_dataset = FG2Kitti360Dataset(
            _resolve_path(data_cfg["test_manifest"], MODELS_ROOT),
            front_size_hw=tuple(data_cfg["front_size_hw"]),
            satellite_size_hw=tuple(data_cfg["satellite_size_hw"]),
            training=False,
            fixed_eval_augmentation=bool(config["evaluation"].get("synthetic_perturbation", True)),
            augmentation_seed=int(config["evaluation"].get("test_seed", seed + 2)),
            shift_range_m=float(data_cfg["shift_range_m"]),
            rotation_range_deg=float(data_cfg["rotation_range_deg"]),
        )
        train_test_overlap = train_dataset.drive_ids() & test_dataset.drive_ids()
        val_test_overlap = val_dataset.drive_ids() & test_dataset.drive_ids()
        if train_test_overlap or val_test_overlap:
            raise ValueError(
                "Drive-level split leakage detected: "
                f"train-test={sorted(train_test_overlap)}, "
                f"val-test={sorted(val_test_overlap)}"
            )
        print(
            "FG2-KITTI-360 split: "
            f"train={len(train_dataset)} {sorted(train_dataset.drive_ids())}; "
            f"val={len(val_dataset)} {sorted(val_dataset.drive_ids())}; "
            f"test={len(test_dataset)} {sorted(test_dataset.drive_ids())}"
        )
        test_loader = _make_loader(
            test_dataset,
            int(config["evaluation"]["batch_size"]),
            int(config["evaluation"]["num_workers"]),
            shuffle=False,
            seed=int(config["evaluation"].get("test_seed", seed + 2)),
        )
    train_loader = _make_loader(
        train_dataset,
        int(config["training"]["batch_size"]),
        int(config["training"]["num_workers"]),
        shuffle=True,
        seed=seed,
    )
    val_loader = _make_loader(
        val_dataset,
        int(config["evaluation"]["batch_size"]),
        int(config["evaluation"]["num_workers"]),
        shuffle=False,
        seed=seed,
    )

    resolution = int(model_cfg["ground_bev_resolution"])
    grid_size_m = float(model_cfg["grid_size_h_m"])
    sat_metric, grd_metric = make_metric_grids(grid_size_m, resolution, device)
    sat_indices, grd_indices = make_index_grids(resolution, device)
    output_dir = _resolve_path(config["training"]["output_dir"], PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=(0.9, 0.999),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    start_epoch = 0
    global_step = 0
    best_top1 = -math.inf
    if args.resume:
        start_epoch, global_step, best_top1 = _load_checkpoint(
            args.resume.expanduser(), model, optimizer, strict=True
        )
    elif args.init_checkpoint:
        _load_checkpoint(args.init_checkpoint.expanduser(), model, strict=False)

    if args.eval_only:
        if not args.resume and not args.init_checkpoint:
            raise ValueError("--eval_only requires --resume or --init_checkpoint.")
        eval_loader = val_loader if args.eval_split == "val" else test_loader
        if eval_loader is None:
            raise ValueError("--eval_split test requires data.test_manifest in the config.")
        metrics = evaluate(
            model=model,
            feature_extractor=feature_extractor,
            loader=eval_loader,
            device=device,
            sat_metric=sat_metric,
            grd_metric=grd_metric,
            config=config,
            max_batches=args.max_val_batches,
        )
        print(json.dumps(metrics, indent=2, allow_nan=True))
        (output_dir / f"metrics_{args.eval_split}_latest.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8"
        )
        return

    tensorboard_dir = _resolve_path(
        config["training"].get("tensorboard_dir", "tensorboard"),
        output_dir,
    )
    writer = _make_tensorboard_writer(tensorboard_dir)
    writer.add_text("run/config", yaml.safe_dump(config, sort_keys=False), global_step=global_step)

    metrics_path = output_dir / "metrics.jsonl"
    num_samples = int(model_cfg["num_samples_matches"])
    vce_cfg = config["loss"]
    max_steps = args.max_steps

    for epoch in range(start_epoch + 1, int(config["training"]["epochs"]) + 1):
        model.train()
        feature_extractor.eval()
        epoch_losses: list[float] = []
        epoch_infonce: list[float] = []
        epoch_vce: list[float] = []

        for batch in train_loader:
            front = batch["front"].to(device, non_blocking=True)
            satellite = batch["satellite"].to(device, non_blocking=True)
            camera_k = batch["camera_k"].to(device, non_blocking=True)
            rotation_gt = batch["rotation_gt"].to(device, non_blocking=True)
            translation_gt = batch["translation_gt"].to(device, non_blocking=True)

            with torch.no_grad():
                front_feature = feature_extractor(front)
                satellite_feature = feature_extractor(satellite)
            score, raw_logits = model(front_feature, satellite_feature, camera_k)

            losses: list[torch.Tensor] = []
            infonce_values: list[float] = []
            vce_values: list[float] = []
            for batch_idx in range(score.shape[0]):
                ground_valid = torch.isfinite(raw_logits[batch_idx, 0])
                if not bool(ground_valid.any()):
                    continue
                sampled = _sample_matches(score[batch_idx], num_samples)
                if sampled is None:
                    continue
                sat_ids, grd_ids = sampled
                info_nce = sampled_bidirectional_infonce(
                    raw_logits[batch_idx : batch_idx + 1],
                    rotation_gt[batch_idx : batch_idx + 1],
                    translation_gt[batch_idx : batch_idx + 1],
                    sat_indices[sat_ids].unsqueeze(0),
                    grd_indices[grd_ids].unsqueeze(0),
                    ground_valid.unsqueeze(0),
                    grid_size_m=grid_size_m,
                    resolution=resolution,
                )
                weights = score[batch_idx].reshape(-1)[sat_ids * score.shape[2] + grd_ids]
                rotation_pred, translation_pred = weighted_procrustes_2d(
                    sat_metric[sat_ids], grd_metric[grd_ids], weights
                )
                if rotation_pred is None or translation_pred is None:
                    total_loss = float(vce_cfg["beta"]) * info_nce
                else:
                    vce = virtual_correspondence_error(
                        rotation_gt[batch_idx],
                        translation_gt[batch_idx],
                        rotation_pred,
                        translation_pred,
                        grid_size_m=float(vce_cfg["grid_size_m"]),
                        resolution=int(vce_cfg["grid_resolution"]),
                    )
                    total_loss = float(vce_cfg["beta"]) * info_nce + vce
                    vce_values.append(float(vce.detach().cpu()))
                losses.append(total_loss)
                infonce_values.append(float(info_nce.detach().cpu()))

            if not losses:
                continue
            loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite FG2 loss at epoch {epoch}, step {global_step}.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_norm = float(config["training"].get("grad_clip_norm", 0.0))
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            global_step += 1
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_infonce.extend(infonce_values)
            epoch_vce.extend(vce_values)
            writer.add_scalar("train/step_loss", epoch_losses[-1], global_step)
            if infonce_values:
                writer.add_scalar("train/step_infonce", float(np.mean(infonce_values)), global_step)
            if vce_values:
                writer.add_scalar("train/step_vce", float(np.mean(vce_values)), global_step)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)

            if global_step % int(config["training"].get("log_interval", 20)) == 0:
                print(
                    f"epoch={epoch} step={global_step} loss={epoch_losses[-1]:.5f} "
                    f"infonce={infonce_values and np.mean(infonce_values):.5f}"
                )
            if max_steps is not None and global_step >= max_steps:
                break

        val_metrics = evaluate(
            model=model,
            feature_extractor=feature_extractor,
            loader=val_loader,
            device=device,
            sat_metric=sat_metric,
            grd_metric=grd_metric,
            config=config,
            max_batches=args.max_val_batches,
        )
        record: dict[str, Any] = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else math.nan,
            "train_infonce": float(np.mean(epoch_infonce)) if epoch_infonce else math.nan,
            "train_vce": float(np.mean(epoch_vce)) if epoch_vce else math.nan,
            **val_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=True) + "\n")

        for tag, key in (
            ("epoch/train_loss", "train_loss"),
            ("epoch/train_infonce", "train_infonce"),
            ("epoch/train_vce", "train_vce"),
        ):
            value = float(record[key])
            if math.isfinite(value):
                writer.add_scalar(tag, value, epoch)
        for key, value in val_metrics.items():
            scalar = float(value)
            if math.isfinite(scalar):
                writer.add_scalar(f"validation/{key}", scalar, epoch)
        writer.flush()

        metric_key = "top1_accuracy_2.5m"
        current_metric = float(val_metrics.get(metric_key, math.nan))
        is_best = math.isfinite(current_metric) and current_metric > best_top1
        checkpoint_best_metric = current_metric if is_best else best_top1
        checkpoint = _save_epoch_checkpoint(
            output_dir,
            model,
            optimizer,
            epoch=epoch,
            step=global_step,
            config=config,
            best_metric=checkpoint_best_metric,
        )
        if is_best:
            best_top1 = current_metric
            best_path = checkpoint.with_name(
                f"fg2_kitti360_best_top1_epoch_{epoch:03d}_step_{global_step:07d}.pth"
            )
            shutil.copy2(checkpoint, best_path)
            with (output_dir / "best_checkpoint.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {"checkpoint": str(best_path), "metric": metric_key, "value": best_top1},
                    handle,
                    indent=2,
                )
        print(json.dumps(record, indent=2, allow_nan=True))
        if max_steps is not None and global_step >= max_steps:
            break
    writer.close()


if __name__ == "__main__":
    main()
