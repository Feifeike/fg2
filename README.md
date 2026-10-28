# FG2 on KITTI-360

This package adapts the released FG2 KITTI point matcher to the existing
Stage-1 KITTI-360 JSON manifests. It does not modify FG2 or the Stage-1 model.
The FG2 network and DINOv2 implementation are imported from the configured
`external.fg2_repository` at runtime.

## Data and geometry

Each manifest item must provide front/satellite image paths, intrinsics,
`camera_to_vehicle`, `vehicle_to_world`, satellite center in world XY, and
satellite meters per pixel. The loader computes the camera pose as
`vehicle_to_world @ camera_to_vehicle`, resizes intrinsics with the front image,
and derives FG2's satellite-to-ground BEV transform. Training applies random
satellite translation and rotation and updates that transform from the exact
image warp. GT is used for the bidirectional pose-supervised InfoNCE/VCE losses,
not to mask inference-time matching scores.

Validation applies a deterministic per-sample satellite translation and
rotation from the same ranges as training. This prevents the existing
vehicle-centered crops from making translation validation trivially close to
zero and keeps evaluation repeatable between checkpoints.

The source Stage-1 train and validation manifests share all seven training
drives. This config holds out drive `0009` from training and uses its 664
validation records; the test manifest is drive `0010`. The loader checks that
the resulting train/validation/test drive sets do not overlap.

The released KITTI model is used as-is, including its non-persistent fixed BEV
query buffer. The adapter re-seeds before model construction so checkpoints can
be resumed/evaluated with the same query initialization. The released forward
also calls `grd_ffn1` at iteration 4 although `grd_ffn4` is defined; this is
preserved for code-path fidelity and should be reported if comparing against a
corrected implementation.

The front image is resized to 364x1232 so the DINOv2 patch map is 26x88, as
assumed by the released FG2 KITTI cross-attention. The satellite crop is resized
to 504x504. Current KITTI-360 manifests use north-up 512x512 satellite crops at
0.2 m/px, so the matching grid spans 102.4 m to preserve the crop's physical
extent after resize.

## Runtime requirements

Use the FG2 environment requirements, including an MMCV build whose
`MultiScaleDeformableAttention` CUDA op matches the active PyTorch/CUDA build.
DINOv2 ViT-L/14 weights can be set in the config or downloaded by the upstream
extractor.

## Train

Place this directory at `models/stage1/fg2_kitti360` so the Python package path
is `stage1.fg2_kitti360`. Copy `config.example.yaml` to `config.yaml`, then set
the paths for the FG2 checkout and KITTI-360 manifests. The machine-specific
`config.yaml` is ignored by Git.

From the `models` directory:

```bash
cp stage1/fg2_kitti360/config.example.yaml stage1/fg2_kitti360/config.yaml
# Edit the FG2 and manifest paths in stage1/fg2_kitti360/config.yaml.
conda activate <fg2-environment>
python -m stage1.fg2_kitti360.train --config stage1/fg2_kitti360/config.yaml
```

Checkpoints are saved under `stage1/outputs_fg2_kitti360/checkpoints/` with both
epoch and step in each filename. Validation reports geometric point-match
accuracy at the configured meter thresholds and pose error after sampled
correspondences are aligned with weighted Procrustes. These are KITTI-360
adaptation metrics, not the published KITTI benchmark numbers.

Training writes TensorBoard events to
`stage1/outputs_fg2_kitti360/tensorboard/`. The logs include per-step total,
InfoNCE, and VCE losses and learning rate, plus per-epoch training summaries
and validation metrics. Install TensorBoard in the FG2 environment if needed:

```bash
pip install tensorboard
tensorboard --logdir stage1/outputs_fg2_kitti360/tensorboard
```

To evaluate a saved checkpoint:

```bash
python -m stage1.fg2_kitti360.train \
  --config stage1/fg2_kitti360/config.yaml \
  --eval_only \
  --eval_split test \
  --resume stage1/outputs_fg2_kitti360/checkpoints/fg2_kitti360_epoch_001_step_000100.pth
```

