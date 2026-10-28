# FG2 on KITTI-360
Copy `config.example.yaml` to `config.yaml` and update the FG2 and manifest
paths before training. To resume from a saved checkpoint, run this from the
`models` directory:

```bash
cd /path/to/models
python -m stage1.fg2_kitti360.train \
  --config stage1/fg2_kitti360/config.yaml \
  --resume stage1/outputs_fg2_kitti360/checkpoints/<checkpoint>.pth
```

