# NeOF Camera Placement Experiments

This repository is based on the RAL 2024 paper
[NeOF Guided Hybrid Optimization of Camera Placement](https://ieeexplore.ieee.org/document/10638696).
This branch keeps the original NeOF solver and adds a comparable BIP baseline, mocap-marker target preprocessing,
shared visualization/evaluation outputs, timestamped result folders, and helper tools for Blender pose export.

The preferred workflow is the mocap data collection setup in `configs/main_mocap.json`.

## Main Workflow

The default config uses preprocessed mocap marker support points:

```text
mocap_data/{mocap_sequence}_targets.csv
```

The optimization output directory is named automatically with:

```text
{mocap_sequence}_{solver}_{camera_constraint_shape}_{YYYYMMDDHHMMSS}
```

For example:

```text
resultModel/random/mocap/p08_bird_correct_neof_box_walls_20260520120314/
```

The default NeOF workflow is:

1. Load the same mocap support points for all solvers.
2. Generate a non-gradient reset-search set.
3. Select the best reset candidate by coverage/triangulation-angle proxy.
4. Run gradient refinement from that candidate.
5. Save the same pose, visualization, and evaluation-compatible files for NeOF and BIP.

Epoch checkpoint evaluation is disabled by default on this branch. The normal evaluation output is the concise
`Initial` / `Optimized` / `Comparison` report from `evaluate_triangulation.py`.

## Environment

The code is currently used with the conda environment name `neof_cam`.

```bash
conda env create -f environment.yml
conda activate neof_cam
```

If your machine does not have CUDA 11.8, edit `environment.yml` and replace the CUDA PyTorch entries with the
CPU-only PyTorch install recommended by the official PyTorch selector.

For an already-created environment, install/update the non-PyTorch packages with:

```bash
pip install -r requirements.txt
```

## Mocap Data

Raw mocap JSON exports can be large and are ignored by Git:

```text
mocap_data/*.json
```

Compact preprocessed targets and summaries can be tracked:

```text
mocap_data/*_targets.csv
mocap_data/*_targets_summary.json
```

To preprocess a raw mocap sequence:

```bash
python preprocess_mocap_markers.py \
  --mocap_sequence p08_bird_correct \
  --frame_sampling farthest \
  --max_frames 50 \
  --voxel_size 0.02
```

This writes:

```text
mocap_data/p08_bird_correct_targets.csv
mocap_data/p08_bird_correct_targets_summary.json
```

## Run Optimization

NeOF:

```bash
python main.py \
  --config configs/main_mocap.json \
  --mocap_sequence p08_bird_correct \
  --solver neof
```

BIP, using the same input targets and output conventions:

```bash
python main.py \
  --config configs/main_mocap.json \
  --mocap_sequence p08_bird_correct \
  --solver bip
```

The BIP config defaults to the OR-Tools CP-SAT backend:

```jsonc
"bip_solver_backend": "ortools_cpsat"
```

## Evaluate

Evaluate the latest NeOF run:

```bash
python evaluate_triangulation.py \
  --config configs/main_mocap.json \
  --mocap_sequence p08_bird_correct \
  --solver neof \
  --run_timestamp latest \
  --pixel_noise_std 10 \
  --trials 5 \
  --seed 0
```

Evaluate the latest BIP run:

```bash
python evaluate_triangulation.py \
  --config configs/main_mocap.json \
  --mocap_sequence p08_bird_correct \
  --solver bip \
  --run_timestamp latest \
  --pixel_noise_std 10 \
  --trials 5 \
  --seed 0
```

For mocap targets, the config uses `eval_visibility_mode: "fov_only"` so the reconstruction rate measures camera
field-of-view coverage of marker/support points without marker self-occlusion.

## Visualize Saved Results

Open a saved visualization scene:

```bash
python view_saved_scene.py \
  --scene resultModel/random/mocap/p08_bird_correct_neof_box_walls_YYYYMMDDHHMMSS/visualization/optimized_scene.zip
```

Export a saved reset-search pose as the same Open3D scene format:

```bash
python export_pose_visualization.py \
  --config configs/main_mocap.json \
  --mocap_sequence p08_bird_correct \
  --solver neof \
  --run_timestamp latest \
  --pose_stage reset_search_selected
```

## Print Camera Poses

Print initial and optimized poses in Blender-friendly command fragments:

```bash
python print_optimized_camera_poses.py \
  --config configs/main_mocap.json \
  --mocap_sequence p08_bird_correct \
  --solver neof \
  --run_timestamp latest \
  --pose_name both \
  --rotation_convention blender
```

Use `--rotation_convention neof` to print Euler angles directly from the saved NeOF world-to-camera rotation
instead of converting OpenCV camera axes to Blender camera-object axes.

## Useful Config Knobs

Camera placement constraint:

```jsonc
"camera_constraint_shape": "box_walls"
```

Other supported shapes include:

```text
box, cylinder, dome, box_surface, box_walls, plane, cylinder_surface, dome_surface
```

Solver selection:

```jsonc
"solver": "neof"
```

or:

```jsonc
"solver": "bip"
```

All-camera coverage:

```jsonc
"require_all_cameras_coverage": true
```

This makes the effective coverage target equal to the configured camera count.

## Original Citation

```bibtex
@article{cao2024neural,
  title={Neural Observation Field Guided Hybrid Optimization of Camera Placement},
  author={Cao, Yihan and Zhang, Jiazhao and Yu, Zhinan and Xu, Kai},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```
