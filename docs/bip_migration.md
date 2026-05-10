# BIP Solver Integration

This branch keeps the NeOF data, visualization, pose saving, and triangulation
evaluation pipeline intact, and swaps only the camera-placement solver when
`--solver bip` is selected.

## Run NeOF

```bash
python main.py --config configs/main_brother.json --solver neof --vismode save
python evaluate_triangulation.py --config configs/main_brother.json
```

## Run BIP On The Same Input

```bash
python main.py --config configs/main_brother.json --solver bip --vismode save
```

The BIP run automatically writes to a solver-suffixed output path, for example
`resultModel/random/brother_bip_plane/`, so it does not overwrite the NeOF run
under `resultModel/random/brother_plane/`.

Evaluate the BIP result by pointing the existing evaluator at the BIP path:

```bash
python evaluate_triangulation.py --config configs/main_brother.json --solver bip
```

`evaluate_triangulation.py` applies the same solver and camera-constraint suffix
rules as `main.py`, so this resolves to `random/brother_bip_plane/` for the
current plane-constrained Brother config.

## Key BIP Parameters

- `solver`: `neof` or `bip`.
- `bip_coverage_mode`: defaults to `pair_angle`, matching the BIP repo's
  triangulation-pair objective. `kcoverage` is kept as an optional ablation that
  matches this repo's K-coverage term more directly.
- `bip_candidate_position_step`: placement-grid spacing in meters.
- `bip_target_count`: number of weighted support points used as look-at targets.
- `bip_max_base_candidates`: cap on pose candidates before slot assignment.
- `bip_pair_candidate_limit`: pair-angle candidate reduction before building
  pair variables; default is 20 for the Brother free-space config. Increase it
  for a denser but slower BIP search.
- `bip_time_limit`: SciPy HiGHS MILP time limit in seconds.
- `bip_min_triangulation_angle_deg` and `bip_max_triangulation_angle_deg`:
  active only in `pair_angle` mode.

The adapter is slot-aware: each configured camera keeps its own intrinsics, and
the BIP chooses one candidate pose per camera slot. This preserves the same
input/output contract as NeOF even for heterogeneous rigs.

If HiGHS reaches `bip_time_limit` after finding a feasible incumbent, the run
continues with that incumbent and records the non-optimal status in
`pose/bip_solution.json`.
