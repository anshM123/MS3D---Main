# MS3D: observable-preserving correction of hidden error in reduced-order models

Code and recovered experimental record for the MS3D (Mishra–Senthilkumar 3D)
project.

A reduced-order model accumulates error, and most of it collects in coordinates
the model does not report. This code reconstructs that hidden error from the
model's own inconsistency with the governing equations — using no observations
and no reference trajectory — and corrects it while leaving every reported
coordinate bit-for-bit unchanged.

## Headline results

| | |
|---|---|
| Causal intervention, 3D Navier–Stokes | 36/36 improved, median gain 1.651, wrong-sign control 0/36 |
| Observable preservation | `ΔP = 0.000e+00` exactly, in every window |
| Reachability bound `gain ≤ 1/√(1−ρ²)` | 540 windows, zero violations |
| Effect-size identity | exact to ≤ 5.4e-12 on four systems |
| Truth-free gain prediction | Spearman 1.0000 with realised gain |
| Adaptive selection vs tuned ensemble 3DVar | 16.65 vs 1.28 and 14.09 vs 3.03 |

## Layout

```
bootstrap/      the Navier–Stokes harness every GPU runner imports
solvers/        self-contained solvers (KS spectral, CGL, 3D NS) — numpy only
navier_stokes/  GPU experiment runners; require bootstrap/
kuramoto_sivashinsky/, ginzburg_landau/, finite_difference/
                self-contained CPU experiments
algorithm/      the truth-free correction algorithm and its baselines
analysis/       cross-system analyses (kappa, curve fits, locality, mixing)
paper/          figure generation and manuscript build
notebooks/      recovered console records (.txt — archival, not runnable)
```

## Running

Everything under `solvers/`, `kuramoto_sivashinsky/`, `ginzburg_landau/`,
`finite_difference/`, `algorithm/` and `analysis/` is **numpy only and runs on
CPU**. Put the relevant `*_core.py` from `solvers/` beside the script, or add
`solvers/` to `PYTHONPATH`:

```bash
PYTHONPATH=solvers python algorithm/alg.py
PYTHONPATH=solvers python analysis/master.py
python finite_difference/fd_sensors.py        # fully self-contained
```

Scripts in `navier_stokes/` need a GPU and the bootstrap:

```bash
M3D_BOOT=bootstrap/m3d_kaggle_bootstrap.py python navier_stokes/m3d_test_x3_alpha_sweep.py
```

Most runners take environment variables for trajectory counts and grid sizes;
see the docstring at the top of each file.

## Key entry points

| file | what it produces |
|---|---|
| `navier_stokes/m3d_test_x3_alpha_sweep.py` | the headline causal result, 1.6510× |
| `navier_stokes/m3d_test_F_reachability.py` | the reachability bound, 540 windows |
| `navier_stokes/m3d_test_AD3D.py` | adaptive selection in 3D |
| `navier_stokes/probe3d.py` | point velocity probes as the observable |
| `algorithm/alg.py` | truth-free amplitude selection |
| `algorithm/apply.py` | predicted-gain applicability test and trust gate |
| `algorithm/base.py` | baselines incl. ensemble 3DVar |
| `analysis/master.py` | κ and λ across all three CPU systems |
| `analysis/isolate.py` | observable vs discretisation |
| `paper/figs.py` | all manuscript figures |

## A note on the record

The project corrected itself sixteen times, fifteen of them internally, and
those corrections are documented rather than removed. Two candidate mechanisms
for the pointwise-observable exception were proposed and refuted. One headline
figure moved down by 14% when a single-precision artefact was found, and one
cost claim moved from 21× to 2.9× when a grid-point estimate was replaced by
wall-clock measurement.

The recovered notebooks in `notebooks/` carry their original console output as
comments, so numbers quoted in the manuscript can be traced to the run that
produced them. They are stored as `.txt` because they concatenate many cells
including shell magics — they are an archival record, not importable modules.
Every `.py` in this repository compiles.

## License

MIT — see `LICENSE`.

## Provenance

Tests A–W, Test X, and the post-X archive were recovered from saved notebooks
and cross-checked: 17 figures in the first, 12 in the second, all matching to
the digits quoted. The most recent CPU experiments (KS bands, CGL, finite
difference, adaptive selection, κ analysis) are reproducible from the scripts
here.
