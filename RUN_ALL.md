# CPU experiments — every file and its command

All 22 files here are **numpy only and run on CPU**. Select CPU as the
accelerator; a GPU buys nothing and wastes quota.

Put every file in ONE directory. Several scripts import a solver core from
alongside them.

    D=/kaggle/input/datasets/<user>/<dir>
    export PYTHONPATH=$D
    cd /kaggle/working

Then run any command below. Env vars set trajectory counts and system
parameters; defaults are the values used for the reported results.

---

## Solver cores — not run directly

| file | used by |
|---|---|
| `ks_core.py` | alg, apply, adapt, base, kappa, kappa2, master, isolate |
| `cgl_core.py` | cgl_run, cgl2, master |
| `ns_core.py` | probe3d, loop3d |

`fd_sensors.py`, `mix.py`, `locality.py` and `modelform.py` inline their own
solver and need nothing beside them.

---

## 1. The algorithm

**Truth-free amplitude selection** — does the method pick the right correction
size without a reference? Reports the fraction of hindsight-optimal benefit
captured.

    ALG_NTRAJ=40 ALG_BAND=damped   ALG_FIT=probe python -u $D/alg.py
    ALG_NTRAJ=40 ALG_BAND=unstable ALG_FIT=probe python -u $D/alg.py
    ALG_NTRAJ=40 ALG_BAND=unstable ALG_FIT=2pt   python -u $D/alg.py
    ALG_NTRAJ=40 ALG_BAND=unstable ALG_FIT=small python -u $D/alg.py

`ALG_FIT` selects how the response is estimated: `probe` evaluates each
candidate directly, `2pt` fits a quadratic from two points, `small` fits near
zero amplitude. ~7 min each.

**Applicability prediction and the trust gate** — does predicted gain track
realised gain, and does the gate decline rather than harm?

    APP_NTRAJ=60 APP_BAND=unstable python -u $D/apply.py
    APP_NTRAJ=60 APP_BAND=damped   python -u $D/apply.py

~9 min each. The unstable band is the one that matters — it contains the
windows where a fixed amplitude loses.

**Adaptive subspace + amplitude selection** — the deployed algorithm.

    AD_NTRAJ=30 AD_BAND=unstable python -u $D/adapt.py
    AD_NTRAJ=30 AD_BAND=damped   python -u $D/adapt.py

~5 min each. For the amplitude control, edit `ALPHAS=[1.0]` in the file and
rerun; that isolates subspace selection from amplitude selection.

**Baselines** — six methods on identical trajectories, including a tuned
ensemble 3DVar, with an external-information column.

    BASE_NTRAJ=30 BASE_BAND=damped   python -u $D/base.py
    BASE_NTRAJ=30 BASE_BAND=unstable python -u $D/base.py
    BASE_NTRAJ=30 BASE_ROBS=1e-8 BASE_BAND=damped python -u $D/base.py

~4 min each. The third checks the 3DVar baseline is not sensitive to its
observation-error parameter.

---

## 2. What sets the effect size

**κ, the hidden-driven share**

    KP_NTRAJ=12 python -u $D/kappa.py
    KP_NTRAJ=12 python -u $D/kappa2.py

`kappa2.py` adds the three-term reconstruction that tests whether the neglected
defect term closes the gap. It does not — that result is reported as negative.
~45 min each.

**All three CPU systems in one run, plus the curve test**

    M_KS=10 M_CGL=12 M_FD=12 python -u $D/master.py

~2–3 h. Writes `master_kappa.json`. Then, in the same directory, seconds each:

    python -u $D/classes.py master_kappa.json
    python -u $D/why.py     master_kappa.json
    python -u $D/curve.py   master_kappa.json

`classes.py` tests whether the shape parameter is set by the observation
operator; `why.py` tests the alignment/dimension explanation (refuted);
`curve.py` fits a single curve across systems.

---

## 3. The validity boundary

**Observable vs discretisation** — the confound-breaking test.

    IS_NTRAJ=10            python -u $D/isolate.py
    IS_NTRAJ=10 IS_NDIM=6  python -u $D/isolate.py
    IS_NTRAJ=10 IS_NDIM=24 python -u $D/isolate.py

~4 min each. Note the metric is *reconstruction error* — low is good.

**Contamination sweep** — how many pointwise functionals break the law.

    MX_NTRAJ=10 python -u $D/mix.py

~8 min. Self-contained.

**Advective locality** — a proposed mechanism, refuted.

    LOC_NTRAJ=8 python -u $D/locality.py

~30 min. Self-contained. Reported as a negative result.

---

## 4. Other systems

**Complex Ginzburg–Landau, cubic nonlinearity**

    CGL_NTRAJ=20  python -u $D/cgl_run.py
    CGL2_NTRAJ=20 python -u $D/cgl2.py

`cgl2.py` is the version with four predictions registered before the run.
~40–60 min each. Needs `cgl_core.py`.

**Finite differences + point sensors** — no spectral operators anywhere.

    FD_NTRAJ=25            python -u $D/fd_sensors.py
    FD_NSEN=6  FD_NTRAJ=25 python -u $D/fd_sensors.py
    FD_NSEN=24 FD_NTRAJ=25 python -u $D/fd_sensors.py

~40 min each. Self-contained.

**3D Navier–Stokes with point velocity probes**

    PB_NTRAJ=12 python -u $D/probe3d.py

~1–2 h. Needs `ns_core.py` beside it (the import is `from ns_core import NS`).
If the reported baseline error is below 0.3%,
raise `PB_AMP` or `PB_NSTEP` — the correction needs something to work on.

**3D closed loop, CPU version**

    LP_NTRAJ=8 LP_M=20 python -u $D/loop3d.py

~1 h. Needs `ns_core.py` and `probe3d.py`. This one failed on scale at N=24 and
is included for completeness; the GPU runner at N=40 is the usable version.

---

## 5. Model-form error — the scope limit

    MF_NTRAJ=10                python -u $D/modelform.py
    MF_NTRAJ=10 MF_TERM=adv    python -u $D/modelform.py
    MF_NTRAJ=10 MF_TERM=diff   python -u $D/modelform.py

~35 min each. Self-contained.

Truth evolves under the exact equation; the model uses a coefficient scaled by
1+δ. This is the only test where the governing equations themselves are wrong,
rather than only the reduction.

**Read the `G_hat/G` column.** At δ = 0 it should be ≈ 1. If it grows with δ,
the method is overconfident under model-form error — which is what the smoke
test showed (7.74 at δ = 0.10, reporting 11.7 while delivering 1.8). That is a
structural limitation, not a bug: the bias sits identically in the ROM and the
estimator, so the method is internally consistent and cannot see it.

---

## Suggested order if you only have one session

1. `alg.py` (probe, both bands) — the core algorithmic claim
2. `adapt.py` (both bands) — the deployed method
3. `modelform.py` — the scope limit, and the newest result
4. `isolate.py` + `mix.py` — the validity boundary
5. `master.py` — the cross-system analysis, if time allows
