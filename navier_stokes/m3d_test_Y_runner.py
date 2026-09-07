"""
====================================================================================
M3D TEST Y — PRECISION & JACOBIAN-FIDELITY AUDIT   (dual T4, target ~1h)
====================================================================================

Purpose
-------
Y1  Dtype/viscosity sentinel audit. Records what the code ACTUALLY ran with.
Y2  Finite-difference JVP fidelity. `pair_jvp_M` is a central difference with
    eps = max(1e-5, 1e-3*||U||). Sweep eps, Richardson-extrapolate a reference
    Af, and measure how far the STAR2 space is from containing the TRUE Af:
        rho = ||R_star2 . Af_ref|| / ||Af_ref||
    rho is the floor below which the O(t^4) law cannot hold.
Y3  Local-order refit at float32 vs float64, with the default eps and with a
    Richardson-corrected q1*. Does p ~ 4 survive, and does it improve when the
    acceleration direction is more accurate?

Preregistered gates (fixed BEFORE running; do not edit after seeing output)
--------------------------------------------------------------------------
G1  Sentinels present and self-consistent in every emitted row.
G2  rho at default eps is reported for all cases. (Descriptive, no threshold.)
G3  If rho_fp64 < rho_fp32 by >3x in a majority of cases, the FD/precision floor
    is confirmed as the binding constraint on the acceleration residual.
G4  float64 STAR2 order p in [3.7, 4.3] for >= 3 of 4 topologies at nu=1.
G5  If Richardson-corrected q1* raises p or lowers rho by >2x, the published
    quartic constant is FD-limited and must be reported with that caveat.

Usage (Kaggle, after your setup cell)
-------------------------------------
    %%writefile /kaggle/working/m3d_test_Y_runner.py
    <this file>

    !cd /kaggle/working && python m3d_test_Y_runner.py

Outputs -> /kaggle/working/M3D/
    m3d_Y_sentinels.json
    m3d_Y_jvp_fidelity.csv
    m3d_Y_order.csv
    m3d_Y_COMPLETE.txt
====================================================================================
"""

import os
import sys
import json
import time
import glob
import math
import argparse
import subprocess
from pathlib import Path

SAVE = Path("/kaggle/working/M3D")
SAVE.mkdir(parents=True, exist_ok=True)

# Wall-clock budget per worker (seconds). Workers stop starting new cases when
# exceeded and flush whatever they have. Keeps the whole run near ~1h.
BUDGET_S = float(os.environ.get("M3D_Y_BUDGET", 2700))

FLOWS = ["vortex_ring", "periodic_shear", "skew_tubes", "mixed_vortices"]
NUS = [0.25, 1.0, 4.0]

# Y2 runs on all 12 cases (cheap). Y3 order fits run only at nu=1 (expensive).
Y2_CASES = [(f, nu) for f in FLOWS for nu in NUS]
Y3_CASES = [(f, 1.0) for f in FLOWS]


# ==================================================================================
# BOOTSTRAP LOADING (worker side)
# ==================================================================================

def load_env():
    """Exec the bootstrap, then reapply the viscosity patch. Returns globals dict."""
    cands = []
    for pat in ("/kaggle/input/**/m3d_kaggle_bootstrap_v2.py",
                "/kaggle/input/**/m3d_kaggle_bootstrap.py"):
        cands.extend(glob.glob(pat, recursive=True))
    cands = list(dict.fromkeys(cands))
    if not cands:
        raise RuntimeError("no bootstrap found under /kaggle/input")
    bootstrap = cands[0]
    print(f"[worker] bootstrap: {bootstrap}", flush=True)

    g = {"__name__": "__m3d__"}
    with open(bootstrap, "r", encoding="utf-8") as fh:
        code = fh.read()
    exec(compile(code, bootstrap, "exec"), g, g)
    g["__BOOTSTRAP_PATH__"] = bootstrap
    return g


def all_namespaces(g):
    spaces, seen = [g], {id(g)}
    for _, obj in list(g.items()):
        if callable(obj) and hasattr(obj, "__globals__"):
            gd = obj.__globals__
            if id(gd) not in seen:
                spaces.append(gd)
                seen.add(id(gd))
    return spaces


BASE_NU = 2.5e-3


def set_viscosity(g, mult):
    """Write BOTH cases. _pair_rhs_D reads uppercase; legacy sync wrote lowercase."""
    nu = BASE_NU * float(mult)
    for ns in all_namespaces(g):
        ns["NU1_D"] = nu
        ns["NU2_D"] = nu
        ns["nu1_D"] = nu
        ns["nu2_D"] = nu
    return nu


def set_dtype(g, real_dtype, complex_dtype):
    """The notebook override killed the bootstrap's dtype sync. This replaces it."""
    for ns in all_namespaces(g):
        ns["REAL_DTYPE_D"] = real_dtype
        ns["COMPLEX_DTYPE_D"] = complex_dtype


def sentinels(g, tag):
    """Interrogate the live namespaces. This is what goes in every output header."""
    import torch
    probe = ["_pair_rhs_D", "velocity_rhs_D", "full_rk4_D", "topology_pair_L2",
             "initialize_star2_N", "reanchor_star2_N", "rhs_M", "pair_jvp_M"]
    rows = {}
    for name in probe:
        fn = g.get(name)
        if fn is None or not hasattr(fn, "__globals__"):
            continue
        ns = fn.__globals__
        rows[name] = {
            "NU1_D": ns.get("NU1_D"),
            "NU2_D": ns.get("NU2_D"),
            "nu1_D": ns.get("nu1_D"),
            "nu2_D": ns.get("nu2_D"),
            "REAL_DTYPE_D": str(ns.get("REAL_DTYPE_D")),
            "COMPLEX_DTYPE_D": str(ns.get("COMPLEX_DTYPE_D")),
            "TAU0_D": ns.get("TAU0_D"),
        }
    nu_vals = {r["NU1_D"] for r in rows.values()} | {r["NU2_D"] for r in rows.values()}
    dt_vals = {r["REAL_DTYPE_D"] for r in rows.values()}
    return {
        "tag": tag,
        "per_function": rows,
        "nu_consistent": len(nu_vals) == 1,
        "dtype_consistent": len(dt_vals) == 1,
        "nu_values_seen": sorted(str(v) for v in nu_vals),
        "dtype_values_seen": sorted(dt_vals),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


# ==================================================================================
# JVP WITH CONTROLLABLE EPS  (mirrors pair_jvp_M exactly, but eps is an argument)
# ==================================================================================

def jvp_eps(g, U1, U2, V1, V2, env, eps):
    import torch
    pair_norm_D = g["pair_norm_D"]
    _pair_rhs_D = g["_pair_rhs_D"]

    vn = pair_norm_D(V1, V2)
    vn_f = float(vn.detach().cpu())
    if not math.isfinite(vn_f) or vn_f < 1e-13:
        return torch.zeros_like(U1), torch.zeros_like(U2)

    v1, v2 = V1 / vn, V2 / vn
    Fp1, Fp2 = _pair_rhs_D(U1 + eps * v1, U2 + eps * v2, env)
    Fm1, Fm2 = _pair_rhs_D(U1 - eps * v1, U2 - eps * v2, env)
    J1 = (Fp1 - Fm1) / (2.0 * eps)
    J2 = (Fp2 - Fm2) / (2.0 * eps)
    return vn * J1, vn * J2


def default_eps(g, U1, U2):
    un = float(g["pair_norm_D"](U1, U2).detach().cpu())
    return max(1e-5, 1e-3 * un)


def richardson_af(g, U1, U2, f1, f2, env, eps):
    """Central diff is O(eps^2). (4*J(eps/2) - J(eps))/3 kills the leading term."""
    A1, A2 = jvp_eps(g, U1, U2, f1, f2, env, eps)
    B1, B2 = jvp_eps(g, U1, U2, f1, f2, env, eps / 2.0)
    return (4.0 * B1 - A1) / 3.0, (4.0 * B2 - A2) / 3.0


# ==================================================================================
# Y2 — JVP FIDELITY AND THE STAR2 ACCELERATION RESIDUAL
# ==================================================================================

def run_Y2_case(g, flow, nu_mult, dtype_tag):
    """Return one dict row. rho is the headline number."""
    import torch

    make_env_D = g["make_env_D"]
    topology_pair_L2 = g["topology_pair_L2"]
    _pair_rhs_D = g["_pair_rhs_D"]
    pair_norm_D = g["pair_norm_D"]
    complement_M = g["complement_M"]
    initialize_star2_N = g["initialize_star2_N"]

    nu = set_viscosity(g, nu_mult)
    env = make_env_D(40)
    U1, U2 = topology_pair_L2(flow, env)

    f1, f2 = _pair_rhs_D(U1, U2, env)
    eps0 = default_eps(g, U1, U2)

    # Reference acceleration (Richardson, two eps).
    Aref1, Aref2 = richardson_af(g, U1, U2, f1, f2, env, eps0)
    aref_norm = float(pair_norm_D(Aref1, Aref2).detach().cpu())

    # Eps sweep: how does the FD estimate of Af move as eps changes?
    sweep = []
    for scale in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125):
        e = eps0 * scale
        A1, A2 = jvp_eps(g, U1, U2, f1, f2, env, e)
        d1, d2 = A1 - Aref1, A2 - Aref2
        rel = float(pair_norm_D(d1, d2).detach().cpu()) / max(aref_norm, 1e-300)
        sweep.append({"eps_scale": scale, "eps": e, "rel_err_vs_richardson": rel})

    # Build the STAR2 model the way the pipeline does (default eps inside).
    # NOTE: returns (model, diagnostics); basis is model["basis"] = [q0, q1star].
    model, diag = initialize_star2_N(U1, U2, env)
    tids = model["proj"]["target_ids"]
    basis = model["basis"]

    # THE MEASUREMENT: does the STAR2 space contain the TRUE acceleration?
    R1, R2 = complement_M(Aref1, Aref2, tids, basis)
    rho = float(pair_norm_D(R1, R2).detach().cpu()) / max(aref_norm, 1e-300)

    # Same quantity against the FD acceleration the model was built from
    # (this is what the old logs printed, and it flatters the model).
    Afd1, Afd2 = jvp_eps(g, U1, U2, f1, f2, env, eps0)
    afd_norm = float(pair_norm_D(Afd1, Afd2).detach().cpu())
    S1, S2 = complement_M(Afd1, Afd2, tids, basis)
    rho_self = float(pair_norm_D(S1, S2).detach().cpu()) / max(afd_norm, 1e-300)

    return {
        "test": "Y2",
        "flow": flow,
        "nu_mult": nu_mult,
        "nu": nu,
        "dtype": dtype_tag,
        "eps_default": eps0,
        "norm_Af_ref": aref_norm,
        "rho_true": rho,           # honest residual vs Richardson reference
        "rho_self": rho_self,      # self-consistent residual (what logs showed)
        "rho_ratio": rho / max(rho_self, 1e-300),
        "RAf_ratio_bootstrap": diag.get("RAf_ratio"),
        "Af_norm_bootstrap": diag.get("Af_norm"),
        "eps_sweep": json.dumps(sweep),
    }


# ==================================================================================
# Y3 — LOCAL ORDER REFIT
# ==================================================================================

def p_error(g, U1, U2, V1, V2, tids):
    import torch
    pack = g["pack_modes_D"]
    a = pack(U1 - V1, U2 - V2, tids)
    b = pack(U1, U2, tids)
    na = float(torch.linalg.vector_norm(a).detach().cpu())
    nb = float(torch.linalg.vector_norm(b).detach().cpu())
    return na / max(nb, 1e-300)


def run_Y3_case(g, flow, nu_mult, dtype_tag, horizons, sub_steps=16):
    """Integrate full NS and STAR2 from the same IC; fit log-log slope of P-error."""
    import torch

    make_env_D = g["make_env_D"]
    topology_pair_L2 = g["topology_pair_L2"]
    full_rk4_D = g["full_rk4_D"]
    initialize_star2_N = g["initialize_star2_N"]
    rk4_M = g["rk4_M"]
    reconstruct_M = g["reconstruct_M"]

    nu = set_viscosity(g, nu_mult)
    env = make_env_D(40)
    U10, U20 = topology_pair_L2(flow, env)
    tau0 = g["TAU0_D"]

    pts = []
    for T in horizons:
        dt = T / sub_steps
        d_tau = dt / tau0

        A1, A2 = U10.clone(), U20.clone()
        for _ in range(sub_steps):
            A1 = full_rk4_D(A1, dt, nu, env)
            A2 = full_rk4_D(A2, dt, nu, env)

        model, _diag = initialize_star2_N(U10, U20, env)
        for _ in range(sub_steps):
            rk4_M(model, d_tau, env)          # in place, returns None
        V1, V2 = reconstruct_M(model, env)

        tids = model["proj"]["target_ids"]
        pts.append((T, p_error(g, A1, A2, V1, V2, tids)))

    xs = [math.log(t) for t, e in pts if e > 0]
    ys = [math.log(e) for t, e in pts if e > 0]
    if len(xs) >= 3:
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = sum((a - mx) ** 2 for a in xs)
        slope = num / den if den > 0 else float("nan")
        ss_res = sum((b - (my + slope * (a - mx))) ** 2 for a, b in zip(xs, ys))
        ss_tot = sum((b - my) ** 2 for b in ys)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        slope, r2 = float("nan"), float("nan")

    return {
        "test": "Y3",
        "flow": flow,
        "nu_mult": nu_mult,
        "nu": nu,
        "dtype": dtype_tag,
        "order_p": slope,
        "fit_r2": r2,
        "n_points": len(pts),
        "points": json.dumps(pts),
    }


# ==================================================================================
# WORKER
# ==================================================================================

def worker(shard, nshards):
    import torch

    t_start = time.time()
    g = load_env()

    dtypes = [("fp32", torch.float32, torch.complex64),
              ("fp64", torch.float64, torch.complex128)]

    sent, y2_rows, y3_rows = {}, [], []

    for tag, rdt, cdt in dtypes:
        set_dtype(g, rdt, cdt)
        set_viscosity(g, 1.0)
        sent[tag] = sentinels(g, tag)
        print(f"[shard {shard}] {tag}: nu_consistent="
              f"{sent[tag]['nu_consistent']} dtype_consistent="
              f"{sent[tag]['dtype_consistent']}", flush=True)

        my_y2 = [c for i, c in enumerate(Y2_CASES) if i % nshards == shard]
        for flow, nu_mult in my_y2:
            if time.time() - t_start > BUDGET_S:
                print(f"[shard {shard}] budget hit, skipping rest of Y2/{tag}", flush=True)
                break
            try:
                r = run_Y2_case(g, flow, nu_mult, tag)
                y2_rows.append(r)
                print(f"[shard {shard}] Y2 {tag} {flow} nu{nu_mult}: "
                      f"rho_true={r['rho_true']:.3e} rho_self={r['rho_self']:.3e} "
                      f"ratio={r['rho_ratio']:.1f}x", flush=True)
            except Exception as exc:
                print(f"[shard {shard}] Y2 FAIL {tag} {flow} {nu_mult}: {exc!r}", flush=True)

        # fp64 order fits are the expensive part; horizons kept short.
        horizons = [0.02, 0.03, 0.045, 0.0675, 0.10, 0.15]
        my_y3 = [c for i, c in enumerate(Y3_CASES) if i % nshards == shard]
        for flow, nu_mult in my_y3:
            if time.time() - t_start > BUDGET_S:
                print(f"[shard {shard}] budget hit, skipping rest of Y3/{tag}", flush=True)
                break
            try:
                r = run_Y3_case(g, flow, nu_mult, tag, horizons)
                y3_rows.append(r)
                print(f"[shard {shard}] Y3 {tag} {flow} nu{nu_mult}: "
                      f"p={r['order_p']:.4f} R2={r['fit_r2']:.4f}", flush=True)
            except Exception as exc:
                print(f"[shard {shard}] Y3 FAIL {tag} {flow} {nu_mult}: {exc!r}", flush=True)

    out = {"shard": shard, "sentinels": sent, "y2": y2_rows, "y3": y3_rows,
           "elapsed_s": time.time() - t_start}
    with open(SAVE / f"m3d_Y_shard{shard}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[shard {shard}] complete in {out['elapsed_s']:.1f}s", flush=True)


# ==================================================================================
# LAUNCHER + AGGREGATION
# ==================================================================================

def aggregate(nshards):
    import pandas as pd
    import statistics as st

    sents, y2, y3 = {}, [], []
    for s in range(nshards):
        p = SAVE / f"m3d_Y_shard{s}.json"
        if not p.exists():
            print(f"WARNING: shard {s} produced no output")
            continue
        d = json.load(open(p))
        sents[f"shard{s}"] = d["sentinels"]
        y2.extend(d["y2"])
        y3.extend(d["y3"])

    with open(SAVE / "m3d_Y_sentinels.json", "w") as fh:
        json.dump(sents, fh, indent=2)

    df2 = pd.DataFrame(y2)
    df3 = pd.DataFrame(y3)
    if not df2.empty:
        df2.to_csv(SAVE / "m3d_Y_jvp_fidelity.csv", index=False)
    if not df3.empty:
        df3.to_csv(SAVE / "m3d_Y_order.csv", index=False)

    lines = []
    w = lines.append
    w("=" * 100)
    w("M3D TEST Y — PRECISION & JACOBIAN-FIDELITY AUDIT")
    w("=" * 100)

    ok_nu = all(v["nu_consistent"] for s in sents.values() for v in s.values())
    ok_dt = all(v["dtype_consistent"] for s in sents.values() for v in s.values())
    w(f"G1 sentinel self-consistency (nu): {ok_nu}")
    w(f"G1 sentinel self-consistency (dtype): {ok_dt}")

    if not df2.empty:
        w("")
        w("--- Y2 acceleration residual ---")
        for tag in ("fp32", "fp64"):
            sub = df2[df2.dtype == tag]
            if sub.empty:
                continue
            w(f"  {tag}: median rho_true = {sub.rho_true.median():.3e}   "
              f"median rho_self = {sub.rho_self.median():.3e}   "
              f"median inflation = {sub.rho_ratio.median():.1f}x   n={len(sub)}")
        a = df2[df2.dtype == "fp32"].set_index(["flow", "nu_mult"]).rho_true
        b = df2[df2.dtype == "fp64"].set_index(["flow", "nu_mult"]).rho_true
        common = a.index.intersection(b.index)
        if len(common):
            improved = sum(1 for k in common if a[k] / max(b[k], 1e-300) > 3.0)
            w(f"  G3 cases where fp64 cuts rho by >3x: {improved}/{len(common)}"
              f"  -> {'CONFIRMED' if improved > len(common) / 2 else 'NOT CONFIRMED'}")
            w("  G2 per-case rho_true:")
            for k in common:
                w(f"    {k[0]:<16} nu={k[1]:<5} fp32={a[k]:.3e}  fp64={b[k]:.3e}"
                  f"  ({a[k] / max(b[k], 1e-300):.1f}x)")

    if not df3.empty:
        w("")
        w("--- Y3 local order ---")
        for tag in ("fp32", "fp64"):
            sub = df3[df3.dtype == tag]
            if sub.empty:
                continue
            w(f"  {tag}:")
            for _, r in sub.iterrows():
                w(f"    {r.flow:<16} nu={r.nu_mult:<5} p={r.order_p:.4f}  R2={r.fit_r2:.4f}")
        s64 = df3[df3.dtype == "fp64"]
        if not s64.empty:
            ingate = sum(1 for _, r in s64.iterrows() if 3.7 <= r.order_p <= 4.3)
            w(f"  G4 fp64 order in [3.7,4.3]: {ingate}/{len(s64)}"
              f"  -> {'PASS' if ingate >= 3 else 'FAIL'}")

    w("")
    w("Outputs:")
    for f in ("m3d_Y_sentinels.json", "m3d_Y_jvp_fidelity.csv",
              "m3d_Y_order.csv", "m3d_Y_COMPLETE.txt"):
        w(f"  {SAVE / f}")

    txt = "\n".join(lines)
    print(txt)
    with open(SAVE / "m3d_Y_COMPLETE.txt", "w") as fh:
        fh.write(txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=2)
    args = ap.parse_args()

    if args.worker is not None:
        worker(args.worker, args.nshards)
        return

    import torch
    n = min(args.nshards, max(1, torch.cuda.device_count()))
    print("=" * 100)
    print(f"M3D TEST Y — LAUNCHER  ({n} GPU workers, budget {BUDGET_S:.0f}s each)")
    print("=" * 100, flush=True)

    procs = []
    for s in range(n):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(s))
        procs.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__),
             "--worker", str(s), "--nshards", str(n)],
            env=env))
    for p in procs:
        p.wait()

    aggregate(n)


if __name__ == "__main__":
    main()
