"""
====================================================================================
M3D TEST Y-v2 — PRECISION & JACOBIAN-FIDELITY AUDIT (analytic-reference version)
====================================================================================

Changes from Y-v1, all in response to review:
  * Reference derivative is now an EXACT ANALYTIC NS JVP, not Richardson.
    Richardson was a poor reference because the NS RHS is quadratic, so central
    differencing has zero truncation error and Richardson only amplifies roundoff.
  * G5 is now actually implemented and printed. The analytic-JVP STAR2 arm is
    built by monkeypatching pair_jvp_M inside the bootstrap namespaces and then
    calling initialize_star2_N, so both arms traverse identical code paths and
    differ only in how DF(U)f is evaluated.
  * Sentinels are stamped into EVERY emitted row, not recorded once per dtype.
  * fp64 runs FIRST so a budget overrun cannot starve the gates that need it.
  * Y3 horizons are specified in DIMENSIONLESS tau and cover M-B's fit window
    (tau in [0.015, 0.060]), making this a reproduction rather than a much
    earlier independent probe. A second early window is retained for contrast.

Preregistered gates (fixed BEFORE running)
------------------------------------------
G0  COMPLETENESS. The dataset is whole: 24 Y2 rows (2 dtypes x 12 cases) and
    16 Y3 rows (2 dtypes x 2 windows x 4 flows). Every downstream gate ALSO
    requires its own full complement, so a truncated run cannot report a pass
    on a partial denominator (e.g. "3/3 PASS" when the gate says 3 of 4).
G1  Sentinels present in every row, mutually self-consistent, AND equal to the
    value the run requested. Self-consistency alone is insufficient: the
    original viscosity bug had every namespace consistently holding the same
    WRONG nu.
G2  rho_fd and rho_analytic reported for all cases. Descriptive, no threshold.
G3  rho_fd(fp64) < 1e-6 * rho_fd(fp32) in >= 9/12 matched cases  ->  the
    residual is set by a precision floor, not by a structural error term.
    (Deliberately weaker than eps-ratio agreement: floating-point error need
    not scale exactly as machine epsilon once FFT roundoff, normalization,
    projection and basis conditioning are in the budget.)
G3b DESCRIPTIVE, no threshold: the observed rho_fp32/rho_fp64 ratio against
    eps32/eps64 = 5.37e8. A truncation term would be dtype-independent, so
    agreement here is the specific fingerprint that rules one out. Reported
    for the record; it does not decide anything.
G4  fp64 STAR2 order p in [3.7, 4.3] for >= 3 of 4 topologies at nu=1, in the
    M-B window.
G5  Analytic-JVP STAR2 vs FD-JVP STAR2, DECIDED ON delta_p ALONE:
      if |delta_p| > 0.05 in any fp64 M-B case -> the FD construction
         materially affects the quartic observation;
      otherwise -> the local quartic behaviour is robust to replacing the FD
         JVP with an analytic one.
    rho_fd / rho_analytic is reported as a numerical-floor diagnostic only.
    It is NOT a gate: a 10x ratio between 1e-15 and 1e-16 is noise about
    noise and says nothing about the order.

Usage
-----
    Add to your Kaggle dataset, then:
    !python /kaggle/input/<...>/m3d_test_Y2_runner.py

Outputs -> /kaggle/working/M3D/
    m3d_Y2_jvp_fidelity.csv
    m3d_Y2_order.csv
    m3d_Y2_COMPLETE.txt
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

BUDGET_S = float(os.environ.get("M3D_Y2_BUDGET", 3000))

FLOWS = ["vortex_ring", "periodic_shear", "skew_tubes", "mixed_vortices"]
NUS = [0.25, 1.0, 4.0]
Y2_CASES = [(f, nu) for f in FLOWS for nu in NUS]
Y3_CASES = [(f, 1.0) for f in FLOWS]

# M-B fit window in dimensionless tau, plus an early window for contrast.
TAU_WINDOWS = {
    "MB": [0.015, 0.021, 0.030, 0.042, 0.060],
    "early": [0.0015, 0.0021, 0.0030, 0.0042, 0.0060],
}
SUB_STEPS = 32

BASE_NU = 2.5e-3


# ==================================================================================
# BOOTSTRAP
# ==================================================================================

def preflight_device():
    """
    Fail fast and legibly if the GPU cannot actually execute kernels.

    Kaggle's PyTorch ships sm_70..sm_120. A Tesla P100 is sm_60, so every CUDA
    op raises `no kernel image is available for execution on the device` — and
    without this check the first failure surfaces from deep inside the
    bootstrap's import-time make_env_D() call, which reads as a bootstrap bug.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("PREFLIGHT: no CUDA device visible.")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    try:
        _ = (torch.ones(8, device="cuda", dtype=torch.float64) * 2.0).sum().item()
        _ = torch.fft.fftn(torch.zeros(4, 4, 4, device="cuda",
                                       dtype=torch.complex128))
    except Exception as exc:
        raise RuntimeError(
            f"PREFLIGHT FAILED on {name} (sm_{cap[0]}{cap[1]}): {exc}\n"
            f"  This PyTorch build has no kernels for this architecture.\n"
            f"  Switch the Kaggle accelerator to T4 x2 and rerun."
        ) from exc
    print(f"[preflight] {name} sm_{cap[0]}{cap[1]} — fp64 + complex128 OK",
          flush=True)


def load_env():
    preflight_device()
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
    exec(compile(open(bootstrap).read(), bootstrap, "exec"), g, g)
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


def set_viscosity(g, mult):
    nu = BASE_NU * float(mult)
    for ns in all_namespaces(g):
        ns["NU1_D"] = nu   # _pair_rhs_D reads uppercase
        ns["NU2_D"] = nu
        ns["nu1_D"] = nu   # legacy sync wrote only lowercase
        ns["nu2_D"] = nu
    return nu


def set_dtype(g, rdt, cdt):
    for ns in all_namespaces(g):
        ns["REAL_DTYPE_D"] = rdt
        ns["COMPLEX_DTYPE_D"] = cdt


def row_sentinels(g):
    """Compact per-row stamp. Interrogates the live namespace, does not assume."""
    fn = g["_pair_rhs_D"]
    ns = fn.__globals__
    probe = ["_pair_rhs_D", "velocity_rhs_D", "full_rk4_D", "initialize_star2_N",
             "rhs_M", "pair_jvp_M"]
    nus, dts = set(), set()
    for name in probe:
        f = g.get(name)
        if f is None or not hasattr(f, "__globals__"):
            continue
        s = f.__globals__
        nus.add(s.get("NU1_D"))
        nus.add(s.get("NU2_D"))
        dts.add(str(s.get("REAL_DTYPE_D")))
    return {
        "sent_NU1_D": ns.get("NU1_D"),
        "sent_NU2_D": ns.get("NU2_D"),
        "sent_real_dtype": str(ns.get("REAL_DTYPE_D")),
        "sent_complex_dtype": str(ns.get("COMPLEX_DTYPE_D")),
        "sent_TAU0_D": ns.get("TAU0_D"),
        "sent_nu_consistent": len(nus) == 1,
        "sent_dtype_consistent": len(dts) == 1,
    }


# ==================================================================================
# EXACT ANALYTIC NS JVP
# ==================================================================================

def analytic_velocity_jvp(g, Uhat, Vhat, nu, env):
    """
    Exact directional derivative of velocity_rhs_D.

    velocity_rhs_D computes, with D = dealias o Leray:
        F(U) = D[ D(FFT(u x w(u))) - nu K^2 D(U) ]
    The cross term is bilinear, so
        DF(U)V = D[ D(FFT(u x w_v + v x w_u)) - nu K^2 Vhat ]
    with uhat = D(U), Vhat = D(V).
    """
    import torch
    dealias_D = g["dealias_D"]
    project_divfree_D = g["project_divfree_D"]
    curl_hat_D = g["curl_hat_D"]
    CDT = g["COMPLEX_DTYPE_D"]

    Uh = dealias_D(project_divfree_D(Uhat, env), env)
    Vh = dealias_D(project_divfree_D(Vhat, env), env)

    u = torch.fft.ifftn(Uh, dim=(-3, -2, -1)).real
    v = torch.fft.ifftn(Vh, dim=(-3, -2, -1)).real
    wu = torch.fft.ifftn(curl_hat_D(Uh, env), dim=(-3, -2, -1)).real
    wv = torch.fft.ifftn(curl_hat_D(Vh, env), dim=(-3, -2, -1)).real

    cross = torch.empty_like(u)
    for i, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
        cross[i] = (u[a] * wv[b] - u[b] * wv[a]) + (v[a] * wu[b] - v[b] * wu[a])

    nhat = dealias_D(project_divfree_D(
        torch.fft.fftn(cross, dim=(-3, -2, -1)), env), env)
    rhs = nhat - float(nu) * env["K2"].unsqueeze(0) * Vh
    return dealias_D(project_divfree_D(rhs, env), env).to(CDT)


def make_analytic_pair_jvp(g):
    """Drop-in replacement for pair_jvp_M with the same (U1,U2,V1,V2,env) signature."""
    def pair_jvp_analytic(U1, U2, V1, V2, env):
        return (analytic_velocity_jvp(g, U1, V1, g["NU1_D"], env),
                analytic_velocity_jvp(g, U2, V2, g["NU2_D"], env))
    return pair_jvp_analytic


def patch_jvp(g, fn):
    """Install fn as pair_jvp_M everywhere. Returns the previous callable."""
    old = g["pair_jvp_M"]
    for ns in all_namespaces(g):
        if "pair_jvp_M" in ns:
            ns["pair_jvp_M"] = fn
    g["pair_jvp_M"] = fn
    return old


# ==================================================================================
# Y2 — ACCELERATION RESIDUAL, FD ARM vs ANALYTIC ARM
# ==================================================================================

def fd_pair_jvp_eps(g, U1, U2, V1, V2, env, eps):
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
    return vn * (Fp1 - Fm1) / (2.0 * eps), vn * (Fp2 - Fm2) / (2.0 * eps)


def run_Y2_case(g, flow, nu_mult, dtype_tag):
    import torch
    nu = set_viscosity(g, nu_mult)
    env = g["make_env_D"](40)
    U1, U2 = g["topology_pair_L2"](flow, env)
    pair_norm_D = g["pair_norm_D"]
    complement_M = g["complement_M"]

    f1, f2 = g["_pair_rhs_D"](U1, U2, env)
    un = float(pair_norm_D(U1, U2).detach().cpu())
    eps0 = max(1e-5, 1e-3 * un)

    ajvp = make_analytic_pair_jvp(g)
    Aan1, Aan2 = ajvp(U1, U2, f1, f2, env)
    an_norm = float(pair_norm_D(Aan1, Aan2).detach().cpu())

    # How far is the FD acceleration from the analytic one, as a function of eps?
    sweep = []
    for scale in (100.0, 10.0, 1.0, 0.1, 0.01):
        e = eps0 * scale
        A1, A2 = fd_pair_jvp_eps(g, U1, U2, f1, f2, env, e)
        rel = float(pair_norm_D(A1 - Aan1, A2 - Aan2).detach().cpu()) / max(an_norm, 1e-300)
        sweep.append({"eps_scale": scale, "eps": e, "rel_err_vs_analytic": rel})

    # Arm 1: STAR2 built with the stock FD JVP.
    m_fd, d_fd = g["initialize_star2_N"](U1, U2, env)
    R1, R2 = complement_M(Aan1, Aan2, m_fd["proj"]["target_ids"], m_fd["basis"])
    rho_fd = float(pair_norm_D(R1, R2).detach().cpu()) / max(an_norm, 1e-300)

    # Arm 2: STAR2 built with the analytic JVP (identical code path otherwise).
    old = patch_jvp(g, ajvp)
    try:
        m_an, d_an = g["initialize_star2_N"](U1, U2, env)
        S1, S2 = complement_M(Aan1, Aan2, m_an["proj"]["target_ids"], m_an["basis"])
        rho_an = float(pair_norm_D(S1, S2).detach().cpu()) / max(an_norm, 1e-300)
    finally:
        patch_jvp(g, old)

    row = {
        "test": "Y2", "flow": flow, "nu_mult": nu_mult, "nu": nu, "dtype": dtype_tag,
        "eps_default": eps0, "norm_Af_analytic": an_norm,
        "rho_fd": rho_fd,                      # FD-built basis vs analytic Af
        "rho_analytic": rho_an,                # analytic-built basis vs analytic Af
        "rho_ratio_fd_over_an": rho_fd / max(rho_an, 1e-300),
        "RAf_ratio_bootstrap": d_fd.get("RAf_ratio"),
        "eps_sweep": json.dumps(sweep),
    }
    row.update(row_sentinels(g))
    return row


# ==================================================================================
# Y3 — LOCAL ORDER, FD ARM vs ANALYTIC ARM  (implements G5)
# ==================================================================================

def p_error(g, U1, U2, V1, V2, tids):
    import torch
    pack = g["pack_modes_D"]
    a = pack(U1 - V1, U2 - V2, tids)
    b = pack(U1, U2, tids)
    return (float(torch.linalg.vector_norm(a).detach().cpu())
            / max(float(torch.linalg.vector_norm(b).detach().cpu()), 1e-300))


def fit_loglog(pts):
    xs = [math.log(t) for t, e in pts if e > 0]
    ys = [math.log(e) for t, e in pts if e > 0]
    if len(xs) < 3:
        return float("nan"), float("nan")
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((a - mx) ** 2 for a in xs)
    if den <= 0:
        return float("nan"), float("nan")
    slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / den
    ss_res = sum((b - (my + slope * (a - mx))) ** 2 for a, b in zip(xs, ys))
    ss_tot = sum((b - my) ** 2 for b in ys)
    return slope, (1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def order_one_arm(g, U10, U20, env, nu, tau_list, tau0):
    """Integrate full NS and the (currently installed) STAR2 over each tau."""
    full_rk4_D = g["full_rk4_D"]
    init = g["initialize_star2_N"]
    rk4_M = g["rk4_M"]
    reconstruct_M = g["reconstruct_M"]

    pts = []
    for tau in tau_list:
        d_tau = tau / SUB_STEPS
        dt = d_tau * tau0
        A1, A2 = U10.clone(), U20.clone()
        for _ in range(SUB_STEPS):
            A1 = full_rk4_D(A1, dt, nu, env)
            A2 = full_rk4_D(A2, dt, nu, env)
        model, _ = init(U10, U20, env)
        for _ in range(SUB_STEPS):
            rk4_M(model, d_tau, env)          # mutates in place, returns None
        V1, V2 = reconstruct_M(model, env)
        pts.append((tau, p_error(g, A1, A2, V1, V2, model["proj"]["target_ids"])))
    return pts


def run_Y3_case(g, flow, nu_mult, dtype_tag, window_name):
    nu = set_viscosity(g, nu_mult)
    env = g["make_env_D"](40)
    U10, U20 = g["topology_pair_L2"](flow, env)
    tau0 = g["TAU0_D"]
    taus = TAU_WINDOWS[window_name]

    pts_fd = order_one_arm(g, U10, U20, env, nu, taus, tau0)
    p_fd, r2_fd = fit_loglog(pts_fd)

    old = patch_jvp(g, make_analytic_pair_jvp(g))
    try:
        pts_an = order_one_arm(g, U10, U20, env, nu, taus, tau0)
        p_an, r2_an = fit_loglog(pts_an)
    finally:
        patch_jvp(g, old)

    row = {
        "test": "Y3", "flow": flow, "nu_mult": nu_mult, "nu": nu,
        "dtype": dtype_tag, "window": window_name, "sub_steps": SUB_STEPS,
        "order_p_fd": p_fd, "fit_r2_fd": r2_fd,
        "order_p_analytic": p_an, "fit_r2_analytic": r2_an,
        "delta_p": p_an - p_fd,
        "points_fd": json.dumps(pts_fd),
        "points_analytic": json.dumps(pts_an),
    }
    row.update(row_sentinels(g))
    return row


# ==================================================================================
# WORKER
# ==================================================================================

def worker(shard, nshards):
    import torch
    t0 = time.time()
    g = load_env()
    y2_rows, y3_rows = [], []

    # fp64 FIRST so a budget overrun cannot starve G3/G4/G5.
    for tag, rdt, cdt in (("fp64", torch.float64, torch.complex128),
                          ("fp32", torch.float32, torch.complex64)):
        set_dtype(g, rdt, cdt)
        set_viscosity(g, 1.0)

        for i, (flow, nu_mult) in enumerate(Y2_CASES):
            if i % nshards != shard:
                continue
            if time.time() - t0 > BUDGET_S:
                print(f"[shard {shard}] budget hit in Y2/{tag}", flush=True)
                break
            try:
                r = run_Y2_case(g, flow, nu_mult, tag)
                y2_rows.append(r)
                print(f"[shard {shard}] Y2 {tag} {flow} nu{nu_mult}: "
                      f"rho_fd={r['rho_fd']:.3e} rho_an={r['rho_analytic']:.3e} "
                      f"fd/an={r['rho_ratio_fd_over_an']:.2f}", flush=True)
            except Exception as exc:
                print(f"[shard {shard}] Y2 FAIL {tag} {flow} {nu_mult}: {exc!r}", flush=True)

        for wname in ("MB", "early"):
            for i, (flow, nu_mult) in enumerate(Y3_CASES):
                if i % nshards != shard:
                    continue
                if time.time() - t0 > BUDGET_S:
                    print(f"[shard {shard}] budget hit in Y3/{tag}/{wname}", flush=True)
                    break
                try:
                    r = run_Y3_case(g, flow, nu_mult, tag, wname)
                    y3_rows.append(r)
                    print(f"[shard {shard}] Y3 {tag} {wname} {flow}: "
                          f"p_fd={r['order_p_fd']:.4f} p_an={r['order_p_analytic']:.4f} "
                          f"dp={r['delta_p']:+.4f}", flush=True)
                except Exception as exc:
                    print(f"[shard {shard}] Y3 FAIL {tag} {wname} {flow}: {exc!r}", flush=True)

    with open(SAVE / f"m3d_Y2_shard{shard}.json", "w") as fh:
        json.dump({"shard": shard, "y2": y2_rows, "y3": y3_rows,
                   "elapsed_s": time.time() - t0}, fh, indent=2)
    print(f"[shard {shard}] complete in {time.time() - t0:.1f}s", flush=True)


# ==================================================================================
# AGGREGATION
# ==================================================================================

def aggregate(nshards):
    import numpy as np
    import pandas as pd

    y2, y3 = [], []
    for s in range(nshards):
        p = SAVE / f"m3d_Y2_shard{s}.json"
        if not p.exists():
            print(f"WARNING: shard {s} produced no output")
            continue
        d = json.load(open(p))
        y2.extend(d["y2"])
        y3.extend(d["y3"])

    df2, df3 = pd.DataFrame(y2), pd.DataFrame(y3)
    if not df2.empty:
        df2.to_csv(SAVE / "m3d_Y2_jvp_fidelity.csv", index=False)
    if not df3.empty:
        df3.to_csv(SAVE / "m3d_Y2_order.csv", index=False)

    L = []
    w = L.append
    w("=" * 100)
    w("M3D TEST Y-v2 — ANALYTIC-REFERENCE PRECISION AUDIT")
    w("=" * 100)

    # ------------------------------------------------------------------
    # G0 — COMPLETENESS / PROVENANCE GATE
    # ------------------------------------------------------------------
    expected_y2 = 2 * len(Y2_CASES)                    # 2 dtypes x 12 cases
    expected_y3 = 2 * len(TAU_WINDOWS) * len(Y3_CASES)  # 2 dtypes x 2 windows x 4
    g0_y2 = len(df2) == expected_y2
    g0_y3 = len(df3) == expected_y3
    g0 = g0_y2 and g0_y3
    w(f"G0 complete dataset: {'PASS' if g0 else 'FAIL'}")
    w(f"  Y2 rows: {len(df2)}/{expected_y2}")
    w(f"  Y3 rows: {len(df3)}/{expected_y3}")
    if not g0:
        w("  WARNING: dataset incomplete. Downstream gates below are reported")
        w("           but MUST NOT be treated as evaluated.")

    # ------------------------------------------------------------------
    # G1 — sentinels: consistent AND correct
    # ------------------------------------------------------------------
    def sentinel_expected_ok(r):
        expected_nu = BASE_NU * float(r["nu_mult"])
        try:
            nu_ok = (abs(float(r["sent_NU1_D"]) - expected_nu) < 1e-15
                     and abs(float(r["sent_NU2_D"]) - expected_nu) < 1e-15)
        except (TypeError, ValueError):
            return False
        dtype_str = str(r["sent_real_dtype"])
        dtype_ok = ("float64" in dtype_str if r["dtype"] == "fp64"
                    else "float32" in dtype_str)
        return (bool(r.get("sent_nu_consistent"))
                and bool(r.get("sent_dtype_consistent"))
                and nu_ok and dtype_ok)

    all_rows = y2 + y3
    bad = [r for r in all_rows if not sentinel_expected_ok(r)]
    g1 = bool(all_rows) and not bad
    w("")
    w(f"G1 per-row sentinels consistent AND correct: {'PASS' if g1 else 'FAIL'}"
      f"  (rows={len(all_rows)}, bad={len(bad)})")
    for r in bad[:8]:
        w(f"  BAD  {r.get('test')} {r.get('dtype')} {r.get('flow')} "
          f"nu_mult={r.get('nu_mult')}  NU1_D={r.get('sent_NU1_D')} "
          f"expected={BASE_NU * float(r['nu_mult']):.6g} "
          f"real_dtype={r.get('sent_real_dtype')}")

    eps_ratio = np.finfo(np.float32).eps / np.finfo(np.float64).eps
    if not df2.empty:
        w("")
        w("--- Y2 acceleration residual vs ANALYTIC reference ---")
        for tag in ("fp64", "fp32"):
            sub = df2[df2.dtype == tag]
            if sub.empty:
                continue
            w(f"  {tag}: median rho_fd={sub.rho_fd.median():.3e}   "
              f"median rho_analytic={sub.rho_analytic.median():.3e}   n={len(sub)}")
        a = df2[df2.dtype == "fp32"].set_index(["flow", "nu_mult"]).rho_fd
        b = df2[df2.dtype == "fp64"].set_index(["flow", "nu_mult"]).rho_fd
        common = a.index.intersection(b.index)
        if len(common):
            w("")
            w("  G3 precision-floor test (gate: rho_fp64 < 1e-6 * rho_fp32):")
            ok = 0
            ratios = []
            for k in common:
                ratio = a[k] / max(b[k], 1e-300)
                ratios.append(ratio)
                good = b[k] < 1e-6 * a[k]
                ok += good
                w(f"    {k[0]:<16} nu={k[1]:<5} fp32={a[k]:.3e} fp64={b[k]:.3e}"
                  f"  ratio={ratio:.3e}  {'ok' if good else 'FAIL'}")
            g3_pass = (len(common) == len(Y2_CASES)) and (ok >= 9)
            w(f"  G3: {ok}/{len(common)} pass (need >=9 of {len(Y2_CASES)} matched)"
              f"  -> {'PASS (precision floor confirmed)' if g3_pass else 'FAIL'}")
            w("")
            w(f"  G3b DESCRIPTIVE (no threshold). eps32/eps64 = {eps_ratio:.3e}")
            med = float(np.median(ratios))
            spread = (max(ratios) - min(ratios)) / med if med > 0 else float('nan')
            w(f"    observed median ratio = {med:.3e}   spread = {100*spread:.1f}%"
              f"   median/eps_ratio = {med/eps_ratio:.3f}")
            w("    (A structural truncation term would be dtype-independent and")
            w("     would NOT scale this way. Recorded, not gated.)")

    if not df3.empty:
        w("")
        w("--- Y3 local order (M-B window tau in [0.015, 0.060]) ---")
        for tag in ("fp64", "fp32"):
            for wname in ("MB", "early"):
                sub = df3[(df3.dtype == tag) & (df3.window == wname)]
                if sub.empty:
                    continue
                w(f"  {tag} / {wname}:")
                for _, r in sub.iterrows():
                    w(f"    {r.flow:<16} p_fd={r.order_p_fd:7.4f} (R2={r.fit_r2_fd:.4f})"
                      f"   p_analytic={r.order_p_analytic:7.4f} (R2={r.fit_r2_analytic:.4f})"
                      f"   dp={r.delta_p:+.4f}")
        s = df3[(df3.dtype == "fp64") & (df3.window == "MB")]
        if not s.empty:
            ingate = sum(1 for _, r in s.iterrows() if 3.7 <= r.order_p_fd <= 4.3)
            g4_pass = (len(s) == len(Y3_CASES)) and (ingate >= 3)
            w(f"  G4 fp64 order_p_fd in [3.7,4.3]: {ingate}/{len(s)}"
              f" (need >=3 of {len(Y3_CASES)})"
              f"  -> {'PASS' if g4_pass else 'FAIL'}")

    # G5 — now actually evaluated.
    w("")
    w("--- G5: does the analytic JVP change the answer? ---")
    if df3.empty and df2.empty:
        w("  G5 NOT EVALUATED (no rows)")
    else:
        # DECISIVE: delta_p on the fp64 M-B cases only.
        decisive = df3[(df3.dtype == "fp64") & (df3.window == "MB")] \
            if not df3.empty else df3
        moved_p = ([(r.flow, r.delta_p) for _, r in decisive.iterrows()
                    if abs(r.delta_p) > 0.05] if not decisive.empty else [])
        w("  decisive quantity: |delta_p| on fp64 / M-B cases")
        if decisive.empty:
            w("    G5 NOT EVALUATED (no fp64 M-B rows)")
        else:
            for _, r in decisive.iterrows():
                w(f"    {r.flow:<16} p_fd={r.order_p_fd:7.4f}"
                  f"  p_analytic={r.order_p_analytic:7.4f}  dp={r.delta_p:+.4f}"
                  f"  {'MOVED' if abs(r.delta_p) > 0.05 else 'stable'}")
            if len(decisive) != len(Y3_CASES):
                verdict = (f"NOT EVALUATED — incomplete fp64 M-B dataset "
                           f"({len(decisive)}/{len(Y3_CASES)})")
            elif moved_p:
                verdict = "FD construction MATERIALLY AFFECTS the quartic observation"
            else:
                verdict = "local quartic behaviour ROBUST to FD -> analytic JVP"
            w(f"  G5 verdict: {verdict}  ({len(moved_p)}/{len(decisive)} moved)")

        # DIAGNOSTIC ONLY: numerical-floor comparison, no gate.
        if not df2.empty:
            w("")
            w("  rho_fd / rho_analytic (numerical-floor diagnostic, NOT a gate):")
            for tag in ("fp64", "fp32"):
                sub = df2[df2.dtype == tag]
                if sub.empty:
                    continue
                w(f"    {tag}: median ratio = "
                  f"{sub.rho_ratio_fd_over_an.median():.2f}   "
                  f"min = {sub.rho_ratio_fd_over_an.min():.2f}   "
                  f"max = {sub.rho_ratio_fd_over_an.max():.2f}")
            w("    (Both arms may sit at the machine floor, where this ratio is")
            w("     meaningless. Interpret only alongside delta_p above.)")

    w("")
    w("Outputs:")
    for f in ("m3d_Y2_jvp_fidelity.csv", "m3d_Y2_order.csv", "m3d_Y2_COMPLETE.txt"):
        w(f"  {SAVE / f}")

    txt = "\n".join(L)
    print(txt)
    open(SAVE / "m3d_Y2_COMPLETE.txt", "w").write(txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=None)
    args = ap.parse_args()

    if args.worker is not None:
        worker(args.worker, args.nshards or 1)
        return

    import torch
    n = args.nshards or max(1, torch.cuda.device_count())
    print("=" * 100)
    print(f"M3D TEST Y-v2 — LAUNCHER  ({n} worker(s), budget {BUDGET_S:.0f}s each)")
    print(f"GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    print("=" * 100, flush=True)
    preflight_device()

    procs = []
    for s in range(n):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(s))
        procs.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__),
             "--worker", str(s), "--nshards", str(n)], env=env))
    for p in procs:
        p.wait()
    aggregate(n)


if __name__ == "__main__":
    main()
