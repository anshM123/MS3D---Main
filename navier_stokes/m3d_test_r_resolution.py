#!/usr/bin/env python3
"""
M3D TEST R — RESOLUTION / REYNOLDS SCALING: is the testbed dynamically rich enough?

Runs the 12 Test-X cases across two independent CUDA subprocesses.

New in v2
---------
  --precision {fp32,fp64}   Precision of the baseline trajectory, the STAR2
                            construction and the forecast branches. The W
                            estimator is ALWAYS fp64 (it already was in v1).
                            fp32 reproduces v1; fp64 is the clean run.
  --jvp {fd,analytic}       DF(U)V used to build STAR2. The stock 'fd' path
                            is exact for a quadratic F but carries a 1/eps
                            roundoff amplification (~400x machine eps, per
                            Test Y-v2). 'analytic' sits at machine eps.
  --deriv-rel FLOAT         Step for map_jet_X. Default 1e-5 as in v1. Note
                            Phi_h is NOT quadratic, so this step has real
                            truncation error, and the Hessian second
                            difference amplifies roundoff by 1/eps^2.
                            eps_machine^(1/4) ~ 1.2e-4 is the textbook
                            optimum for the Hessian term.
  --noise-sweep             Add perturbed-estimator branches (see below).

Estimator-tolerance sweep
-------------------------
Test X as run in v1 cannot separate "the correction works" from "the oracle
works", because the estimator is accurate enough that EST and ORACLE are
indistinguishable. The sweep degrades the estimator on purpose and finds
where the benefit dies, turning the bridge inequality

    E_est <= E_oracle + L_P ||q_hat - q||

from a stated bound into a measured threshold.

Two perturbation families, both confined to range(Q) so that P(V+c) = PV is
preserved exactly and every branch still starts from an identical resolved
state:

  scale     q_hat -> (1 + delta) * q_hat          (amplitude error)
  direction q_hat -> q_hat + delta*||q_hat||*n,   (direction error)
            n a unit random vector projected into range(Q)

Preregistered sweep gates (fixed BEFORE running)
------------------------------------------------
  N0  Every perturbed branch has start_target_mismatch < 1e-6, i.e. the
      perturbation stayed inside range(Q).
  N1  Median gain is monotonically non-increasing in delta for the direction
      family: Spearman(delta, median gain) < 0.
  N2  At delta = 0.10 (a 10% estimator error, ~1000x worse than the measured
      W residual) the median gain still exceeds 1.0, for both families.
  N3  DESCRIPTIVE, no threshold: the delta at which median gain crosses 1.0,
      reported per family and per case.
Each worker sees only one physical GPU via CUDA_VISIBLE_DEVICES, so the
baseline's DEVICE_D='cuda' and frozen default args map to the correct T4.

Usage in Kaggle:
    !python /kaggle/input/<dataset-name>/m3d_test_x_dual_gpu.py

The script auto-finds m3d_kaggle_bootstrap*.py under /kaggle/input.
Outputs go to /kaggle/working/M3D.
"""

import os, sys, glob, subprocess, argparse, time, zlib
from pathlib import Path


def _find_bootstrap():
    pats = [
        "/kaggle/input/**/m3d_kaggle_bootstrap_v2.py",
        "/kaggle/input/**/m3d_kaggle_bootstrap.py",
        "/kaggle/working/**/m3d_kaggle_bootstrap_v2.py",
        "/kaggle/working/**/m3d_kaggle_bootstrap.py",
    ]
    hits = []
    for p in pats:
        hits.extend(glob.glob(p, recursive=True))
    if not hits:
        raise FileNotFoundError(
            "Could not find m3d_kaggle_bootstrap.py or v2 under /kaggle/input or /kaggle/working"
        )
    # Prefer v2 if both are present.
    hits = sorted(set(hits), key=lambda x: ("v2" not in Path(x).name, x))
    return hits[0]


BUILD_ID = "R1-2026-08-30-resolution-reynolds"


def _banner(where):
    """Fingerprint the running file. A stale copy cannot print this."""
    print(f"[{where}] BUILD_ID = {BUILD_ID}", flush=True)
    print(f"[{where}] source   = {os.path.abspath(__file__)}", flush=True)


def parent_main(precision="fp64", jvp="analytic", deriv_rel=1e-5,
                noise_sweep=True, tag="fp64_analytic"):
    # Do not import torch in the parent before launching children; children set CUDA_VISIBLE_DEVICES first.
    out = Path("/kaggle/working/M3D")
    out.mkdir(parents=True, exist_ok=True)

    script = str(Path(__file__).resolve())
    print("=" * 100)
    print("M3D TEST R — DUAL GPU LAUNCHER  [resolution x Reynolds sweep]")
    print("=" * 100)
    _banner("launcher")
    print(f"precision (trajectory/forecast): {precision}   "
          f"(W estimator is always fp64)")
    print(f"STAR2 JVP: {jvp}")
    print(f"map_jet deriv_rel: {deriv_rel:.3e}")
    print(f"noise sweep: {noise_sweep}")
    print(f"output tag: {tag}")
    print("Launching GPU 0 and GPU 1 workers in parallel...")
    print("Each worker checkpoints its own 6 cases.")
    print()

    procs = []
    for gpu in (0, 1):
        env = os.environ.copy()
        # The worker will also set this before importing torch; set it here too.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [sys.executable, "-u", script, "--worker", "--gpu", str(gpu),
               "--precision", precision, "--jvp", jvp,
               "--deriv-rel", repr(deriv_rel), "--tag", tag]
        procs.append((gpu, subprocess.Popen(cmd, env=env)))

    failed = []
    for gpu, p in procs:
        rc = p.wait()
        if rc != 0:
            failed.append((gpu, rc))

    if failed:
        raise RuntimeError(
            f"One or more GPU workers failed: {failed}. Re-run this same command; completed shard cases are checkpointed."
        )

    finalize_outputs(tag, precision, jvp)


def worker_main(physical_gpu: int, precision="fp64", jvp="analytic",
                deriv_rel=1e-5, noise_sweep=True, tag="fp64_analytic"):
    # IMPORTANT: must happen before importing torch/bootstrap.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)

    import gc
    import math
    import numpy as np
    import pandas as pd
    import torch

    # Worker sees exactly one GPU, always addressed internally as cuda:0.
    if not torch.cuda.is_available():
        raise RuntimeError(f"GPU worker {physical_gpu}: CUDA unavailable")
    torch.cuda.set_device(0)

    bootstrap = _find_bootstrap()
    print(f"[GPU{physical_gpu}] bootstrap: {bootstrap}", flush=True)
    exec(open(bootstrap, encoding="utf-8").read(), globals())

    # Robust viscosity sync patch — works even if the user uploaded the older bootstrap.
    def sync_fixed(real_dtype, complex_dtype, nu, tau0):
        globals()["REAL_DTYPE_D"] = real_dtype
        globals()["COMPLEX_DTYPE_D"] = complex_dtype
        globals()["NU1_D"] = float(nu)
        globals()["NU2_D"] = float(nu)
        globals()["nu1_D"] = float(nu)
        globals()["nu2_D"] = float(nu)
        globals()["TAU0_D"] = float(tau0)
        for gd in NAMESPACES_MB:
            gd["REAL_DTYPE_D"] = real_dtype
            gd["COMPLEX_DTYPE_D"] = complex_dtype
            gd["NU1_D"] = float(nu)
            gd["NU2_D"] = float(nu)
            gd["nu1_D"] = float(nu)
            gd["nu2_D"] = float(nu)
            gd["TAU0_D"] = float(tau0)
    globals()["sync_m3d_globals_MB"] = sync_fixed

    SAVE_DIR_X = Path("/kaggle/working/M3D")
    SAVE_DIR_X.mkdir(parents=True, exist_ok=True)

    # ---- trajectory / forecast precision (W estimator stays fp64 regardless) ----
    RDT_X = torch.float64 if precision == "fp64" else torch.float32
    CDT_X = torch.complex128 if precision == "fp64" else torch.complex64

    # ---- optional exact analytic NS Jacobian-vector product ----
    # velocity_rhs_D computes, with D = dealias o Leray:
    #     F(U) = D[ D(FFT(u x w(u))) - nu K^2 D(U) ]
    # The cross term is bilinear, so
    #     DF(U)V = D[ D(FFT(u x w_v + v x w_u)) - nu K^2 Vhat ]
    def _analytic_velocity_jvp(Uhat, Vhat, nu, env):
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
        return dealias_D(project_divfree_D(rhs, env), env).to(COMPLEX_DTYPE_D)

    def _pair_jvp_analytic(U1, U2, V1, V2, env):
        return (_analytic_velocity_jvp(U1, V1, NU1_D, env),
                _analytic_velocity_jvp(U2, V2, NU2_D, env))

    _pair_jvp_fd = globals()["pair_jvp_M"]
    if jvp == "analytic":
        globals()["pair_jvp_M"] = _pair_jvp_analytic
        for _gd in NAMESPACES_MB:
            if "pair_jvp_M" in _gd:
                _gd["pair_jvp_M"] = _pair_jvp_analytic
        print(f"[GPU{physical_gpu}] STAR2 JVP: analytic", flush=True)
    else:
        print(f"[GPU{physical_gpu}] STAR2 JVP: finite difference (stock)", flush=True)

    # ---- estimator-tolerance sweep configuration ----
    # direction: q_hat + delta*||q_hat||*n     (n random, projected into Q)
    # scale:     q_hat -> (1+delta)*q_hat, i.e. correction amplitude alpha=1+delta
    # The scale grid runs past alpha=2 because the clean run showed gain still
    # rising there: Theorem C's optimum minimizes ||b + alpha*Tq + a^2/2 H[q,q]||,
    # which has no reason to sit at alpha=1.
    NOISE_DELTAS_DIR_X = [0.03, 0.10, 0.30, 1.00]
    NOISE_DELTAS_SCALE_X = [0.03, 0.10, 0.30, 1.00, 2.00, 3.00, 5.00, 7.00]
    _banner(f"GPU{physical_gpu}")
    assert "SWEEP_CONFIGS_R" in open(os.path.abspath(__file__)).read(), "wrong file"
    assert "crc32" in open(os.path.abspath(__file__)).read(), "stale seed — wrong file"
    print(f"[GPU{physical_gpu}] alpha grid: "
          f"{[1.0 + d for d in NOISE_DELTAS_SCALE_X]}", flush=True)
    NOISE_FAMILIES_X = ["scale", "direction"]

    # ------------------------------ frozen Test-X settings ------------------------------
    TOPOLOGIES_X = ["vortex_ring", "periodic_shear", "skew_tubes", "mixed_vortices"]
    NU_MULTS_X = [0.25, 1.0, 4.0]
    ALL_CASES_X = [(t, m) for t in TOPOLOGIES_X for m in NU_MULTS_X]
    CASES_THIS_GPU = ALL_CASES_X[physical_gpu::2]  # 6 cases per GPU

    BASE_NU_X = 2.5e-3
    TAU0_X = 40.0 / 3.0
    DT_PHYS_X = 0.01
    DTAU_X = DT_PHYS_X / TAU0_X
    SEGMENT_TAU_X = 0.15
    SEGMENT_STEPS_X = int(round(SEGMENT_TAU_X / DTAU_X))
    FINAL_BASELINE_TAU_X = 1.80
    N_BASELINE_SEGMENTS_X = int(round(FINAL_BASELINE_TAU_X / SEGMENT_TAU_X))
    INTERVENTION_TAUS_X = [0.90, 1.20, 1.50]
    FORECAST_TAU_X = 0.30
    FORECAST_SEGMENTS_X = int(round(FORECAST_TAU_X / SEGMENT_TAU_X))
    POD_RANK_G = int(os.environ.get("M3D_G_POD_RANK", 8))
    POD_SNAPSHOTS_G = int(os.environ.get("M3D_G_POD_SNAPSHOTS", 48))
    # G1 was NULL: snapshots came from the whole trajectory the ROM was then
    # asked to reproduce. In-sample POD on a numerically rank-13 trajectory
    # reproduced it to 1e-7, leaving no accumulated hidden error to correct.
    # Snapshots now come from a TRAINING WINDOW only; everything the ROM is
    # scored on lies beyond it.
    POD_TRAIN_FRAC_G = float(os.environ.get("M3D_G_TRAIN_FRAC", 0.35))
    # Minimum baseline ROM error for the test to be meaningful at all.
    NULL_TEST_FLOOR_G = float(os.environ.get("M3D_G_NULL_FLOOR", 1e-4))
    DERIV_REL_X = float(deriv_rel)

    POINT_FILE = SAVE_DIR_X / f"m3d_R_{tag}_gpu{physical_gpu}_interventions_partial.csv"
    CASE_FILE = SAVE_DIR_X / f"m3d_R_{tag}_gpu{physical_gpu}_cases_partial.csv"
    SWEEP_FILE = SAVE_DIR_X / f"m3d_R_{tag}_gpu{physical_gpu}_sweep_partial.csv"

    # ------------------------------ helpers ------------------------------
    def assert_viscosity_X(nu, real_dtype, complex_dtype):
        sync_m3d_globals_MB(real_dtype, complex_dtype, nu, TAU0_X)
        assert abs(float(globals()["NU1_D"]) - float(nu)) < 1e-14
        assert abs(float(globals()["NU2_D"]) - float(nu)) < 1e-14

    def pair_inner_X(A1, A2, B1, B2):
        return (torch.sum(torch.conj(A1) * B1) + torch.sum(torch.conj(A2) * B2)).real

    def pair_norm_tensor_X(A1, A2):
        return torch.sqrt(torch.clamp(pair_inner_X(A1, A2, A1, A2), min=0.0))

    def pair_norm_X(A1, A2):
        return float(pair_norm_tensor_X(A1, A2).detach().cpu())

    def vec_norm_X(x):
        return float(torch.linalg.vector_norm(x).detach().cpu())

    @torch.no_grad()
    def split_PQ_X(D1, D2, target_ids):
        P1 = torch.zeros_like(D1)
        P2 = torch.zeros_like(D2)
        D1f, D2f = D1.reshape(3, -1), D2.reshape(3, -1)
        P1f, P2f = P1.reshape(3, -1), P2.reshape(3, -1)
        P1f[:, target_ids] = D1f[:, target_ids]
        P2f[:, target_ids] = D2f[:, target_ids]
        return P1, P2, D1 - P1, D2 - P2

    @torch.no_grad()
    def phi_X(U1, U2, env, nu):
        A1, A2 = U1.clone(), U2.clone()
        for _ in range(SEGMENT_STEPS_X):
            A1 = full_rk4_D(A1, DT_PHYS_X, nu, env)
            A2 = full_rk4_D(A2, DT_PHYS_X, nu, env)
        return A1, A2

    @torch.no_grad()
    def map_jet_X(V1, V2, e1, e2, base_phi1, base_phi2, env, nu):
        vnorm = pair_norm_X(V1, V2)
        enorm = pair_norm_X(e1, e2)
        if enorm < 1e-30:
            Z1, Z2 = torch.zeros_like(V1), torch.zeros_like(V2)
            return Z1, Z2, Z1.clone(), Z2.clone()
        eps = float(np.clip(DERIV_REL_X * vnorm / enorm, 1e-6, 5e-2))
        P1, P2 = phi_X(V1 + eps * e1, V2 + eps * e2, env, nu)
        M1, M2 = phi_X(V1 - eps * e1, V2 - eps * e2, env, nu)
        J1, J2 = (P1 - M1) / (2.0 * eps), (P2 - M2) / (2.0 * eps)
        H1 = (P1 + M1 - 2.0 * base_phi1) / (eps ** 2)
        H2 = (P2 + M2 - 2.0 * base_phi2) / (eps ** 2)
        return J1, J2, H1, H2

    @torch.no_grad()
    def star2_forecast_X(U1_start, U2_start, env):
        model, diag = initialize_star2_N(U1_start, U2_start, env)
        for seg in range(FORECAST_SEGMENTS_X):
            for _ in range(SEGMENT_STEPS_X):
                rk4_M(model, DTAU_X, env)
            if seg < FORECAST_SEGMENTS_X - 1:
                model, diag, jump, build = reanchor_star2_N(model, env)
        return reconstruct_M(model, env)

    def target_error_X(A1, A2, T1, T2, target_ids):
        xa = pack_modes_D(A1, A2, target_ids)
        xt = pack_modes_D(T1, T2, target_ids)
        return vec_norm_X(xa - xt) / (vec_norm_X(xt) + 1e-30)

    # ==================================================================
    # SWEEP DESIGN
    # ==================================================================
    # Two questions, deliberately separated:
    #
    #  A. RESOLUTION at fixed Reynolds. N up, nu fixed, L FIXED at 8.0.
    #     (The bootstrap presets scale L with N, which is domain extension,
    #      not refinement. We pin L so this is a genuine grid study.)
    #     If N=40 already resolves the flow, the POD spectrum should not move.
    #
    #  B. REYNOLDS at adequate resolution. nu down, N up enough to stay
    #     resolved. This is what actually raises the dynamical dimension.
    #
    # The decisive number is the numerical rank of the truth trajectory.
    # G1 showed rank ~13 at (N=40, nu=2.5e-3), which is why a 13-mode POD
    # reproduced it to 1e-7 and the genericity test came back null.
    POINT_FILE = SAVE_DIR_X / f"m3d_R_{tag}_gpu{physical_gpu}_interventions_partial.csv"
    CASE_FILE = SAVE_DIR_X / f"m3d_R_{tag}_gpu{physical_gpu}_cases_partial.csv"
    if POINT_FILE.exists():
        ROWS_X = pd.read_csv(POINT_FILE).to_dict("records")
    else:
        ROWS_X = []
    COMPLETED_X = set(r["config"] for r in ROWS_X if "config" in r)

    def flush_shard():
        if ROWS_X:
            pd.DataFrame(ROWS_X).to_csv(POINT_FILE, index=False)

    print("=" * 120, flush=True)
    print(f"[GPU{physical_gpu}] {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[GPU{physical_gpu}] recovered: {sorted(COMPLETED_X)}", flush=True)
    print("=" * 120, flush=True)

    BASE_NU_R = 2.5e-3
    SWEEP_CONFIGS_R = [
        # (label,           N,  nu_mult, do_leakage)
        ("A_N40_nu1",       40,  1.0,    True),
        ("A_N64_nu1",       64,  1.0,    True),
        ("A_N80_nu1",       80,  1.0,    True),
        ("B_N64_nu0.4",     64,  0.4,    True),
        ("B_N80_nu0.16",    80,  0.16,   True),
        ("B_N96_nu0.0625",  96,  0.0625, False),   # spectrum only (memory)
    ]
    TOPO_R = os.environ.get("M3D_R_TOPOLOGY", "skew_tubes")
    N_SNAPS_R = int(os.environ.get("M3D_R_SNAPS", 32))
    LEAK_PROBES_R = int(os.environ.get("M3D_R_PROBES", 12))
    L_FIXED_R = 8.0

    def _redot_R(a1, a2, b1, b2):
        return float((torch.real(torch.vdot(a1.reshape(-1), b1.reshape(-1)))
                      + torch.real(torch.vdot(a2.reshape(-1), b2.reshape(-1))))
                     .detach().cpu())

    @torch.no_grad()
    def pod_spectrum_R(S1, S2):
        """Eigenvalues of the snapshot Gram matrix. Snapshots live on CPU;
        only two are resident on GPU at a time, so N=96 stays in memory."""
        n = len(S1)
        dev = "cuda"
        m1 = S1[0].to(dev).clone()
        m2 = S2[0].to(dev).clone()
        for i in range(1, n):
            m1 += S1[i].to(dev)
            m2 += S2[i].to(dev)
        m1 /= n
        m2 /= n
        G = torch.zeros(n, n, dtype=torch.float64)
        for i in range(n):
            a1 = S1[i].to(dev) - m1
            a2 = S2[i].to(dev) - m2
            for j in range(i, n):
                b1 = S1[j].to(dev) - m1
                b2 = S2[j].to(dev) - m2
                v = _redot_R(a1, a2, b1, b2)
                G[i, j] = v
                G[j, i] = v
                del b1, b2
            del a1, a2
        ev = torch.linalg.eigvalsh(G).flip(0).clamp(min=0.0)
        return ev

    @torch.no_grad()
    def rank_at_R(ev, rel_tol):
        return int((ev > ev[0] * rel_tol).sum())

    @torch.no_grad()
    def modes_for_energy_R(ev, frac):
        tot = float(ev.sum())
        c = 0.0
        for i, e in enumerate(ev):
            c += float(e)
            if c >= frac * tot:
                return i + 1
        return len(ev)

    @torch.no_grad()
    def leakage_concentration_R(V1, V2, dQ1, dQ2, tids, env, nu, n_probe):
        """Randomised range-finding on T = P DPhi_h(V) Q.
        Returns top-3 operator-energy fraction and the realised-response
        fraction recovered by the top-3 directions."""
        base1, base2 = phi_R(V1, V2, env, nu)
        scale = max(pair_norm_R(V1, V2), 1e-30)
        cols = []
        for j in range(n_probe):
            g = torch.Generator(device="cpu").manual_seed(
                zlib.crc32(f"probe{j}".encode()) & 0x7FFFFFFF)
            r1 = torch.complex(torch.randn(V1.shape, generator=g, dtype=torch.float64),
                               torch.randn(V1.shape, generator=g, dtype=torch.float64)
                               ).to(V1.device).to(V1.dtype)
            r2 = torch.complex(torch.randn(V2.shape, generator=g, dtype=torch.float64),
                               torch.randn(V2.shape, generator=g, dtype=torch.float64)
                               ).to(V2.device).to(V2.dtype)
            _, _, r1, r2 = split_PQ_R(r1, r2, tids)
            rn = pair_norm_R(r1, r2)
            if rn < 1e-30:
                continue
            eps = 1e-6 * scale / rn
            p1, p2 = phi_R(V1 + eps * r1, V2 + eps * r2, env, nu)
            cols.append(pack_modes_D((p1 - base1) / eps, (p2 - base2) / eps, tids))
        if len(cols) < 4:
            return float("nan"), float("nan"), float("nan")
        Y = torch.stack(cols, dim=1)
        sv = torch.linalg.svdvals(Y).to(torch.float64)
        e2 = (sv ** 2)
        top3_op = float(e2[:3].sum() / max(float(e2.sum()), 1e-300))

        # realised response of the ACTUAL hidden error, and how much of it
        # the top-3 left singular directions of T account for
        dn = pair_norm_R(dQ1, dQ2)
        if dn < 1e-30:
            return top3_op, float("nan"), float("nan")
        eps = 1e-6 * scale / dn
        p1, p2 = phi_R(V1 + eps * dQ1, V2 + eps * dQ2, env, nu)
        resp = pack_modes_D((p1 - base1) / eps, (p2 - base2) / eps, tids)
        U, _, _ = torch.linalg.svd(Y, full_matrices=False)
        proj3 = U[:, :3] @ (U[:, :3].conj().transpose(0, 1) @ resp)
        frac3 = float((torch.linalg.vector_norm(proj3)
                       / max(float(torch.linalg.vector_norm(resp)), 1e-300)).real)
        return top3_op, frac3, float(sv[0] / max(float(sv[2]), 1e-300))

    def pair_norm_R(a1, a2):
        return float(torch.sqrt(torch.clamp(
            torch.real(torch.vdot(a1.reshape(-1), a1.reshape(-1)))
            + torch.real(torch.vdot(a2.reshape(-1), a2.reshape(-1))),
            min=0.0)).detach().cpu())

    def split_PQ_R(A1, A2, tids):
        P1 = torch.zeros_like(A1)
        P2 = torch.zeros_like(A2)
        P1.reshape(3, -1)[:, tids] = A1.reshape(3, -1)[:, tids]
        P2.reshape(3, -1)[:, tids] = A2.reshape(3, -1)[:, tids]
        return P1, P2, A1 - P1, A2 - P2

    def phi_R(A1, A2, env, nu):
        B1, B2 = A1.clone(), A2.clone()
        for _ in range(SEG_STEPS_R):
            B1 = full_rk4_D(B1, DT_R, nu, env)
            B2 = full_rk4_D(B2, DT_R, nu, env)
        return B1, B2

    TAU0_R = float(globals()["TAU0_D"])
    SEG_TAU_R = 0.15
    SEG_STEPS_R = 20
    DT_R = SEG_TAU_R * TAU0_R / SEG_STEPS_R
    N_SEG_R = 10

    # ------------------------------ main shard loop ------------------------------
    my_cfgs = [c for i, c in enumerate(SWEEP_CONFIGS_R) if i % 2 == physical_gpu]
    print(f"[GPU{physical_gpu}] configs: {[c[0] for c in my_cfgs]}", flush=True)

    for label, Nres, nu_mult, do_leak in my_cfgs:
        if label in COMPLETED_X:
            print(f"[GPU{physical_gpu}] SKIP {label}", flush=True)
            continue
        nu = BASE_NU_R * nu_mult
        print(f"\n[GPU{physical_gpu}] CONFIG {label}  N={Nres} L={L_FIXED_R} "
              f"nu={nu:.4e}", flush=True)
        assert_viscosity_X(nu, torch.float64, torch.complex128)
        env = make_env_D(Nres, L=L_FIXED_R)
        U1, U2 = topology_pair_L2(TOPO_R, env)

        star, _ = initialize_star2_N(U1, U2, env)
        tids = star["proj"]["target_ids"]
        n_target = int(tids.numel())
        dim_state = int(U1.numel() + U2.numel())
        del star

        # ---- truth trajectory with CPU-resident snapshots ----
        every = max(1, (N_SEG_R * SEG_STEPS_R) // N_SNAPS_R)
        t1, t2 = U1.clone(), U2.clone()
        S1, S2 = [t1.detach().cpu().clone()], [t2.detach().cpu().clone()]
        for step in range(N_SEG_R * SEG_STEPS_R):
            t1 = full_rk4_D(t1, DT_R, nu, env)
            t2 = full_rk4_D(t2, DT_R, nu, env)
            if (step + 1) % every == 0:
                S1.append(t1.detach().cpu().clone())
                S2.append(t2.detach().cpu().clone())
        ev = pod_spectrum_R(S1, S2)
        r6, r9, r12 = (rank_at_R(ev, 1e-6), rank_at_R(ev, 1e-9),
                       rank_at_R(ev, 1e-12))
        m99, m999, m9999 = (modes_for_energy_R(ev, 0.99),
                            modes_for_energy_R(ev, 0.999),
                            modes_for_energy_R(ev, 0.9999))
        del S1, S2

        # ---- STAR2 ROM trajectory to the same time ----
        m, _ = initialize_star2_N(U1, U2, env)
        V1, V2 = U1.clone(), U2.clone()
        for seg in range(N_SEG_R):
            for _ in range(SEG_STEPS_R):
                rk4_M(m, SEG_TAU_R / SEG_STEPS_R, env)
            V1, V2 = reconstruct_M(m, env)
            m, _ = initialize_star2_N(V1, V2, env)

        eP = pack_modes_D(V1 - t1, V2 - t2, tids)
        tP = pack_modes_D(t1, t2, tids)
        rom_P_err = float((torch.linalg.vector_norm(eP)
                           / torch.linalg.vector_norm(tP)).real.detach().cpu())
        dP1, dP2, dQ1, dQ2 = split_PQ_R(t1 - V1, t2 - V2, tids)
        reservoir = pair_norm_R(dQ1, dQ2) / max(pair_norm_R(dP1, dP2), 1e-30)

        top3_op = frac3 = sv_ratio = float("nan")
        if do_leak:
            top3_op, frac3, sv_ratio = leakage_concentration_R(
                V1, V2, dQ1, dQ2, tids, env, nu, LEAK_PROBES_R)

        ROWS_X.append({
            "test": "R", "config": label, "N": Nres, "L": L_FIXED_R,
            "nu": nu, "nu_mult": nu_mult, "topology": TOPO_R,
            "dim_state": dim_state, "n_target_modes": n_target,
            "dim_Q_approx": dim_state - n_target,
            "pod_rank_1e6": r6, "pod_rank_1e9": r9, "pod_rank_1e12": r12,
            "modes_for_99pct": m99, "modes_for_99.9pct": m999,
            "modes_for_99.99pct": m9999,
            "star2_P_error_final": rom_P_err,
            "Q_over_P_error_ratio": reservoir,
            "leak_top3_operator_energy": top3_op,
            "leak_top3_realized_fraction": frac3,
            "leak_sv1_over_sv3": sv_ratio,
            "sent_real_dtype": str(globals()["REAL_DTYPE_D"]),
            "sent_NU1_D": float(globals()["NU1_D"]),
            "sent_nu_expected": float(nu),
        })
        print(f"[GPU{physical_gpu}] {label}: dim(Q)~{dim_state-n_target}  "
              f"POD rank(1e-9)={r9}  modes@99.9%={m999}  "
              f"STAR2 P-err={100*rom_P_err:.5f}%  Q/P={reservoir:.2f}  "
              f"top3_op={top3_op:.4f}  top3_realized={frac3:.4f}", flush=True)
        COMPLETED_X.add(label)
        flush_shard()
        del env, U1, U2, t1, t2, V1, V2, m
        torch.cuda.empty_cache()

    flush_shard()
    print(f"[GPU{physical_gpu}] shard complete ✅", flush=True)


def finalize_outputs(tag="r", precision="fp64", jvp="analytic"):
    import numpy as np
    import pandas as pd
    SAVE_DIR = Path("/kaggle/working/M3D")
    files = [SAVE_DIR / f"m3d_R_{tag}_gpu{g}_interventions_partial.csv" for g in (0, 1)]
    have = [f for f in files if f.exists()]
    if not have:
        print("no shard output found")
        return
    R = pd.concat([pd.read_csv(f) for f in have], ignore_index=True)
    R = R.sort_values("config").reset_index(drop=True)
    R.to_csv(SAVE_DIR / f"m3d_R_{tag}_scaling.csv", index=False)

    L = []
    w = L.append
    w("=" * 124)
    w("M3D TEST R — RESOLUTION x REYNOLDS SCALING")
    w("=" * 124)
    w(f"G0 configs completed: {len(R)}/6")
    w("")
    w("Sweep A holds nu fixed and refines the grid at FIXED L (a true grid study).")
    w("Sweep B lowers nu with enough N to stay resolved (raises the dynamical dimension).")
    w("")
    w(f"  {'config':<18}{'N':>5}{'nu':>11}{'dim(Q)':>10}{'rank1e-9':>10}"
      f"{'@99.9%':>9}{'STAR2 err':>12}{'Q/P':>8}{'top3_op':>9}{'top3_real':>11}")
    for _, r in R.iterrows():
        w(f"  {r['config']:<18}{int(r['N']):>5}{r['nu']:>11.3e}"
          f"{int(r['dim_Q_approx']):>10}{int(r['pod_rank_1e9']):>10}"
          f"{int(r['modes_for_99.9pct']):>9}{100*r['star2_P_error_final']:>11.5f}%"
          f"{r['Q_over_P_error_ratio']:>8.2f}"
          f"{r['leak_top3_operator_energy']:>9.4f}{r['leak_top3_realized_fraction']:>11.4f}")

    A = R[R.config.str.startswith("A_")]
    B = R[R.config.str.startswith("B_")]
    w("")
    w("--- R1: does grid refinement alone change the dynamical dimension? ---")
    if len(A) >= 2:
        lo, hi = A.iloc[0], A.iloc[-1]
        w(f"  rank(1e-9): {int(lo['pod_rank_1e9'])} at N={int(lo['N'])}"
          f"  ->  {int(hi['pod_rank_1e9'])} at N={int(hi['N'])}")
        flat = abs(int(hi['pod_rank_1e9']) - int(lo['pod_rank_1e9'])) <= 3
        msg = ("FLAT: N=40 already resolves this flow; grid is not the limitation"
               if flat else "RANK GREW: N=40 was under-resolved")
        w(f"  -> {msg}")
    w("")
    w("--- R2: does lowering nu raise the dynamical dimension? ---")
    if len(B) >= 1 and len(A) >= 1:
        base = A.iloc[0]
        w(f"  baseline (N=40, nu=2.5e-3): rank(1e-9)={int(base['pod_rank_1e9'])}, "
          f"modes@99.9%={int(base['modes_for_99.9pct'])}")
        for _, r in B.iterrows():
            w(f"  {r['config']:<18} rank(1e-9)={int(r['pod_rank_1e9']):>4}  "
              f"modes@99.9%={int(r['modes_for_99.9pct']):>4}  "
              f"({int(r['pod_rank_1e9'])/max(int(base['pod_rank_1e9']),1):.1f}x baseline)")
        grew = int(B.pod_rank_1e9.max()) >= 2 * int(base['pod_rank_1e9'])
        msg = ("PASS: lower nu gives a genuinely richer testbed" if grew
               else "FAIL: these flows stay low-dimensional even at lower nu")
        w(f"  -> {msg}")
    w("")
    w("--- R3: does leakage stay concentrated as dim(Q) grows? ---")
    Rl = R.dropna(subset=["leak_top3_operator_energy"])
    if not Rl.empty:
        for _, r in Rl.iterrows():
            w(f"  {r['config']:<18} dim(Q)~{int(r['dim_Q_approx']):>8}  "
              f"top3 operator energy={r['leak_top3_operator_energy']:.4f}  "
              f"top3 realized={r['leak_top3_realized_fraction']:.4f}")
        held = bool((Rl.leak_top3_operator_energy > 0.80).all())
        msg = ("PASS: concentration survives; low-rank leakage is not an N=40 artifact"
               if held else "FAIL: concentration degrades as dim(Q) grows")
        w(f"  -> {msg}")
    w("")
    w("Outputs:")
    w(f"  {SAVE_DIR / f'm3d_R_{tag}_scaling.csv'}")
    w(f"  {SAVE_DIR / f'm3d_R_{tag}_COMPLETE.txt'}")
    txt = "\n".join(L)
    print(txt)
    open(SAVE_DIR / f"m3d_R_{tag}_COMPLETE.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--precision", choices=["fp32", "fp64"], default="fp64")
    ap.add_argument("--jvp", choices=["fd", "analytic"], default="analytic")
    ap.add_argument("--deriv-rel", type=float, default=1e-5)
    ap.add_argument("--tag", default="r")
    args = ap.parse_args()
    if args.worker:
        if args.gpu not in (0, 1):
            raise SystemExit("--gpu must be 0 or 1")
        worker_main(args.gpu, args.precision, args.jvp, args.deriv_rel, False, args.tag)
    else:
        parent_main(args.precision, args.jvp, args.deriv_rel, False, args.tag)
