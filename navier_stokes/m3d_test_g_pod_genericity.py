#!/usr/bin/env python3
"""
M3D TEST G — GENERICITY: does the hidden-error framework hold for POD-Galerkin?

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


BUILD_ID = "G2-2026-08-30-pod-outofsample"


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
    print("M3D TEST G — DUAL GPU LAUNCHER  [POD-Galerkin genericity]")
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
    assert "POD_TRAIN_FRAC_G" in open(os.path.abspath(__file__)).read(), "wrong file"
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

    POINT_FILE = SAVE_DIR_X / f"m3d_G_{tag}_gpu{physical_gpu}_interventions_partial.csv"
    CASE_FILE = SAVE_DIR_X / f"m3d_G_{tag}_gpu{physical_gpu}_cases_partial.csv"
    SWEEP_FILE = SAVE_DIR_X / f"m3d_G_{tag}_gpu{physical_gpu}_sweep_partial.csv"

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

    # ------------------------------ resume shard ------------------------------
    if POINT_FILE.exists():
        ROWS_X = pd.read_csv(POINT_FILE).to_dict("records")
    else:
        ROWS_X = []
    if CASE_FILE.exists():
        CASE_ROWS_X = pd.read_csv(CASE_FILE).to_dict("records")
    else:
        CASE_ROWS_X = []
    if SWEEP_FILE.exists():
        SWEEP_ROWS_X = pd.read_csv(SWEEP_FILE).to_dict("records")
    else:
        SWEEP_ROWS_X = []
    COMPLETED_X = {str(r["case"]) for r in CASE_ROWS_X}

    print("=" * 120, flush=True)
    print(f"[GPU{physical_gpu}] {torch.cuda.get_device_name(0)} | assigned cases: {CASES_THIS_GPU}", flush=True)
    print(f"[GPU{physical_gpu}] recovered: {sorted(COMPLETED_X)}", flush=True)
    print("=" * 120, flush=True)

    def flush_shard():
        pd.DataFrame(ROWS_X).to_csv(POINT_FILE, index=False)
        if CASE_ROWS_X:
            pd.DataFrame(CASE_ROWS_X).to_csv(CASE_FILE, index=False)

    # ------------------------------ main shard loop ------------------------------
    # ==================================================================
    # ANCHORED POD-GALERKIN ROM
    # ==================================================================
    # The genericity question: is the hidden-error framework a property of
    # STAR2, or of reduced models generally? To answer it, ONLY the ROM
    # changes. The P/Q split, the W estimator and the intervention protocol
    # are byte-identical to Test X.
    #
    #   U_r = U_anchor + sum_i a_i phi_i,      a(0) = 0 at each anchor
    #   da_i/dt = <phi_i, F(U_r)>
    #
    # phi_i are POD modes of truth-trajectory fluctuations, orthonormal under
    # the real Fourier inner product Re sum conj(u) v (the physical L2 product
    # for the Hermitian-symmetric coefficients of a real field).

    def _redot_G(a1, a2, b1, b2):
        return float((torch.real(torch.vdot(a1.reshape(-1), b1.reshape(-1)))
                      + torch.real(torch.vdot(a2.reshape(-1), b2.reshape(-1))))
                     .detach().cpu())

    @torch.no_grad()
    def pod_basis_G(snaps1, snaps2, rank):
        """Economy POD via the Gram matrix (n_snap x n_snap), not a full SVD."""
        n = len(snaps1)
        m1 = sum(snaps1[1:], snaps1[0].clone()) / n
        m2 = sum(snaps2[1:], snaps2[0].clone()) / n
        F1 = [s - m1 for s in snaps1]
        F2 = [s - m2 for s in snaps2]
        G = torch.zeros(n, n, dtype=torch.float64, device=snaps1[0].device)
        for i in range(n):
            for j in range(i, n):
                v = _redot_G(F1[i], F2[i], F1[j], F2[j])
                G[i, j] = v
                G[j, i] = v
        evals, evecs = torch.linalg.eigh(G)
        order = torch.argsort(evals, descending=True)
        evals, evecs = evals[order], evecs[:, order]
        keep = min(rank, int((evals > evals[0] * 1e-12).sum()))
        modes1, modes2, energy = [], [], []
        for k in range(keep):
            c = evecs[:, k]
            p1 = torch.zeros_like(F1[0])
            p2 = torch.zeros_like(F2[0])
            for i in range(n):
                ci = c[i].to(p1.dtype)
                p1 += ci * F1[i]
                p2 += ci * F2[i]
            nrm = math.sqrt(max(_redot_G(p1, p2, p1, p2), 1e-300))
            modes1.append(p1 / nrm)
            modes2.append(p2 / nrm)
            energy.append(float(evals[k].detach().cpu()))
        tot = float(evals.clamp(min=0).sum().detach().cpu())
        captured = sum(energy) / max(tot, 1e-300)
        return modes1, modes2, captured

    @torch.no_grad()
    def pod_rhs_G(anchor1, anchor2, modes1, modes2, a, env, nu):
        U1 = anchor1.clone()
        U2 = anchor2.clone()
        for i, ai in enumerate(a):
            U1 = U1 + float(ai) * modes1[i]
            U2 = U2 + float(ai) * modes2[i]
        F1 = velocity_rhs_D(U1, nu, env)
        F2 = velocity_rhs_D(U2, nu, env)
        return [_redot_G(modes1[i], modes2[i], F1, F2) for i in range(len(a))]

    @torch.no_grad()
    def pod_step_G(anchor1, anchor2, modes1, modes2, a, dt, env, nu):
        k1 = pod_rhs_G(anchor1, anchor2, modes1, modes2, a, env, nu)
        a2 = [a[i] + 0.5 * dt * k1[i] for i in range(len(a))]
        k2 = pod_rhs_G(anchor1, anchor2, modes1, modes2, a2, env, nu)
        a3 = [a[i] + 0.5 * dt * k2[i] for i in range(len(a))]
        k3 = pod_rhs_G(anchor1, anchor2, modes1, modes2, a3, env, nu)
        a4 = [a[i] + dt * k3[i] for i in range(len(a))]
        k4 = pod_rhs_G(anchor1, anchor2, modes1, modes2, a4, env, nu)
        return [a[i] + dt / 6.0 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
                for i in range(len(a))]

    @torch.no_grad()
    def pod_reconstruct_G(anchor1, anchor2, modes1, modes2, a):
        U1 = anchor1.clone()
        U2 = anchor2.clone()
        for i, ai in enumerate(a):
            U1 = U1 + float(ai) * modes1[i]
            U2 = U2 + float(ai) * modes2[i]
        return U1, U2

    @torch.no_grad()
    def pod_forecast_G(S1, S2, modes1, modes2, env, nu):
        """Same shape as star2_forecast_X: anchor, integrate, reanchor between."""
        A1, A2 = S1.clone(), S2.clone()
        for seg in range(FORECAST_SEGMENTS_X):
            a = [0.0] * len(modes1)
            for _ in range(SEGMENT_STEPS_X):
                a = pod_step_G(A1, A2, modes1, modes2, a, DT_PHYS_X, env, nu)
            A1, A2 = pod_reconstruct_G(A1, A2, modes1, modes2, a)
        return A1, A2

    # ------------------------------ main shard loop ------------------------------
    for topology_X, nu_mult_X in CASES_THIS_GPU:
        nu_X = BASE_NU_X * nu_mult_X
        case_X = f"{topology_X}__nu{nu_mult_X:g}"
        if case_X in COMPLETED_X:
            print(f"[GPU{physical_gpu}] SKIP {case_X}", flush=True)
            continue
        print(f"\n[GPU{physical_gpu}] CASE {case_X}", flush=True)

        assert_viscosity_X(nu_X, RDT_X, CDT_X)
        ENV32_X = make_env_D(40, L=8.0)
        U10_X, U20_X = topology_pair_L2(topology_X, ENV32_X)

        # P/Q split taken from the SAME projector STAR2 uses, so the resolved
        # variable set is identical across the two ROMs.
        star_ref, _ = initialize_star2_N(U10_X, U20_X, ENV32_X)
        target_ids_X = star_ref["proj"]["target_ids"]
        del star_ref

        # ---- snapshots for POD, from the truth trajectory ----
        train_steps = max(POD_SNAPSHOTS_G,
                          int(round(POD_TRAIN_FRAC_G
                                    * N_BASELINE_SEGMENTS_X * SEGMENT_STEPS_X)))
        snap_every = max(1, train_steps // POD_SNAPSHOTS_G)
        t1, t2 = U10_X.clone(), U20_X.clone()
        S1, S2 = [t1.clone()], [t2.clone()]
        for step in range(train_steps):
            t1 = full_rk4_D(t1, DT_PHYS_X, nu_X, ENV32_X)
            t2 = full_rk4_D(t2, DT_PHYS_X, nu_X, ENV32_X)
            if (step + 1) % snap_every == 0:
                S1.append(t1.clone())
                S2.append(t2.clone())
        MODES1, MODES2, POD_CAPTURED = pod_basis_G(S1, S2, POD_RANK_G)
        del S1, S2, t1, t2
        print(f"[GPU{physical_gpu}] POD rank={len(MODES1)} "
              f"(requested {POD_RANK_G}) trained on first {train_steps} steps "
              f"= tau {train_steps*DTAU_X:.3f}; "
              f"snapshot-energy captured={100*POD_CAPTURED:.4f}%", flush=True)

        # ---- baseline POD-Galerkin trajectory + truth ----
        truth1_X, truth2_X = U10_X.clone(), U20_X.clone()
        V1_X, V2_X = U10_X.clone(), U20_X.clone()
        MODEL_X = [(V1_X.clone(), V2_X.clone())]
        TRUTH_X = [(truth1_X.clone(), truth2_X.clone())]
        for seg in range(1, N_BASELINE_SEGMENTS_X + 1):
            a = [0.0] * len(MODES1)
            for _ in range(SEGMENT_STEPS_X):
                truth1_X = full_rk4_D(truth1_X, DT_PHYS_X, nu_X, ENV32_X)
                truth2_X = full_rk4_D(truth2_X, DT_PHYS_X, nu_X, ENV32_X)
                a = pod_step_G(V1_X, V2_X, MODES1, MODES2, a, DT_PHYS_X,
                               ENV32_X, nu_X)
            V1_X, V2_X = pod_reconstruct_G(V1_X, V2_X, MODES1, MODES2, a)
            MODEL_X.append((V1_X.clone(), V2_X.clone()))
            TRUTH_X.append((truth1_X.clone(), truth2_X.clone()))

        # ---- W estimator, IDENTICAL to Test X ----
        assert_viscosity_X(nu_X, torch.float64, torch.complex128)
        ENV64_X = make_env_D(40, L=8.0)
        MODEL64_X = [(a.to(torch.complex128), b.to(torch.complex128))
                     for a, b in MODEL_X]
        EHAT1_X = torch.zeros_like(MODEL64_X[0][0])
        EHAT2_X = torch.zeros_like(MODEL64_X[0][1])
        EHAT_TRAJ_X = [(EHAT1_X.clone(), EHAT2_X.clone())]
        max_needed = int(round(max(INTERVENTION_TAUS_X) / SEGMENT_TAU_X))
        for k in range(max_needed):
            V1, V2 = MODEL64_X[k]
            VN1, VN2 = MODEL64_X[k + 1]
            PHI1, PHI2 = phi_X(V1, V2, ENV64_X, nu_X)
            ETA1, ETA2 = PHI1 - VN1, PHI2 - VN2
            if pair_norm_X(EHAT1_X, EHAT2_X) < 1e-30:
                EHAT1_X, EHAT2_X = ETA1.clone(), ETA2.clone()
            else:
                J1, J2, H1, H2 = map_jet_X(V1, V2, EHAT1_X, EHAT2_X,
                                           PHI1, PHI2, ENV64_X, nu_X)
                EHAT1_X = ETA1 + J1 + 0.5 * H1
                EHAT2_X = ETA2 + J2 + 0.5 * H2
            EHAT_TRAJ_X.append((EHAT1_X.clone(), EHAT2_X.clone()))

        # ---- interventions, IDENTICAL protocol to Test X ----
        assert_viscosity_X(nu_X, RDT_X, CDT_X)
        case_window_rows = []
        for tau_int_X in INTERVENTION_TAUS_X:
            idx = int(round(tau_int_X / SEGMENT_TAU_X))
            V1, V2 = MODEL_X[idx]
            T1, T2 = TRUTH_X[idx]
            TF1, TF2 = TRUTH_X[idx + FORECAST_SEGMENTS_X]
            EH1_64, EH2_64 = EHAT_TRAJ_X[idx]
            _, _, EHQ1_64, EHQ2_64 = split_PQ_X(EH1_64, EH2_64, target_ids_X)
            EHQ1, EHQ2 = EHQ1_64.to(CDT_X), EHQ2_64.to(CDT_X)
            _, _, AQ1, AQ2 = split_PQ_X(T1 - V1, T2 - V2, target_ids_X)

            xb = pack_modes_D(V1, V2, target_ids_X)
            xe = pack_modes_D(V1 + EHQ1, V2 + EHQ2, target_ids_X)
            start_mismatch = vec_norm_X(xe - xb) / (vec_norm_X(xb) + 1e-30)

            FB1, FB2 = pod_forecast_G(V1, V2, MODES1, MODES2, ENV32_X, nu_X)
            FE1, FE2 = pod_forecast_G(V1 + EHQ1, V2 + EHQ2, MODES1, MODES2,
                                      ENV32_X, nu_X)
            FN1, FN2 = pod_forecast_G(V1 - EHQ1, V2 - EHQ2, MODES1, MODES2,
                                      ENV32_X, nu_X)
            FO1, FO2 = pod_forecast_G(V1 + AQ1, V2 + AQ2, MODES1, MODES2,
                                      ENV32_X, nu_X)

            be = target_error_X(FB1, FB2, TF1, TF2, target_ids_X)
            ee = target_error_X(FE1, FE2, TF1, TF2, target_ids_X)
            ne = target_error_X(FN1, FN2, TF1, TF2, target_ids_X)
            oe = target_error_X(FO1, FO2, TF1, TF2, target_ids_X)

            # Q-residual of the truth-free estimate against the true hidden error
            q_res = (pair_norm_X(EHQ1 - AQ1.to(CDT_X), EHQ2 - AQ2.to(CDT_X))
                     / max(pair_norm_X(AQ1, AQ2), 1e-30))

            row = {
                "test": "G", "case": case_X, "topology": topology_X,
                "nu_mult": nu_mult_X, "tau_intervention": tau_int_X,
                "rom": "pod_galerkin", "pod_rank": len(MODES1),
                "pod_energy_captured": POD_CAPTURED,
                "pod_train_frac": POD_TRAIN_FRAC_G,
                "start_target_mismatch": start_mismatch,
                "baseline_P_error": be, "est_P_error": ee,
                "neg_P_error": ne, "oracle_P_error": oe,
                "gain_est": be / max(ee, 1e-30),
                "gain_neg": be / max(ne, 1e-30),
                "gain_oracle": be / max(oe, 1e-30),
                "est_beats_baseline": bool(ee < be),
                "est_beats_negative": bool(ee < ne),
                "oracle_beats_baseline": bool(oe < be),
                "hidden_Q_residual": q_res,
                "sent_precision": precision, "sent_jvp": jvp,
                "sent_real_dtype": str(globals()["REAL_DTYPE_D"]),
                "sent_NU1_D": float(globals()["NU1_D"]),
                "sent_nu_expected": float(nu_X),
            }
            ROWS_X.append(row)
            case_window_rows.append(row)
            print(f"[GPU{physical_gpu}] G {case_X} tau={tau_int_X:.2f}: "
                  f"BASE={100*be:.5f}% +EST={100*ee:.5f}% -EST={100*ne:.5f}% "
                  f"ORACLE={100*oe:.5f}% gain={be/max(ee,1e-30):.3f}x "
                  f"Qres={q_res:.3e}", flush=True)

        CASE_ROWS_X.append({
            "case": case_X, "topology": topology_X, "nu_mult": nu_mult_X,
            "rom": "pod_galerkin", "pod_rank": len(MODES1),
            "pod_energy_captured": POD_CAPTURED,
            "median_gain_est": float(np.median(
                [r["gain_est"] for r in case_window_rows])),
            "median_hidden_Q_residual": float(np.median(
                [r["hidden_Q_residual"] for r in case_window_rows])),
        })
        flush_shard()
        del MODEL_X, TRUTH_X, MODEL64_X, EHAT_TRAJ_X, MODES1, MODES2

    flush_shard()
    print(f"[GPU{physical_gpu}] shard complete ✅", flush=True)


def finalize_outputs(tag="g", precision="fp64", jvp="analytic"):
    import numpy as np
    import pandas as pd

    SAVE_DIR = Path("/kaggle/working/M3D")
    pfiles = [SAVE_DIR / f"m3d_G_{tag}_gpu{g}_interventions_partial.csv" for g in (0, 1)]
    cfiles = [SAVE_DIR / f"m3d_G_{tag}_gpu{g}_cases_partial.csv" for g in (0, 1)]
    have = [f for f in pfiles if f.exists()]
    if not have:
        print("no shard output found")
        return
    R = pd.concat([pd.read_csv(f) for f in have], ignore_index=True)
    C = (pd.concat([pd.read_csv(f) for f in cfiles if f.exists()], ignore_index=True)
         if any(f.exists() for f in cfiles) else pd.DataFrame())
    R.to_csv(SAVE_DIR / f"m3d_G_{tag}_interventions.csv", index=False)
    if not C.empty:
        C.to_csv(SAVE_DIR / f"m3d_G_{tag}_case_summary.csv", index=False)

    L = []
    w = L.append
    w("=" * 112)
    w("M3D TEST G — GENERICITY: HIDDEN-ERROR FRAMEWORK ON A POD-GALERKIN ROM")
    w("=" * 112)
    w(f"G0 complete dataset: {'PASS' if len(R) == 36 else 'FAIL'}  ({len(R)}/36 windows)")
    if R.empty:
        print("\n".join(L)); return
    nu_ok = bool(np.allclose(R.sent_NU1_D, R.sent_nu_expected, atol=1e-14))
    want = "float64" if precision == "fp64" else "float32"
    dt_ok = bool(R.sent_real_dtype.astype(str).str.contains(want).all())
    w(f"G1 sentinels consistent AND correct: {'PASS' if (nu_ok and dt_ok) else 'FAIL'}")
    w(f"    POD rank {int(R.pod_rank.median())}, "
      f"median snapshot energy captured {100*R.pod_energy_captured.median():.4f}%")

    # ---- N0: is there any ROM error to correct? ----
    floor = float(os.environ.get("M3D_G_NULL_FLOOR", 1e-4))
    per_case_base = R.groupby("case").baseline_P_error.median()
    n_live = int((per_case_base > floor).sum())
    w("")
    w("--- N0: is this test meaningful? ---")
    w(f"  median baseline ROM error = {R.baseline_P_error.median():.3e}"
      f"   (floor for a live test: {floor:.0e})")
    w(f"  cases with baseline error above floor: {n_live}/12")
    if n_live < 10:
        w("  N0 -> NULL TEST. The POD ROM reproduces the truth to near machine")
        w("        precision, so there is no accumulated hidden error to correct.")
        w("        The gates below are NOT evaluated: a gain of ~1.0 here means")
        w("        'nothing to fix', not 'the framework failed'.")
        w("        Lower M3D_G_POD_RANK or M3D_G_TRAIN_FRAC and rerun.")
        txt = "\n".join(L)
        print(txt)
        open(SAVE_DIR / f"m3d_G_{tag}_COMPLETE.txt", "w").write(txt + "\n")
        return
    w(f"  N0 -> LIVE ({n_live}/12 cases have correctable error)")

    w("")
    w("--- the six Test-X gates, re-run against a completely different ROM ---")
    mm = float(R.start_target_mismatch.max())
    w(f"  numerics / equal-P-start        max mismatch {mm:.3e}"
      f"   -> {'PASS' if mm < 1e-6 else 'FAIL'}")
    n_or = int(R.oracle_beats_baseline.astype(bool).sum())
    w(f"  oracle hidden-causality         {n_or}/36  -> {'PASS' if n_or == 36 else 'FAIL'}")
    n_est = int(R.est_beats_baseline.astype(bool).sum())
    w(f"  truth-free intervention         {n_est}/36  -> {'PASS' if n_est == 36 else 'FAIL'}")
    n_dir = int(R.est_beats_negative.astype(bool).sum())
    w(f"  directional +Q vs -Q            {n_dir}/36  -> {'PASS' if n_dir == 36 else 'FAIL'}")
    cap = (R.baseline_P_error - R.est_P_error) / (
        R.baseline_P_error - R.oracle_P_error).replace(0, np.nan)
    w(f"  oracle-benefit capture          median {cap.median():.4f}")
    percase = R.groupby("case").gain_est.median()
    n_case = int((percase > 1.0).sum())
    w(f"  cross-case generalization       {n_case}/12 cases"
      f"  -> {'PASS' if n_case == 12 else 'FAIL'}")

    w("")
    w("--- headline, POD-Galerkin vs STAR2 ---")
    w(f"  median EST-Q gain            {R.gain_est.median():.6f}x    (STAR2: 1.650998x)")
    w(f"  p10 EST-Q gain               {R.gain_est.quantile(0.10):.6f}x    (STAR2: 1.299859x)")
    w(f"  median wrong-sign gain       {R.gain_neg.median():.6f}x    (STAR2: 0.692838x)")
    w(f"  median oracle gain           {R.gain_oracle.median():.6f}x")
    w(f"  median resolved reduction    {100*(1-1/R.gain_est.median()):.3f}%   (STAR2: 39.417%)")
    w(f"  median hidden-Q residual     {R.hidden_Q_residual.median():.3e}")
    w("")
    w("  VERDICT: the hidden-error framework is " +
      ("ROM-GENERIC — it reproduces on POD-Galerkin, so the result is about"
       if (n_est == 36 and n_dir == 36) else
       "NOT reproduced on POD-Galerkin — the result may be specific to"))
    w("           " + ("reduced models, not about STAR2."
                       if (n_est == 36 and n_dir == 36) else "STAR2's construction."))

    w("")
    w(f"  {'case':<26}{'median gain':>13}{'median Qres':>14}")
    for c, g in percase.sort_values(ascending=False).items():
        qr = R[R.case == c].hidden_Q_residual.median()
        w(f"  {c:<26}{g:>13.4f}{qr:>14.3e}")

    w("")
    w("Outputs:")
    for f in (f"m3d_G_{tag}_interventions.csv", f"m3d_G_{tag}_case_summary.csv",
              f"m3d_G_{tag}_COMPLETE.txt"):
        w(f"  {SAVE_DIR / f}")
    txt = "\n".join(L)
    print(txt)
    open(SAVE_DIR / f"m3d_G_{tag}_COMPLETE.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--precision", choices=["fp32", "fp64"], default="fp64")
    ap.add_argument("--jvp", choices=["fd", "analytic"], default="analytic")
    ap.add_argument("--deriv-rel", type=float, default=1e-5)
    ap.add_argument("--tag", default="g")
    args = ap.parse_args()
    if args.worker:
        if args.gpu not in (0, 1):
            raise SystemExit("--gpu must be 0 or 1")
        worker_main(args.gpu, args.precision, args.jvp, args.deriv_rel,
                    False, args.tag)
    else:
        parent_main(args.precision, args.jvp, args.deriv_rel, False, args.tag)
