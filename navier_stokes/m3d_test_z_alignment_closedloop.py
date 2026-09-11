#!/usr/bin/env python3
"""
M3D TEST Z — alpha* alignment prediction (Z1) + closed-loop correction (Z2)

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


BUILD_ID = "Z1-2026-08-30-alignment-closedloop"


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
    print("M3D TEST Z — DUAL GPU LAUNCHER  [Z1 alignment + Z2 closed-loop]")
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
    assert "ALPHA_PRED_GRID" in open(os.path.abspath(__file__)).read(), "wrong file"
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
    N_CLOSED_LOOP_SEGMENTS_Z = int(os.environ.get("M3D_Z_CL_SEGMENTS", 12))
    CL_ALPHA_Z = float(os.environ.get("M3D_Z_CL_ALPHA", 1.0))
    DERIV_REL_X = float(deriv_rel)

    POINT_FILE = SAVE_DIR_X / f"m3d_Z_{tag}_gpu{physical_gpu}_interventions_partial.csv"
    CASE_FILE = SAVE_DIR_X / f"m3d_Z_{tag}_gpu{physical_gpu}_cases_partial.csv"
    SWEEP_FILE = SAVE_DIR_X / f"m3d_Z_{tag}_gpu{physical_gpu}_sweep_partial.csv"

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
    for topology_X, nu_mult_X in CASES_THIS_GPU:
        nu_X = BASE_NU_X * nu_mult_X
        case_X = f"{topology_X}__nu{nu_mult_X:g}"
        if case_X in COMPLETED_X:
            print(f"[GPU{physical_gpu}] SKIP {case_X}", flush=True)
            continue

        print(f"\n[GPU{physical_gpu}] CASE {case_X}", flush=True)

        # baseline + truth trajectory at the selected precision
        assert_viscosity_X(nu_X, RDT_X, CDT_X)
        ENV32_X = make_env_D(40, L=8.0)
        U10_X, U20_X = topology_pair_L2(topology_X, ENV32_X)
        truth1_X, truth2_X = U10_X.clone(), U20_X.clone()
        model_X, init_diag_X = initialize_star2_N(U10_X, U20_X, ENV32_X)
        target_ids_X = model_X["proj"]["target_ids"]
        V01_X, V02_X = reconstruct_M(model_X, ENV32_X)
        MODEL_X = [(V01_X.clone(), V02_X.clone())]
        TRUTH_X = [(truth1_X.clone(), truth2_X.clone())]
        print(f"[GPU{physical_gpu}] initial RAf={init_diag_X['RAf_ratio']:.3e}", flush=True)

        for seg in range(1, N_BASELINE_SEGMENTS_X + 1):
            for _ in range(SEGMENT_STEPS_X):
                truth1_X = full_rk4_D(truth1_X, DT_PHYS_X, nu_X, ENV32_X)
                truth2_X = full_rk4_D(truth2_X, DT_PHYS_X, nu_X, ENV32_X)
                rk4_M(model_X, DTAU_X, ENV32_X)
            model_X, diag_X, jump_X, build_X = reanchor_star2_N(model_X, ENV32_X)
            V1_X, V2_X = reconstruct_M(model_X, ENV32_X)
            MODEL_X.append((V1_X.clone(), V2_X.clone()))
            TRUTH_X.append((truth1_X.clone(), truth2_X.clone()))

        # W second-order estimator along baseline in fp64 through tau=1.50
        assert_viscosity_X(nu_X, torch.float64, torch.complex128)
        ENV64_X = make_env_D(40, L=8.0)
        MODEL64_X = [(a.to(torch.complex128), b.to(torch.complex128)) for a, b in MODEL_X]
        EHAT1_X = torch.zeros_like(MODEL64_X[0][0])
        EHAT2_X = torch.zeros_like(MODEL64_X[0][1])
        EHAT_TRAJ_X = [(EHAT1_X.clone(), EHAT2_X.clone())]
        max_needed_segments_X = int(round(max(INTERVENTION_TAUS_X) / SEGMENT_TAU_X))

        for k in range(max_needed_segments_X):
            V1, V2 = MODEL64_X[k]
            VN1, VN2 = MODEL64_X[k + 1]
            PHI1, PHI2 = phi_X(V1, V2, ENV64_X, nu_X)
            ETA1, ETA2 = PHI1 - VN1, PHI2 - VN2
            if pair_norm_X(EHAT1_X, EHAT2_X) < 1e-30:
                NEW1, NEW2 = ETA1.clone(), ETA2.clone()
            else:
                J1, J2, H1, H2 = map_jet_X(V1, V2, EHAT1_X, EHAT2_X, PHI1, PHI2, ENV64_X, nu_X)
                NEW1 = ETA1 + J1 + 0.5 * H1
                NEW2 = ETA2 + J2 + 0.5 * H2
            EHAT1_X, EHAT2_X = NEW1, NEW2
            EHAT_TRAJ_X.append((EHAT1_X.clone(), EHAT2_X.clone()))

        # ==================================================================
        # Z1 — alpha* PREDICTED FROM ERROR ALIGNMENT
        # ==================================================================
        # Theorem C: correcting with alpha*q leaves  ||b_PP + (1-alpha) b_QP||,
        # minimised at   alpha* = 1 + <b_PP, b_QP> / ||b_QP||^2.
        # Everything needed is already available at each window:
        #   b = P[Forecast(V) - Truth]          baseline future resolved error
        #   d = P[Forecast(V+q) - Forecast(V)]  effect of the full oracle correction
        #   b_QP = -d          (the piece the alpha=1 correction cancels)
        #   b_PP =  b + d      (what survives a full correction)
        # A second forecast at alpha=2 gives the curvature term, so the
        # quadratic prediction can be compared against the linear one.
        ALPHA_PRED_GRID = np.linspace(0.0, 8.0, 1601)

        assert_viscosity_X(nu_X, RDT_X, CDT_X)
        ENV32_X = make_env_D(40, L=8.0)
        case_window_rows = []

        for tau_int_X in INTERVENTION_TAUS_X:
            idx = int(round(tau_int_X / SEGMENT_TAU_X))
            idx_end = idx + FORECAST_SEGMENTS_X
            V1, V2 = MODEL_X[idx]
            T1, T2 = TRUTH_X[idx]
            TF1, TF2 = TRUTH_X[idx_end]

            ACT1, ACT2 = T1 - V1, T2 - V2
            _, _, AQ1, AQ2 = split_PQ_X(ACT1, ACT2, target_ids_X)

            FB1, FB2 = star2_forecast_X(V1, V2, ENV32_X)                 # alpha = 0
            FO1, FO2 = star2_forecast_X(V1 + AQ1, V2 + AQ2, ENV32_X)     # alpha = 1
            F21, F22 = star2_forecast_X(V1 + 2*AQ1, V2 + 2*AQ2, ENV32_X) # alpha = 2

            b  = pack_modes_D(FB1 - TF1, FB2 - TF2, target_ids_X)
            d1 = pack_modes_D(FO1 - FB1, FO2 - FB2, target_ids_X)        # Delta(1)
            d2 = pack_modes_D(F21 - FB1, F22 - FB2, target_ids_X)        # Delta(2)

            def _re_dot(u, v):
                return float(torch.real(torch.vdot(u, v)).detach().cpu())

            b_QP = -d1
            b_PP = b + d1
            nQP2 = _re_dot(b_QP, b_QP)
            alpha_lin = 1.0 + _re_dot(b_PP, b_QP) / max(nQP2, 1e-300)
            cos_theta = (_re_dot(b_PP, b_QP)
                         / max(math.sqrt(_re_dot(b_PP, b_PP) * nQP2), 1e-300))

            # curvature split:  Delta(a) = a*g1 + a^2/2 * g2
            g1 = 2.0 * d1 - 0.5 * d2
            g2 = d2 - 2.0 * d1
            errs = [vec_norm_X(b + float(a) * g1 + 0.5 * float(a) ** 2 * g2)
                    for a in ALPHA_PRED_GRID]
            alpha_quad = float(ALPHA_PRED_GRID[int(np.argmin(errs))])

            base_err = target_error_X(FB1, FB2, TF1, TF2, target_ids_X)
            ora_err = target_error_X(FO1, FO2, TF1, TF2, target_ids_X)

            row = {
                "test": "Z1", "case": case_X, "topology": topology_X,
                "nu_mult": nu_mult_X, "tau_intervention": tau_int_X,
                "baseline_P_error": base_err, "oracle_P_error": ora_err,
                "norm_b": vec_norm_X(b), "norm_bPP": math.sqrt(_re_dot(b_PP, b_PP)),
                "norm_bQP": math.sqrt(nQP2),
                "ratio_bPP_over_bQP": math.sqrt(_re_dot(b_PP, b_PP) / max(nQP2, 1e-300)),
                "cos_theta_bPP_bQP": cos_theta,
                "alpha_star_linear": alpha_lin,
                "alpha_star_quadratic": alpha_quad,
                "curvature_ratio": vec_norm_X(g2) / max(vec_norm_X(g1), 1e-300),
                "sent_precision": precision, "sent_jvp": jvp,
                "sent_real_dtype": str(globals()["REAL_DTYPE_D"]),
                "sent_NU1_D": float(globals()["NU1_D"]),
                "sent_nu_expected": float(nu_X),
            }
            ROWS_X.append(row)
            case_window_rows.append(row)
            print(f"[GPU{physical_gpu}] Z1 {case_X} tau={tau_int_X:.2f}: "
                  f"cos={cos_theta:+.4f}  |bPP|/|bQP|={row['ratio_bPP_over_bQP']:.3f}  "
                  f"alpha*_lin={alpha_lin:.3f}  alpha*_quad={alpha_quad:.3f}", flush=True)

        # ==================================================================
        # Z2 — CLOSED-LOOP CORRECTION
        # ==================================================================
        # Every prior intervention was open-loop and single-shot. Here the
        # correction is applied at EVERY segment, and the W recurrence is fed
        # the corrected trajectory, so the estimator sees its own influence.
        # After a correction removes Q e_hat, the retained estimate is P e_hat.
        assert_viscosity_X(nu_X, torch.float64, torch.complex128)
        ENV64_Z = make_env_D(40, L=8.0)

        m_cl, _ = initialize_star2_N(U10_X.to(torch.complex128),
                                     U20_X.to(torch.complex128), ENV64_Z)
        tids_cl = m_cl["proj"]["target_ids"]
        EH1 = torch.zeros_like(MODEL64_X[0][0])
        EH2 = torch.zeros_like(MODEL64_X[0][1])
        cl_rows, blew_up = [], False

        for k in range(N_CLOSED_LOOP_SEGMENTS_Z):
            Vc1, Vc2 = reconstruct_M(m_cl, ENV64_Z)
            for _ in range(SEGMENT_STEPS_X):
                rk4_M(m_cl, DTAU_X, ENV64_Z)
            Vn1, Vn2 = reconstruct_M(m_cl, ENV64_Z)

            PH1, PH2 = phi_X(Vc1, Vc2, ENV64_Z, nu_X)
            ET1, ET2 = PH1 - Vn1, PH2 - Vn2
            if pair_norm_X(EH1, EH2) < 1e-30:
                EH1, EH2 = ET1.clone(), ET2.clone()
            else:
                J1, J2, H1, H2 = map_jet_X(Vc1, Vc2, EH1, EH2, PH1, PH2, ENV64_Z, nu_X)
                EH1, EH2 = ET1 + J1 + 0.5 * H1, ET2 + J2 + 0.5 * H2

            _, _, EHQ1, EHQ2 = split_PQ_X(EH1, EH2, tids_cl)
            Vk1, Vk2 = Vn1 + CL_ALPHA_Z * EHQ1, Vn2 + CL_ALPHA_Z * EHQ2
            # correction consumed the hidden component; retain only P e_hat
            EH1, EH2, _, _ = split_PQ_X(EH1, EH2, tids_cl)

            m_cl, _ = initialize_star2_N(Vk1, Vk2, ENV64_Z)

            Tt1, Tt2 = TRUTH_X[k + 1]
            Tt1, Tt2 = Tt1.to(torch.complex128), Tt2.to(torch.complex128)
            Vo1, Vo2 = MODEL_X[k + 1]
            Vo1, Vo2 = Vo1.to(torch.complex128), Vo2.to(torch.complex128)
            e_cl = target_error_X(Vk1, Vk2, Tt1, Tt2, tids_cl)
            e_ol = target_error_X(Vo1, Vo2, Tt1, Tt2, tids_cl)
            if (not math.isfinite(e_cl)) or e_cl > 1.0:
                blew_up = True
            cl_rows.append({
                "test": "Z2", "case": case_X, "topology": topology_X,
                "nu_mult": nu_mult_X, "segment": k + 1,
                "tau": (k + 1) * SEGMENT_TAU_X,
                "closed_loop_P_error": e_cl, "open_loop_P_error": e_ol,
                "improvement": e_ol / max(e_cl, 1e-30),
                "cl_alpha": CL_ALPHA_Z, "blew_up": blew_up,
                "sent_precision": precision, "sent_jvp": jvp,
            })
            if blew_up:
                print(f"[GPU{physical_gpu}] Z2 {case_X}: DIVERGED at segment {k+1}",
                      flush=True)
                break

        ROWS_X.extend(cl_rows)
        if cl_rows:
            last = cl_rows[-1]
            print(f"[GPU{physical_gpu}] Z2 {case_X}: tau={last['tau']:.2f} "
                  f"closed={last['closed_loop_P_error']*100:.4f}% "
                  f"open={last['open_loop_P_error']*100:.4f}% "
                  f"improve={last['improvement']:.3f}x", flush=True)

        CASE_ROWS_X.append({
            "case": case_X, "topology": topology_X, "nu_mult": nu_mult_X,
            "median_alpha_star_linear": float(np.median(
                [r["alpha_star_linear"] for r in case_window_rows])),
            "median_alpha_star_quadratic": float(np.median(
                [r["alpha_star_quadratic"] for r in case_window_rows])),
            "median_cos_theta": float(np.median(
                [r["cos_theta_bPP_bQP"] for r in case_window_rows])),
            "closed_loop_final_improvement": (cl_rows[-1]["improvement"]
                                              if cl_rows else float("nan")),
            "closed_loop_diverged": blew_up,
        })
        flush_shard()
        del MODEL_X, TRUTH_X, MODEL64_X, EHAT_TRAJ_X, model_X, m_cl

    flush_shard()
    print(f"[GPU{physical_gpu}] shard complete ✅", flush=True)


def finalize_outputs(tag="z", precision="fp64", jvp="analytic"):
    import numpy as np
    import pandas as pd

    SAVE_DIR = Path("/kaggle/working/M3D")
    pfiles = [SAVE_DIR / f"m3d_Z_{tag}_gpu{g}_interventions_partial.csv" for g in (0, 1)]
    cfiles = [SAVE_DIR / f"m3d_Z_{tag}_gpu{g}_cases_partial.csv" for g in (0, 1)]
    have = [f for f in pfiles if f.exists()]
    if not have:
        print("no shard output found")
        return
    R = pd.concat([pd.read_csv(f) for f in have], ignore_index=True)
    C = (pd.concat([pd.read_csv(f) for f in cfiles if f.exists()], ignore_index=True)
         if any(f.exists() for f in cfiles) else pd.DataFrame())
    Z1 = R[R.test == "Z1"].copy()
    Z2 = R[R.test == "Z2"].copy()
    Z1.to_csv(SAVE_DIR / f"m3d_Z_{tag}_alignment.csv", index=False)
    Z2.to_csv(SAVE_DIR / f"m3d_Z_{tag}_closedloop.csv", index=False)
    if not C.empty:
        C.to_csv(SAVE_DIR / f"m3d_Z_{tag}_case_summary.csv", index=False)

    L = []
    w = L.append
    w("=" * 110)
    w("M3D TEST Z — ALPHA* ALIGNMENT PREDICTION (Z1) + CLOSED-LOOP CORRECTION (Z2)")
    w("=" * 110)

    g0 = len(Z1) == 36
    w(f"G0 complete Z1 dataset: {'PASS' if g0 else 'FAIL'}  ({len(Z1)}/36 windows)")
    if not Z1.empty:
        nu_ok = bool(np.allclose(Z1.sent_NU1_D, Z1.sent_nu_expected, atol=1e-14))
        want = "float64" if precision == "fp64" else "float32"
        dt_ok = bool(Z1.sent_real_dtype.astype(str).str.contains(want).all())
        w(f"G1 sentinels consistent AND correct: {'PASS' if (nu_ok and dt_ok) else 'FAIL'}")

        w("")
        w("--- Z1: does error alignment predict the optimal correction amplitude? ---")
        w(f"  median cos(b_PP, b_QP)     = {Z1.cos_theta_bPP_bQP.median():+.4f}")
        w(f"  median |b_PP| / |b_QP|     = {Z1.ratio_bPP_over_bQP.median():.4f}")
        w(f"  median alpha* (linear)     = {Z1.alpha_star_linear.median():.4f}")
        w(f"  median alpha* (quadratic)  = {Z1.alpha_star_quadratic.median():.4f}")
        w(f"  median curvature |g2|/|g1| = {Z1.curvature_ratio.median():.4f}")
        w("")
        n_pos = int((Z1.cos_theta_bPP_bQP > 0).sum())
        w("  A1  cos(b_PP,b_QP) > 0 in >= 30/36 windows  "
          "-> overcorrection helps because the two error parts are aligned")
        w(f"      {n_pos}/{len(Z1)}  -> {'PASS' if n_pos >= 30 else 'FAIL'}")
        aq = float(Z1.alpha_star_quadratic.median())
        w("  A2  median predicted alpha* in [1.5, 3.0]  "
          "(X-v3 measured argmax was 2.00)")
        w(f"      {aq:.4f}  -> {'PASS' if 1.5 <= aq <= 3.0 else 'FAIL'}")
        w("  A3  DESCRIPTIVE per-window predicted alpha*:")
        for lo, hi in [(0, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 99)]:
            n = int(((Z1.alpha_star_quadratic >= lo)
                     & (Z1.alpha_star_quadratic < hi)).sum())
            w(f"      alpha* in [{lo:>4.1f},{hi:>5.1f}) : {n:2d} windows")

    w("")
    w("--- Z2: closed-loop correction applied at every segment ---")
    if Z2.empty:
        w("  no closed-loop rows")
    else:
        div = Z2.groupby("case").blew_up.max().astype(bool)
        w(f"  C1 no divergence: {int((~div).sum())}/{len(div)} cases stable"
          f"  -> {'PASS' if (~div).all() else 'FAIL'}")
        fin = Z2.sort_values("segment").groupby("case").tail(1)
        n_better = int((fin.improvement > 1.0).sum())
        w(f"  C2 closed-loop beats open-loop at final time: {n_better}/{len(fin)}"
          f"  -> {'PASS' if n_better >= 10 else 'FAIL'}")
        w(f"  C3 DESCRIPTIVE median final improvement = {fin.improvement.median():.4f}x")
        w("")
        w(f"  {'case':<26}{'tau':>7}{'closed %':>12}{'open %':>12}{'improve':>10}")
        for _, r in fin.sort_values("improvement", ascending=False).iterrows():
            w(f"  {r['case']:<26}{r['tau']:>7.2f}"
              f"{100*r['closed_loop_P_error']:>12.4f}"
              f"{100*r['open_loop_P_error']:>12.4f}{r['improvement']:>10.3f}x")

    w("")
    w("Outputs:")
    for f in (f"m3d_Z_{tag}_alignment.csv", f"m3d_Z_{tag}_closedloop.csv",
              f"m3d_Z_{tag}_case_summary.csv", f"m3d_Z_{tag}_COMPLETE.txt"):
        w(f"  {SAVE_DIR / f}")
    txt = "\n".join(L)
    print(txt)
    open(SAVE_DIR / f"m3d_Z_{tag}_COMPLETE.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--precision", choices=["fp32", "fp64"], default="fp64")
    ap.add_argument("--jvp", choices=["fd", "analytic"], default="analytic")
    ap.add_argument("--deriv-rel", type=float, default=1e-5)
    ap.add_argument("--tag", default="z")
    args = ap.parse_args()
    if args.worker:
        if args.gpu not in (0, 1):
            raise SystemExit("--gpu must be 0 or 1")
        worker_main(args.gpu, args.precision, args.jvp, args.deriv_rel,
                    False, args.tag)
    else:
        parent_main(args.precision, args.jvp, args.deriv_rel, False, args.tag)
