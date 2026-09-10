#!/usr/bin/env python3
"""
M3D TEST X-v2 — dual-T4 Kaggle runner with estimator-tolerance sweep

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


def parent_main(precision="fp64", jvp="analytic", deriv_rel=1e-5,
                noise_sweep=True, tag="fp64_analytic"):
    # Do not import torch in the parent before launching children; children set CUDA_VISIBLE_DEVICES first.
    out = Path("/kaggle/working/M3D")
    out.mkdir(parents=True, exist_ok=True)

    script = str(Path(__file__).resolve())
    print("=" * 100)
    print("M3D TEST X-v2 — DUAL GPU LAUNCHER")
    print("=" * 100)
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
        if not noise_sweep:
            cmd.append("--no-noise-sweep")
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

    finalize_outputs(tag, noise_sweep, precision, jvp)


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
    DERIV_REL_X = float(deriv_rel)

    POINT_FILE = SAVE_DIR_X / f"m3d_X2_{tag}_gpu{physical_gpu}_interventions_partial.csv"
    CASE_FILE = SAVE_DIR_X / f"m3d_X2_{tag}_gpu{physical_gpu}_cases_partial.csv"
    SWEEP_FILE = SAVE_DIR_X / f"m3d_X2_{tag}_gpu{physical_gpu}_sweep_partial.csv"

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

        # intervention forecasts at the selected precision
        assert_viscosity_X(nu_X, RDT_X, CDT_X)
        ENV32_X = make_env_D(40, L=8.0)
        case_window_rows = []

        for tau_int_X in INTERVENTION_TAUS_X:
            idx = int(round(tau_int_X / SEGMENT_TAU_X))
            idx_end = idx + FORECAST_SEGMENTS_X
            V1, V2 = MODEL_X[idx]
            T1, T2 = TRUTH_X[idx]
            TF1, TF2 = TRUTH_X[idx_end]
            EH1_64, EH2_64 = EHAT_TRAJ_X[idx]
            _, _, EHQ1_64, EHQ2_64 = split_PQ_X(EH1_64, EH2_64, target_ids_X)
            EHQ1, EHQ2 = EHQ1_64.to(CDT_X), EHQ2_64.to(CDT_X)

            ACT1, ACT2 = T1 - V1, T2 - V2
            _, _, AQ1, AQ2 = split_PQ_X(ACT1, ACT2, target_ids_X)

            BASE1, BASE2 = V1.clone(), V2.clone()
            EST1, EST2 = V1 + EHQ1, V2 + EHQ2
            NEG1, NEG2 = V1 - EHQ1, V2 - EHQ2
            ORA1, ORA2 = V1 + AQ1, V2 + AQ2

            xb = pack_modes_D(BASE1, BASE2, target_ids_X)
            xe = pack_modes_D(EST1, EST2, target_ids_X)
            xn = pack_modes_D(NEG1, NEG2, target_ids_X)
            xo = pack_modes_D(ORA1, ORA2, target_ids_X)
            base_target_norm = vec_norm_X(xb) + 1e-30
            start_mismatch = max(vec_norm_X(xe-xb), vec_norm_X(xn-xb), vec_norm_X(xo-xb)) / base_target_norm

            FB1, FB2 = star2_forecast_X(BASE1, BASE2, ENV32_X)
            FE1, FE2 = star2_forecast_X(EST1, EST2, ENV32_X)
            FN1, FN2 = star2_forecast_X(NEG1, NEG2, ENV32_X)
            FO1, FO2 = star2_forecast_X(ORA1, ORA2, ENV32_X)

            base_err = target_error_X(FB1, FB2, TF1, TF2, target_ids_X)
            est_err = target_error_X(FE1, FE2, TF1, TF2, target_ids_X)
            neg_err = target_error_X(FN1, FN2, TF1, TF2, target_ids_X)
            ora_err = target_error_X(FO1, FO2, TF1, TF2, target_ids_X)

            gain_est = base_err / max(est_err, 1e-30)
            gain_neg = base_err / max(neg_err, 1e-30)
            gain_ora = base_err / max(ora_err, 1e-30)
            red_est = (base_err-est_err) / max(base_err, 1e-30)
            red_neg = (base_err-neg_err) / max(base_err, 1e-30)
            red_ora = (base_err-ora_err) / max(base_err, 1e-30)
            oracle_capture = red_est / red_ora if red_ora > 0.05 else np.nan

            row = {
                "case": case_X, "topology": topology_X, "nu_mult": nu_mult_X,
                "tau_intervention": tau_int_X, "tau_endpoint": tau_int_X + FORECAST_TAU_X,
                "start_target_mismatch": start_mismatch,
                "estimated_Q_norm": pair_norm_X(EHQ1, EHQ2), "oracle_Q_norm": pair_norm_X(AQ1, AQ2),
                "baseline_P_error": base_err, "estimated_Q_P_error": est_err,
                "negative_Q_P_error": neg_err, "oracle_Q_P_error": ora_err,
                "estimated_gain": gain_est, "negative_gain": gain_neg, "oracle_gain": gain_ora,
                "estimated_reduction": red_est, "negative_reduction": red_neg,
                "oracle_reduction": red_ora, "oracle_capture": oracle_capture,
                "estimated_beats_baseline": bool(est_err < base_err),
                "estimated_beats_negative": bool(est_err < neg_err),
                "oracle_beats_baseline": bool(ora_err < base_err),
                # provenance sentinels, interrogated live
                "sent_precision": precision,
                "sent_real_dtype": str(globals()["REAL_DTYPE_D"]),
                "sent_complex_dtype": str(globals()["COMPLEX_DTYPE_D"]),
                "sent_NU1_D": float(globals()["NU1_D"]),
                "sent_nu_expected": float(nu_X),
                "sent_jvp": jvp,
                "sent_deriv_rel": DERIV_REL_X,
            }
            ROWS_X.append(row)
            case_window_rows.append(row)

            # ---------------- estimator-tolerance sweep ----------------
            # Degrade q_hat on purpose and find where the benefit dies.
            # Every perturbation is projected back into range(Q), so
            # P(V + c) = PV still holds exactly for every branch.
            if noise_sweep:
                ehq_norm = pair_norm_X(EHQ1, EHQ2)
                for family in NOISE_FAMILIES_X:
                    _deltas = (NOISE_DELTAS_SCALE_X if family == "scale"
                               else NOISE_DELTAS_DIR_X)
                    for delta in _deltas:
                        if ehq_norm < 1e-30:
                            continue
                        if family == "scale":
                            PT1 = (1.0 + delta) * EHQ1
                            PT2 = (1.0 + delta) * EHQ2
                        else:
                            # deterministic per (case, tau, delta) for reproducibility
                            # NOTE: never use hash() here — Python salts str
                            # hashing per process, which makes the direction
                            # family irreproducible across runs.
                            _key = (f"{case_X}|{tau_int_X:.4f}|{delta:.6f}"
                                    ).encode("utf-8")
                            seed = zlib.crc32(_key) & 0x7FFFFFFF
                            gen = torch.Generator(device="cpu").manual_seed(seed)
                            def _rnd(like):
                                re = torch.randn(like.shape, generator=gen,
                                                 dtype=torch.float64)
                                im = torch.randn(like.shape, generator=gen,
                                                 dtype=torch.float64)
                                return torch.complex(re, im).to(like.device).to(like.dtype)
                            G1, G2 = _rnd(EHQ1), _rnd(EHQ2)
                            _, _, G1, G2 = split_PQ_X(G1, G2, target_ids_X)
                            gn = pair_norm_X(G1, G2)
                            if gn < 1e-30:
                                continue
                            scale = delta * ehq_norm / gn
                            PT1 = EHQ1 + scale * G1
                            PT2 = EHQ2 + scale * G2
                        # enforce range(Q) after arithmetic
                        _, _, PT1, PT2 = split_PQ_X(PT1, PT2, target_ids_X)

                        PB1, PB2 = V1 + PT1, V2 + PT2
                        xp = pack_modes_D(PB1, PB2, target_ids_X)
                        pert_mismatch = vec_norm_X(xp - xb) / base_target_norm

                        FP1, FP2 = star2_forecast_X(PB1, PB2, ENV32_X)
                        pert_err = target_error_X(FP1, FP2, TF1, TF2, target_ids_X)
                        rel_est_err = pair_norm_X(PT1 - EHQ1, PT2 - EHQ2) / ehq_norm

                        SWEEP_ROWS_X.append({
                            "case": case_X, "topology": topology_X,
                            "nu_mult": nu_mult_X, "tau_intervention": tau_int_X,
                            "family": family, "delta": delta,
                            "alpha": (1.0 + delta) if family == "scale" else 1.0,
                            "realized_rel_estimator_error": rel_est_err,
                            "start_target_mismatch": pert_mismatch,
                            "baseline_P_error": base_err,
                            "perturbed_P_error": pert_err,
                            "unperturbed_est_P_error": est_err,
                            "perturbed_gain": base_err / max(pert_err, 1e-30),
                            "unperturbed_gain": gain_est,
                            "still_beats_baseline": bool(pert_err < base_err),
                            "sent_precision": precision, "sent_jvp": jvp,
                        })
                _msg = "  ".join(
                    f"{r['family'][:3]}/{r['delta']:g}={r['perturbed_gain']:.2f}x"
                    for r in SWEEP_ROWS_X
                    if r["case"] == case_X and r["tau_intervention"] == tau_int_X
                )
                print(f"[GPU{physical_gpu}]   sweep tau={tau_int_X:.2f}: {_msg}",
                      flush=True)
            print(
                f"[GPU{physical_gpu}] {case_X} tau={tau_int_X:.2f}: "
                f"BASE={100*base_err:.5f}% +EST={100*est_err:.5f}% "
                f"-EST={100*neg_err:.5f}% ORACLE={100*ora_err:.5f}% gain={gain_est:.3f}x",
                flush=True,
            )

        cdf = pd.DataFrame(case_window_rows)
        CASE_ROWS_X.append({
            "case": case_X, "topology": topology_X, "nu_mult": nu_mult_X,
            "median_estimated_gain": float(cdf["estimated_gain"].median()),
            "median_negative_gain": float(cdf["negative_gain"].median()),
            "median_oracle_gain": float(cdf["oracle_gain"].median()),
            "estimated_win_rate": float(cdf["estimated_beats_baseline"].mean()),
            "direction_win_rate": float(cdf["estimated_beats_negative"].mean()),
            "oracle_win_rate": float(cdf["oracle_beats_baseline"].mean()),
        })

        pd.DataFrame(ROWS_X).to_csv(POINT_FILE, index=False)
        pd.DataFrame(CASE_ROWS_X).to_csv(CASE_FILE, index=False)
        if SWEEP_ROWS_X:
            pd.DataFrame(SWEEP_ROWS_X).to_csv(SWEEP_FILE, index=False)
        COMPLETED_X.add(case_X)

        del MODEL_X, TRUTH_X, MODEL64_X, EHAT_TRAJ_X, model_X
        gc.collect(); torch.cuda.empty_cache()

    print(f"[GPU{physical_gpu}] shard complete ✅", flush=True)


def finalize_outputs(tag="fp64_analytic", noise_sweep=True,
                     precision="fp64", jvp="analytic"):
    import numpy as np
    import pandas as pd

    SAVE_DIR = Path("/kaggle/working/M3D")
    pfiles = [SAVE_DIR / f"m3d_X2_{tag}_gpu{g}_interventions_partial.csv" for g in (0, 1)]
    cfiles = [SAVE_DIR / f"m3d_X2_{tag}_gpu{g}_cases_partial.csv" for g in (0, 1)]
    sfiles = [SAVE_DIR / f"m3d_X2_{tag}_gpu{g}_sweep_partial.csv" for g in (0, 1)]
    for f in pfiles + cfiles:
        if not f.exists():
            raise RuntimeError(f"Missing shard output: {f}")

    RESULTS_X = pd.concat([pd.read_csv(f) for f in pfiles], ignore_index=True)
    CASES_X = pd.concat([pd.read_csv(f) for f in cfiles], ignore_index=True)
    RESULTS_X = RESULTS_X.sort_values(["topology", "nu_mult", "tau_intervention"]).reset_index(drop=True)
    CASES_X = CASES_X.sort_values(["topology", "nu_mult"]).reset_index(drop=True)

    if RESULTS_X["case"].nunique() != 12 or len(RESULTS_X) != 36:
        raise RuntimeError(f"Incomplete merge: {RESULTS_X['case'].nunique()} cases, {len(RESULTS_X)} windows")

    RESULTS_X.to_csv(SAVE_DIR / f"m3d_X2_{tag}_causal_interventions.csv", index=False)
    CASES_X.to_csv(SAVE_DIR / f"m3d_X2_{tag}_case_summary.csv", index=False)

    # ---- G1 provenance: sentinels must be present AND correct on every row ----
    SENT_OK_X, SENT_NOTES_X = True, []
    if "sent_NU1_D" in RESULTS_X.columns:
        nu_ok = np.allclose(RESULTS_X["sent_NU1_D"].values,
                            RESULTS_X["sent_nu_expected"].values, atol=1e-14)
        if not nu_ok:
            SENT_NOTES_X.append("viscosity sentinel != requested nu")
        # Key off the ACTUAL requested precision, never the free-text tag.
        want = "float64" if precision == "fp64" else "float32"
        dt_ok = RESULTS_X["sent_real_dtype"].astype(str).str.contains(want).all()
        if not dt_ok:
            SENT_NOTES_X.append(
                f"real dtype != {want} (saw "
                f"{sorted(RESULTS_X['sent_real_dtype'].astype(str).unique())})")
        # Resumed shards can silently mix configurations. Every row must agree.
        uni_ok = True
        for col, want_val in (("sent_precision", precision), ("sent_jvp", jvp)):
            if col in RESULTS_X.columns:
                vals = sorted(RESULTS_X[col].astype(str).unique())
                if len(vals) != 1 or vals[0] != str(want_val):
                    uni_ok = False
                    SENT_NOTES_X.append(
                        f"{col} not uniformly {want_val}: saw {vals} "
                        f"— checkpoints may be from a different config")
        SENT_OK_X = bool(nu_ok and dt_ok and uni_ok)

    MAX_START_MISMATCH_X = float(RESULTS_X["start_target_mismatch"].max())
    MEDIAN_EST_GAIN_X = float(RESULTS_X["estimated_gain"].median())
    P10_EST_GAIN_X = float(RESULTS_X["estimated_gain"].quantile(0.10))
    MEDIAN_NEG_GAIN_X = float(RESULTS_X["negative_gain"].median())
    MEDIAN_ORACLE_GAIN_X = float(RESULTS_X["oracle_gain"].median())
    EST_WIN_RATE_X = float(RESULTS_X["estimated_beats_baseline"].astype(float).mean())
    DIRECTION_WIN_RATE_X = float(RESULTS_X["estimated_beats_negative"].astype(float).mean())
    ORACLE_WIN_RATE_X = float(RESULTS_X["oracle_beats_baseline"].astype(float).mean())
    MEDIAN_EST_REDUCTION_X = float(RESULTS_X["estimated_reduction"].median())
    MEDIAN_ORACLE_REDUCTION_X = float(RESULTS_X["oracle_reduction"].median())

    ACTIONABLE_X = RESULTS_X[RESULTS_X["oracle_reduction"] > 0.05].copy()
    N_ACTIONABLE_X = len(ACTIONABLE_X)
    if N_ACTIONABLE_X:
        ACTIONABLE_EST_WIN_X = float(ACTIONABLE_X["estimated_beats_baseline"].astype(float).mean())
        ACTIONABLE_DIRECTION_WIN_X = float(ACTIONABLE_X["estimated_beats_negative"].astype(float).mean())
        MEDIAN_ORACLE_CAPTURE_X = float(ACTIONABLE_X["oracle_capture"].median())
        MEDIAN_ACTIONABLE_GAIN_X = float(ACTIONABLE_X["estimated_gain"].median())
    else:
        ACTIONABLE_EST_WIN_X = ACTIONABLE_DIRECTION_WIN_X = MEDIAN_ORACLE_CAPTURE_X = MEDIAN_ACTIONABLE_GAIN_X = np.nan

    CASE_POSITIVE_RATE_X = float((CASES_X["median_estimated_gain"] > 1.0).mean())

    NUMERICS_GATE_X = bool(MAX_START_MISMATCH_X < 1e-6)
    ORACLE_CAUSAL_GATE_X = bool(ORACLE_WIN_RATE_X >= 0.75 and N_ACTIONABLE_X >= 12)
    ESTIMATED_INTERVENTION_GATE_X = bool(EST_WIN_RATE_X >= 0.75 and MEDIAN_EST_GAIN_X >= 1.10 and P10_EST_GAIN_X >= 0.90)
    DIRECTIONAL_CAUSALITY_GATE_X = bool(DIRECTION_WIN_RATE_X >= 0.85 and MEDIAN_EST_GAIN_X > MEDIAN_NEG_GAIN_X)
    ORACLE_CAPTURE_GATE_X = bool(
        N_ACTIONABLE_X >= 12 and ACTIONABLE_EST_WIN_X >= 0.80 and
        ACTIONABLE_DIRECTION_WIN_X >= 0.90 and MEDIAN_ORACLE_CAPTURE_X >= 0.70
    )
    GENERALIZATION_GATE_X = bool(CASE_POSITIVE_RATE_X >= 0.75)
    CAUSAL_HIDDEN_CORRECTION_X = bool(
        NUMERICS_GATE_X and ORACLE_CAUSAL_GATE_X and ESTIMATED_INTERVENTION_GATE_X and
        DIRECTIONAL_CAUSALITY_GATE_X and ORACLE_CAPTURE_GATE_X and GENERALIZATION_GATE_X
    )

    print("\n" + "=" * 156)
    print("M3D TEST X — DUAL-GPU FINAL DIAGNOSIS")
    print("=" * 156)
    print(f"cases: {RESULTS_X['case'].nunique()}")
    print(f"intervention windows: {len(RESULTS_X)}")
    print(f"maximum intervention-start P mismatch: {MAX_START_MISMATCH_X:.3e}")
    print()
    print(f"median EST-Q forecast gain: {MEDIAN_EST_GAIN_X:.6f}x")
    print(f"10th percentile EST-Q gain: {P10_EST_GAIN_X:.6f}x")
    print(f"EST-Q win rate vs baseline: {100*EST_WIN_RATE_X:.3f}%")
    print(f"median wrong-sign gain: {MEDIAN_NEG_GAIN_X:.6f}x")
    print(f"EST-Q beats wrong-sign rate: {100*DIRECTION_WIN_RATE_X:.3f}%")
    print(f"median oracle-Q gain: {MEDIAN_ORACLE_GAIN_X:.6f}x")
    print(f"oracle-Q win rate: {100*ORACLE_WIN_RATE_X:.3f}%")
    print(f"median EST-Q relative error reduction: {100*MEDIAN_EST_REDUCTION_X:.3f}%")
    print(f"median oracle-Q relative error reduction: {100*MEDIAN_ORACLE_REDUCTION_X:.3f}%")
    print()
    print(f"oracle-actionable windows: {N_ACTIONABLE_X}")
    print(f"actionable EST-Q win rate: {100*ACTIONABLE_EST_WIN_X:.3f}%" if np.isfinite(ACTIONABLE_EST_WIN_X) else "actionable EST-Q win rate: nan")
    print(f"actionable EST-Q vs wrong-sign win rate: {100*ACTIONABLE_DIRECTION_WIN_X:.3f}%" if np.isfinite(ACTIONABLE_DIRECTION_WIN_X) else "actionable EST-Q vs wrong-sign win rate: nan")
    print(f"median fraction of oracle benefit captured: {100*MEDIAN_ORACLE_CAPTURE_X:.3f}%" if np.isfinite(MEDIAN_ORACLE_CAPTURE_X) else "median fraction of oracle benefit captured: nan")
    print(f"median actionable EST-Q gain: {MEDIAN_ACTIONABLE_GAIN_X:.6f}x" if np.isfinite(MEDIAN_ACTIONABLE_GAIN_X) else "median actionable EST-Q gain: nan")
    print(f"fraction cases with positive median EST-Q gain: {100*CASE_POSITIVE_RATE_X:.3f}%")
    print()
    print("numerics / equal-P-start gate:", NUMERICS_GATE_X)
    print("oracle hidden-causality gate:", ORACLE_CAUSAL_GATE_X)
    print("truth-free intervention gate:", ESTIMATED_INTERVENTION_GATE_X)
    print("directional +Q vs -Q gate:", DIRECTIONAL_CAUSALITY_GATE_X)
    print("oracle-benefit capture gate:", ORACLE_CAPTURE_GATE_X)
    print("cross-case generalization gate:", GENERALIZATION_GATE_X)
    print()
    print("CAUSAL HIDDEN-ERROR CORRECTION:", CAUSAL_HIDDEN_CORRECTION_X)
    print("G1 per-row sentinels consistent AND correct:", SENT_OK_X)
    for _n in SENT_NOTES_X:
        print("   G1 note:", _n)

    # ================================================================
    # ESTIMATOR-TOLERANCE SWEEP
    # ================================================================
    SWEEP_SUMMARY = {}
    if noise_sweep and all(f.exists() for f in sfiles):
        SW = pd.concat([pd.read_csv(f) for f in sfiles], ignore_index=True)
        SW.to_csv(SAVE_DIR / f"m3d_X2_{tag}_estimator_sweep.csv", index=False)

        print("\n" + "=" * 156)
        print("M3D TEST X-v2 — ESTIMATOR-TOLERANCE SWEEP")
        print("=" * 156)

        N0 = bool(SW["start_target_mismatch"].max() < 1e-6)
        print(f"N0 perturbations stayed in range(Q): {N0}"
              f"  (max start-P mismatch {SW['start_target_mismatch'].max():.3e})")

        print(f"\nunperturbed median gain: {MEDIAN_EST_GAIN_X:.4f}x")
        print(f"{'family':<11}{'delta':>8}{'median gain':>14}"
              f"{'p10 gain':>11}{'win rate':>11}{'n':>6}")
        n1_ok, n2_ok = True, True
        for fam in sorted(SW["family"].unique()):
            sub_f = SW[SW.family == fam]
            meds, dels = [], []
            for d in sorted(sub_f["delta"].unique()):
                q = sub_f[sub_f.delta == d]
                med = float(q["perturbed_gain"].median())
                p10 = float(q["perturbed_gain"].quantile(0.10))
                wr = float(q["still_beats_baseline"].astype(float).mean())
                meds.append(med); dels.append(d)
                print(f"{fam:<11}{d:>8.2f}{med:>14.4f}{p10:>11.4f}"
                      f"{100*wr:>10.1f}%{len(q):>6}")
                if abs(d - 0.10) < 1e-9 and med <= 1.0:
                    n2_ok = False
            if len(meds) >= 3:
                r = np.corrcoef(np.argsort(np.argsort(dels)),
                                np.argsort(np.argsort(meds)))[0, 1]
                print(f"{'':<11}Spearman(delta, median gain) = {r:+.4f}")
                # N1 is preregistered for the DIRECTION family only. The scale
                # family measures amplitude of a correction whose optimum need
                # not sit at alpha=1 (Theorem C), so a positive slope there is
                # a result, not a gate failure.
                if fam == "direction" and r >= 0:
                    n1_ok = False
            SWEEP_SUMMARY[fam] = dict(zip(dels, meds))

        print(f"\nN1 gain non-increasing in delta (DIRECTION family): {n1_ok}")
        print(f"N2 median gain > 1.0 at delta=0.10 (both families): {n2_ok}")
        print("\nN3 DESCRIPTIVE — delta at which median gain crosses 1.0:")
        for fam, curve in SWEEP_SUMMARY.items():
            ds = sorted(curve)
            cross = next((d for d in ds if curve[d] <= 1.0), None)
            print(f"  {fam:<11}"
                  + ("no crossing within tested range (>%.2f)" % max(ds)
                     if cross is None else "crosses between %.2f and %.2f"
                     % (max([d for d in ds if curve[d] > 1.0], default=0.0), cross)))
        print("\n  W's measured median hidden-Q residual was ~1.1e-4, so a delta")
        print("  of 0.10 is roughly 1000x worse than the deployed estimator.")

        # ---- optimal correction amplitude (Theorem C) ----
        if "scale" in SWEEP_SUMMARY:
            curve = dict(SWEEP_SUMMARY["scale"])
            curve[0.0] = MEDIAN_EST_GAIN_X          # alpha = 1 is delta = 0
            best_d = max(curve, key=lambda k: curve[k])
            print("\n--- optimal correction amplitude ---")
            print(f"{'alpha':>8}{'median gain':>14}")
            for d in sorted(curve):
                mark = "  <-- max" if d == best_d else ""
                print(f"{1.0 + d:>8.2f}{curve[d]:>14.4f}{mark}")
            print(f"  argmax alpha = {1.0 + best_d:.2f}  "
                  f"(gain {curve[best_d]:.4f}x vs {MEDIAN_EST_GAIN_X:.4f}x at alpha=1)")
            if best_d == max(curve):
                print("  WARNING: optimum is at the grid edge — extend "
                      "NOISE_DELTAS_SCALE_X and rerun.")
            print("  Zeroing the hidden error and minimizing future resolved")
            print("  error are different objectives; alpha* != 1 is expected.")

    with open(SAVE_DIR / f"m3d_X2_{tag}_COMPLETE.txt", "w") as f:
        f.write(
            "M3D TEST X DUAL-GPU COMPLETE\n"
            f"windows={len(RESULTS_X)}\n"
            f"max_start_mismatch={MAX_START_MISMATCH_X}\n"
            f"median_est_gain={MEDIAN_EST_GAIN_X}\n"
            f"p10_est_gain={P10_EST_GAIN_X}\n"
            f"est_win_rate={EST_WIN_RATE_X}\n"
            f"median_neg_gain={MEDIAN_NEG_GAIN_X}\n"
            f"direction_win_rate={DIRECTION_WIN_RATE_X}\n"
            f"median_oracle_gain={MEDIAN_ORACLE_GAIN_X}\n"
            f"oracle_win_rate={ORACLE_WIN_RATE_X}\n"
            f"actionable_windows={N_ACTIONABLE_X}\n"
            f"median_oracle_capture={MEDIAN_ORACLE_CAPTURE_X}\n"
            f"case_positive_rate={CASE_POSITIVE_RATE_X}\n"
            f"numerics={NUMERICS_GATE_X}\n"
            f"oracle_causal={ORACLE_CAUSAL_GATE_X}\n"
            f"estimated_intervention={ESTIMATED_INTERVENTION_GATE_X}\n"
            f"directionality={DIRECTIONAL_CAUSALITY_GATE_X}\n"
            f"oracle_capture={ORACLE_CAPTURE_GATE_X}\n"
            f"generalization={GENERALIZATION_GATE_X}\n"
            f"causal_hidden_correction={CAUSAL_HIDDEN_CORRECTION_X}\n"
            f"sentinels_ok={SENT_OK_X}\n"
            f"tag={tag}\n"
            f"sweep_summary={SWEEP_SUMMARY}\n"
        )

    print("\nOutputs:")
    print(SAVE_DIR / f"m3d_X2_{tag}_causal_interventions.csv")
    print(SAVE_DIR / f"m3d_X2_{tag}_case_summary.csv")
    if noise_sweep:
        print(SAVE_DIR / f"m3d_X2_{tag}_estimator_sweep.csv")
    print(SAVE_DIR / f"m3d_X2_{tag}_COMPLETE.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--precision", choices=["fp32", "fp64"], default="fp64")
    ap.add_argument("--jvp", choices=["fd", "analytic"], default="analytic")
    ap.add_argument("--deriv-rel", type=float, default=1e-5)
    ap.add_argument("--noise-sweep", action="store_true", default=True)
    ap.add_argument("--no-noise-sweep", dest="noise_sweep", action="store_false")
    ap.add_argument("--tag", default=None,
                    help="output suffix; defaults to <precision>_<jvp>")
    args = ap.parse_args()
    tag = args.tag or f"{args.precision}_{args.jvp}"
    if args.worker:
        if args.gpu not in (0, 1):
            raise SystemExit("--gpu must be 0 or 1")
        worker_main(args.gpu, args.precision, args.jvp, args.deriv_rel,
                    args.noise_sweep, tag)
    else:
        parent_main(args.precision, args.jvp, args.deriv_rel,
                    args.noise_sweep, tag)
