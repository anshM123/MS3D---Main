# ============================================================================
# M3D TEST C — COARSE-GRID DEFECT ESTIMATION
# ============================================================================
# The cost objection: W needs Phi_h, J_k and H_k at full resolution every
# segment, which costs ~3x the full solver. But the leakage analysis showed
# ~60% of the defect's energy sits at |k| <= 5 (inside the P band) and 89%
# within |k| <= 10. If that is right, a COARSE flow map should reproduce the
# defect well enough to correct with.
#
# Everything about the ROM and the intervention is unchanged. Only the
# resolution at which Phi_h, J and H are evaluated changes.
#
#   eta_M = Prolong_M->N [ Phi_h^M ( Restrict_N->M V_k ) ] - V_{k+1}
#
# Gates (fixed before running):
#   C1  coarse-eta Q-residual vs full-eta < 0.25 for at least one M < N/2
#   C2  the coarse-derived correction still beats baseline (gain > 1) 
#   C3  DESCRIPTIVE: gain(M) vs cost ratio (M/N)^3
# ============================================================================
import os, glob, math
import torch, numpy as np, pandas as pd
from pathlib import Path

SAVE = Path("/kaggle/working/M3D"); SAVE.mkdir(parents=True, exist_ok=True)
BOOT = glob.glob("/kaggle/input/**/m3d_kaggle_bootstrap.py", recursive=True)[0]
print("bootstrap:", BOOT)
G = {"__name__": "__m3d__"}
exec(compile(open(BOOT).read(), BOOT, "exec"), G, G)

def spaces():
    out, seen = [G], {id(G)}
    for _, o in list(G.items()):
        if callable(o) and hasattr(o, "__globals__") and id(o.__globals__) not in seen:
            out.append(o.__globals__); seen.add(id(o.__globals__))
    return out

def setup(nu):
    for ns in spaces():
        ns["NU1_D"] = ns["NU2_D"] = ns["nu1_D"] = ns["nu2_D"] = float(nu)
        ns["REAL_DTYPE_D"] = torch.float64
        ns["COMPLEX_DTYPE_D"] = torch.complex128

make_env = G["make_env_D"];   topo  = G["topology_pair_L2"]
full_rk4 = G["full_rk4_D"];   init2 = G["initialize_star2_N"]
rk4M     = G["rk4_M"];        recon = G["reconstruct_M"]
packm    = G["pack_modes_D"]; TAU0  = float(G["TAU0_D"])
dealias  = G["dealias_D"];    leray = G["project_divfree_D"]

L_BOX, SEG_TAU, SEG_STEPS = 8.0, 0.15, 20
DT = SEG_TAU * TAU0 / SEG_STEPS
N_FINE = int(os.environ.get("M3D_C_NFINE", 80))
COARSE = [int(x) for x in os.environ.get("M3D_C_COARSE", "20,32,40,56,80").split(",")]
TAU_INT = 1.20
FORECAST_SEGS = 2
DERIV_REL = 1e-5
CASES = [("skew_tubes", 1.0), ("skew_tubes", 0.16)]
BASE_NU = 2.5e-3

# ---------------- spectral restriction / prolongation ----------------
def _idx(n, m):
    """Fourier index map: low-|k| shells of an n-grid -> an m-grid."""
    h = m // 2
    src = list(range(0, h + 1)) + list(range(n - h + 1, n))
    dst = list(range(0, h + 1)) + list(range(m - h + 1, m))
    return src, dst

def restrict(Uh, N, M):
    """Spectral truncation N -> M. For even M the index list d is exactly
    0..M-1, so a plain index_select on each axis gives the right layout."""
    if M == N:
        return Uh.clone()
    s, _ = _idx(N, M)
    si = torch.tensor(s, device=Uh.device, dtype=torch.long)
    A = (Uh.index_select(1, si).index_select(2, si).index_select(3, si))
    return A * (float(M) / float(N)) ** 3

def prolong(Uh, M, N):
    """Zero-pad M -> N, inverse of restrict up to the truncated content."""
    if M == N:
        return Uh.clone()
    s, _ = _idx(N, M)
    si = torch.tensor(s, device=Uh.device, dtype=torch.long)
    out = torch.zeros((3, N, N, N), dtype=Uh.dtype, device=Uh.device)
    out[:, si[:, None, None], si[None, :, None], si[None, None, :]] = (
        Uh * (float(N) / float(M)) ** 3)
    return out

def phi_at(A1, A2, envC, nu, N, M):
    """One segment of the full solver, evaluated on the M-grid."""
    if M == N:
        B1, B2 = A1.clone(), A2.clone()
    else:
        B1 = dealias(leray(restrict(A1, N, M), envC), envC)
        B2 = dealias(leray(restrict(A2, N, M), envC), envC)
    for _ in range(SEG_STEPS):
        B1 = full_rk4(B1, DT, nu, envC); B2 = full_rk4(B2, DT, nu, envC)
    if M == N:
        return B1, B2
    return prolong(B1, M, N), prolong(B2, M, N)

def pnorm(a1, a2):
    return float(torch.sqrt(torch.clamp(
        torch.real(torch.vdot(a1.reshape(-1), a1.reshape(-1)))
        + torch.real(torch.vdot(a2.reshape(-1), a2.reshape(-1))), min=0)).cpu())

def splitPQ(A1, A2, tids):
    P1 = torch.zeros_like(A1); P2 = torch.zeros_like(A2)
    P1.reshape(3, -1)[:, tids] = A1.reshape(3, -1)[:, tids]
    P2.reshape(3, -1)[:, tids] = A2.reshape(3, -1)[:, tids]
    return P1, P2, A1 - P1, A2 - P2

def terr(A1, A2, T1, T2, tids):
    a = packm(A1 - T1, A2 - T2, tids); b = packm(T1, T2, tids)
    return float((torch.linalg.vector_norm(a) / torch.linalg.vector_norm(b)).real.cpu())

rows = []
for topo_name, numult in CASES:
    nu = BASE_NU * numult
    print(f"\n=== {topo_name} nu={nu:.4e}  N_fine={N_FINE} ===", flush=True)
    setup(nu)
    envF = make_env(N_FINE, L=L_BOX)
    U1, U2 = topo(topo_name, envF)
    star, _ = init2(U1, U2, envF); tids = star["proj"]["target_ids"]; del star

    n_seg = int(round(TAU_INT / SEG_TAU)) + FORECAST_SEGS
    t1, t2 = U1.clone(), U2.clone()
    m, _ = init2(U1, U2, envF); V1, V2 = U1.clone(), U2.clone()
    TR, MO = [(t1.clone(), t2.clone())], [(V1.clone(), V2.clone())]
    for seg in range(n_seg):
        for _ in range(SEG_STEPS):
            t1 = full_rk4(t1, DT, nu, envF); t2 = full_rk4(t2, DT, nu, envF)
            rk4M(m, SEG_TAU / SEG_STEPS, envF)
        V1, V2 = recon(m, envF); m, _ = init2(V1, V2, envF)
        TR.append((t1.clone(), t2.clone())); MO.append((V1.clone(), V2.clone()))

    kint = int(round(TAU_INT / SEG_TAU))
    Vi1, Vi2 = MO[kint]; Ti1, Ti2 = TR[kint]
    TF1, TF2 = TR[kint + FORECAST_SEGS]
    _, _, AQ1, AQ2 = splitPQ(Ti1 - Vi1, Ti2 - Vi2, tids)

    def forecast(S1, S2):
        A1, A2 = S1.clone(), S2.clone()
        for _ in range(FORECAST_SEGS):
            mm, _ = init2(A1, A2, envF)
            for _ in range(SEG_STEPS):
                rk4M(mm, SEG_TAU / SEG_STEPS, envF)
            A1, A2 = recon(mm, envF)
        return A1, A2

    FB1, FB2 = forecast(Vi1, Vi2)
    base_err = terr(FB1, FB2, TF1, TF2, tids)
    EH_REF = None

    for M in COARSE:
        envC = envF if M == N_FINE else make_env(M, L=L_BOX)
        E1 = torch.zeros_like(MO[0][0]); E2 = torch.zeros_like(MO[0][1])
        for k in range(kint):
            Vk1, Vk2 = MO[k]; Vn1, Vn2 = MO[k + 1]
            B1, B2 = phi_at(Vk1, Vk2, envC, nu, N_FINE, M)
            eta1, eta2 = B1 - Vn1, B2 - Vn2
            if pnorm(E1, E2) < 1e-30:
                E1, E2 = eta1.clone(), eta2.clone()
            else:
                en = max(pnorm(E1, E2), 1e-30)
                eps = min(max(DERIV_REL * pnorm(Vk1, Vk2) / en, 1e-6), 5e-2)
                P1, P2 = phi_at(Vk1 + eps*E1, Vk2 + eps*E2, envC, nu, N_FINE, M)
                Mi1, Mi2 = phi_at(Vk1 - eps*E1, Vk2 - eps*E2, envC, nu, N_FINE, M)
                J1 = (P1 - Mi1) / (2*eps); J2 = (P2 - Mi2) / (2*eps)
                H1 = (P1 + Mi1 - 2*B1) / (eps*eps)
                H2 = (P2 + Mi2 - 2*B2) / (eps*eps)
                E1 = eta1 + J1 + 0.5*H1; E2 = eta2 + J2 + 0.5*H2
        _, _, EQ1, EQ2 = splitPQ(E1, E2, tids)
        if M == N_FINE: EH_REF = (EQ1.clone(), EQ2.clone())

        FE1, FE2 = forecast(Vi1 + EQ1, Vi2 + EQ2)
        est_err = terr(FE1, FE2, TF1, TF2, tids)
        q_res = pnorm(EQ1 - AQ1, EQ2 - AQ2) / max(pnorm(AQ1, AQ2), 1e-30)
        vs_ref = (pnorm(EQ1 - EH_REF[0], EQ2 - EH_REF[1])
                  / max(pnorm(*EH_REF), 1e-30)) if EH_REF is not None else float("nan")
        rows.append(dict(topology=topo_name, nu=nu, N_fine=N_FINE, M=M,
                         cost_ratio=(M / N_FINE) ** 3,
                         baseline_P_error=base_err, est_P_error=est_err,
                         gain=base_err / max(est_err, 1e-30),
                         hidden_Q_residual=q_res, coarse_vs_fine_eta=vs_ref))
        print(f"  M={M:>3} cost={(M/N_FINE)**3:6.3f}x  gain={base_err/max(est_err,1e-30):7.4f}x"
              f"  Qres={q_res:.4f}  vs_fine={vs_ref:.4f}", flush=True)
        if M != N_FINE: del envC
    del envF, U1, U2, t1, t2, V1, V2, m, TR, MO
    torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv(SAVE / "m3d_C_coarse_defect.csv", index=False)
print("\n" + "=" * 96)
print("M3D TEST C — COARSE-GRID DEFECT ESTIMATION")
print("=" * 96)
print(df.to_string(index=False))
sub = df[df.M < df.N_fine / 2]
c1 = bool((sub.coarse_vs_fine_eta < 0.25).any())
c2 = bool((sub.gain > 1.0).all()) if not sub.empty else False
print(f"\nC1 coarse eta within 25% of fine, for some M < N/2: {'PASS' if c1 else 'FAIL'}")
print(f"C2 coarse-derived correction still beats baseline:   {'PASS' if c2 else 'FAIL'}")
best = sub[sub.gain > 1.0].sort_values("cost_ratio").head(1)
if not best.empty:
    b = best.iloc[0]
    print(f"C3 cheapest working M = {int(b.M)} at {b.cost_ratio:.3f}x the flow-map cost, "
          f"gain {b.gain:.4f}x (full-resolution gain "
          f"{df[df.M==df.N_fine].gain.iloc[0]:.4f}x)")
print("\nsaved:", SAVE / "m3d_C_coarse_defect.csv")
